from abc import abstractmethod
from functools import lru_cache

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from x_transformers.x_transformers import Rezero
from einops import rearrange
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding, broadcat
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    silu,
    adagn_silu,
    adagn_silu_extended_32_8,
    adagn_silu_extended_32_6,
    adagn_silu_extended_32_1,
    AdaGN,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
    expanded_timestep_embedding,
    scale_module,
    AxialPositionalEmbeddingShape,
)
import os
import re
import html
import urllib.parse as ul
import torch
from transformers import T5EncoderModel, AutoTokenizer
from huggingface_hub import hf_hub_download
from .text_nn import TextEncoder, CrossAttention, WeaveAttention

import clip
from transformer_utils.partial_forward import *

def partial_forward(
                    model,
                    output_names,
                    *args,
                    verbose=False,
                    debug=False,
                    unsafe=False,
                    **kwargs,
                    ):
        vprint = make_print_if_verbose(verbose)

        if (not unsafe) or (not hasattr(model, "_output_sink_names")):
            add_partial_forward_hooks(
                model, verbose=verbose, debug=debug, output_names=output_names
            )

        for k in model._partial_forward_force_false_kwargs:
            if kwargs.get(k):
                warnings.warn(PARTIAL_FORWARD_FORCE_FALSE_KWARGS_MSG.format(kwarg=repr(k)))
            kwargs[k] = False

        model._output_sink_names = output_names

        if hasattr(model, "_output_sink"):
            vprint("clearing existing _output_sink")
            for v in model._output_sink.values():
                del v
            del model._output_sink

        model._output_sink = {}

        try:
            model(*args, **kwargs)
        except AfterStoppingPointException as e:
            pass

        if not unsafe:
            del model._output_sink_names

        return_val = model._output_sink
        del model._output_sink

        return return_val

class TextEncoder:
    def __init__(self):
        self.visual = None

class T5Model:
    available_models = ['t5-v1_1-xxl']

    def __init__(self, name='t5-v1_1-xxl', *, cache_dir=None, hf_token=None, use_text_preprocessing=True,
                 t5_model_kwargs=None, torch_dtype=None, use_offload_folder=None, freeze_capt_encoder = True):
        self.device = torch.device('cuda')
        self.capt_embd_dim = None
        if name == 't5-v1_1-xxl':
            self.capt_embd_dim = 4096
        self.torch_dtype = torch_dtype or torch.bfloat16
        if t5_model_kwargs is None:
            t5_model_kwargs = {'low_cpu_mem_usage': True, 'torch_dtype': self.torch_dtype}
            t5_model_kwargs['device_map'] = {'shared': self.device, 'encoder': self.device}

        self.hf_token = hf_token
        self.cache_dir = cache_dir or os.path.expanduser('~/.cache/t5_encoders/')
        self.name = name

        tokenizer_path, path = name, name
        if name in self.available_models:
            cache_dir = os.path.join(self.cache_dir, name)
            for filename in [
                'config.json', 'special_tokens_map.json', 'spiece.model', 'tokenizer_config.json',
                'pytorch_model.bin.index.json', 'pytorch_model-00001-of-00002.bin', 'pytorch_model-00002-of-00002.bin'
            ]:
                hf_hub_download(repo_id=f'DeepFloyd/{name}', filename=filename, cache_dir=cache_dir,
                                force_filename=filename, token=self.hf_token)
            tokenizer_path, path = cache_dir, cache_dir
        else:
            cache_dir = os.path.join(self.cache_dir, 't5-v1_1-xxl')
            for filename in [
                'config.json', 'special_tokens_map.json', 'spiece.model', 'tokenizer_config.json',
            ]:
                hf_hub_download(repo_id='DeepFloyd/t5-v1_1-xxl', filename=filename, cache_dir=cache_dir,
                                force_filename=filename, token=self.hf_token)
            tokenizer_path = cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = T5EncoderModel.from_pretrained(path, **t5_model_kwargs).eval()
        if freeze_capt_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

    def tokenize(self, texts):
        
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=77,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        return text_tokens_and_mask['input_ids'], text_tokens_and_mask['attention_mask']

    def encode(self, texts, dtype = th.float32, out_format = 'ndl'):        
        input_ids, att_masks = self.tokenize(texts)
        with torch.no_grad():
            text_encoder_embs = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=att_masks.to(self.device),
            )['last_hidden_state'].detach()
        if out_format == 'ndl':
            text_encoder_embs = text_encoder_embs.permute(0, 2, 1)    
        return text_encoder_embs.to(dtype), att_masks.type(dtype)

            
class ClipModel:        
    def __init__(self, clipname="", clip_use_penultimate_layer = False, freeze_capt_encoder = False, glide_style_capt_attn = False):
        clipmod, _ = clip.load(name=clipname)
        del clipmod.visual
        self.clipmod = clipmod
        if not freeze_capt_encoder:
            self.clipmod.float()

        self.clipmod.positional_embedding = clipmod.positional_embedding
        self.clipmod.transformer = clipmod.transformer
        self.capt_embd_dim = clipmod.ln_final.weight.shape[0]
        self.clip_use_penultimate_layer = clip_use_penultimate_layer
        if self.clip_use_penultimate_layer:
            self.capt_ln_final = nn.LayerNorm(self.capt_embd_dim)
        else:
            self.capt_ln_final = clipmod.ln_final
        
        

    def tokenize(self, batch, truncate = True):
        return self.clipmod.tokenize(batch, truncate=truncate)
    
    def encode(self, toks, dtype=th.float32, out_format='nld'):
        clip_dtype = self.clipmod.transformer.resblocks[0].attn.out_proj.weight.dtype
        x = self.clipmod.token_embedding(toks).type(clip_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clipmod.positional_embedding.type(clip_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        if self.clip_use_penultimate_layer:
            # from imagen paper - note that cos sims between layer outputs get way lower in the final one
            out_name = 'resblocks.' + str(len(self.clipmod.transformer.resblocks) - 2)
            x = partial_forward(self.clipmod.transformer, [out_name], x, unsafe=True)[out_name]
        else:
            x = self.clipmod.transformer(x)

        if self.capt_ln_final is not None:
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = x.type(dtype)
            x = self.capt_ln_final(x)
            if out_format == 'ndl':
                x = x.permute(0, 2, 1)  # NLD -> NDL
        else:
            if out_format == 'nld':
                x = x.permute(1, 0, 2)  # LND -> NLD
            elif out_format == 'ndl':
                x = x.permute(1, 2, 0)  # LND -> NDL
            else:
                raise ValueError(out_format)
        # x = ln_final(x)
        x = x.type(dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]

        return x


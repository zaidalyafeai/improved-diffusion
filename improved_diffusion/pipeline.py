import argparse
import os
from typing import Union, List, Optional

import numpy as np
import torch as th
import torch.nn as nn
import torch.distributed as dist
import blobfile as bf

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_config_to_model
)
from improved_diffusion.image_datasets import load_tokenizer, tokenize

from improved_diffusion.unet import UNetModel
from improved_diffusion.respace import SpacedDiffusion



class SamplingModel(nn.Module):
    def __init__(self, model: UNetModel, diffusion: SpacedDiffusion, tokenizer, is_super_res=False):
        super().__init__()
        self.model = model
        self.diffusion = diffusion  # TODO: allow changing spacing w/o reloading model
        self.tokenizer = tokenizer
        self.is_super_res = is_super_res

    @staticmethod
    def from_config(checkpoint_path, config_path, timestep_respacing=""):
        model, diffusion, tokenizer, is_super_res = load_config_to_model(config_path, timestep_respacing=timestep_respacing)
        model.load_state_dict(
            dist_util.load_state_dict(checkpoint_path, map_location="cpu")
        )
        model.to(dist_util.dev())
        model.eval()

        return SamplingModel(model=model, diffusion=diffusion, tokenizer=tokenizer, is_super_res=is_super_res)

    def sample(self,
               text: Union[str, List[str]],
               batch_size: int,
               n_samples: int,
               clip_denoised=True,
               use_ddim=False,
               low_res=None,
               seed=None,
               to_visible=True,
               from_visible=True
               ):
        dist_util.setup_dist()

        if self.is_super_res and low_res is None:
            raise ValueError('must pass low_res for super res')

        if isinstance(text, str):
            batch_text = batch_size * [text]
        else:
            if len(text) != batch_size:
                raise ValueError(f"got len({text}) texts for bs {batch_size}")
            batch_text = text

        n_batches = n_samples // batch_size

        if seed is not None:
            print(f"setting seed to {seed}")
            th.manual_seed(seed)

        sample_fn = (
            self.diffusion.p_sample_loop if not use_ddim else self.diffusion.ddim_sample_loop
        )

        model_kwargs = {}

        txt = tokenize(self.tokenizer, batch_text)
        txt = th.as_tensor(txt).to(dist_util.dev())
        model_kwargs["txt"] = txt

        if self.is_super_res:
            # TODO: shape vs. text shape
            print(f"batch_size: {batch_size} vs low_res shape {low_res.shape}")

            low_res = th.from_numpy(low_res).float()

            if from_visible:
                low_res = low_res / 127.5 - 1.0
                low_res = low_res.permute(0, 3, 1, 2)

            # model_kwargs['low_res'] = th.cat([low_res for _ in range(batch_size)])
            # model_kwargs['low_res'] = th.stack([low_res for _ in range(batch_size)])
            model_kwargs['low_res'] = low_res.to(dist_util.dev())
            print(f"batch_size: {batch_size} vs low_res kwarg shape {model_kwargs['low_res'].shape}")

        image_channels = self.model.in_channels
        if self.is_super_res:
            image_channels -= model_kwargs['low_res'].shape[1]

        all_images = []

        while len(all_images) * batch_size < n_samples:
            if False: # self.is_super_res:
                pass
            else:
                sample = sample_fn(
                    self.model,
                    (batch_size, image_channels, self.model.image_size, self.model.image_size),
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
                if to_visible:
                    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                    sample = sample.permute(0, 2, 3, 1)
                    sample = sample.contiguous()

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        all_images = np.concatenate(all_images, axis=0)

        return all_images


class SamplingPipeline(nn.Module):
    def __init__(self, base_model: SamplingModel, super_res_model: SamplingModel):
        super().__init__()
        self.base_model = base_model
        self.super_res_model = super_res_model

    def sample(self,
               text: Union[str, List[str]],
               batch_size: int,
               n_samples: int,
               clip_denoised=True,
               use_ddim=False,
               low_res=None,
               seed=None
               ):
        low_res = self.base_model.sample(text, batch_size, n_samples,
                                         clip_denoised=clip_denoised, use_ddim=use_ddim,
                                         seed=seed, to_visible=False)
        high_res = self.super_res_model.sample(text, batch_size, n_samples,
                                               low_res=low_res,
                                               clip_denoised=clip_denoised, use_ddim=use_ddim,
                                               seed=seed, from_visible=False)
        return high_res

    def sample_with_pruning(
        self,
        text: Union[str, List[str]],
        batch_size: int,
        n_samples: int,
        prune_fn,
        continue_if_all_pruned=True,
        clip_denoised=True,
        use_ddim=False,
        low_res=None,
        seed=None
        ):
        low_res = self.base_model.sample(text, batch_size, n_samples,
                                         clip_denoised=clip_denoised, use_ddim=use_dim,
                                         seed=seed, to_visible=True)
        low_res_pruned, text_pruned = prune_fn(low_res, text)
        if len(low_res_pruned) == 0:
            if continue_if_all_pruned:
                print(f"all {len(low_res)} low res samples would be pruned, skipping prune")
                low_res_pruned = low_res
            else:
                return low_res_pruned

        high_res = self.super_res_model.sample(text_pruned, batch_size, n_samples,
                                               low_res=low_res_pruned,
                                               clip_denoised=clip_denoised, use_ddim=use_dim,
                                               seed=seed, from_visible=True)
        return high_res

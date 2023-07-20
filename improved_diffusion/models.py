from . import pipeline
import matplotlib.pyplot as plt
from glob import glob
import torch
import warnings
import time
import random
from textwrap import fill
import numpy as np
from PIL import Image
from datetime import datetime
import os
from tqdm.auto import tqdm
import wget
import sys

def bar_progress(current, total, width=80):
  progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
  # Don't use print() as it will print in new line every time.
  sys.stdout.write("\r" + progress_message)
  sys.stdout.flush()

bucket_path = 'https://storage.googleapis.com/arbml-bucket'

class DiffusionModel:

  def __init__(self, mode="pretrained", nsteps_64 = 250, frameskip_64 = 25):
    os.makedirs('bucket', exist_ok = True)
    ckpt_path = f"bucket/{mode}_model.pt"
    config_path = f"bucket/{mode}_config.json"
    if not os.path.exists(ckpt_path):
      if mode == 'pretrained':
        wget.download(f'{bucket_path}/model_1m_mulfont_bs_128_64x64_brown_with_clipv2/model050000.pt', out = f'bucket/{mode}_model.pt', bar = bar_progress)
        wget.download(f'{bucket_path}/model_1m_mulfont_bs_128_64x64_brown_with_clipv2/config.json', out = f'bucket/{mode}_config.json', bar = bar_progress)
      elif mode == 'finetuned':
        wget.download(f'{bucket_path}/fintuned_model_2k_calliar_bs_128_64x64v2/ema_0.9999_057000.pt', out = f'bucket/{mode}_model.pt', bar = bar_progress)
        wget.download(f'{bucket_path}/fintuned_model_2k_calliar_bs_128_64x64v2/config.json', out = f'bucket/{mode}_config.json', bar = bar_progress)
      else:
        raise('error not available mode')
    
    clipname = "ViT-L/14@336px"
    self.model_64 = pipeline.SamplingModel.from_config(
        checkpoint_path=ckpt_path,
        config_path=config_path,
        timestep_respacing="250",
        clipname=clipname
    )
    self.nsteps_64 = nsteps_64
    self.frameskip_64 = frameskip_64
    self.model_64.cuda()
    self.frameskip_64 = min(self.frameskip_64, self.nsteps_64 // 2)

    self.model_64.set_timestep_respacing(str(self.nsteps_64))
    
  def _lower(self, t):
    if isinstance(t, torch.Tensor):
        return t.cpu().numpy()
    return np.asarray(t)

  def generate(self, prompt, capt="",
              uneven_steps_for_upsamplers = True,
              guidance_scale_64 = 2,
              dynamic_threshold_p_64 = 0.99,
              seed = -1):
    t = time.time()
    t_last_render = t
    if capt == "":
      capt = "unknown"
    
    denoised_images = []
    with torch.cuda.amp.autocast():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

        count = 0
        gen_64 = self.model_64.sample(
            prompt,
            1,
            1,
            yield_intermediates=True,
            clip_denoised=True,
            clf_free_guidance=True,
            guidance_scale=guidance_scale_64,
            dynamic_threshold_p=dynamic_threshold_p_64,
            capt=capt,
            seed=seed,
            verbose=False,
        )
        pbar = tqdm(total=10)
        for i, (s, xs) in enumerate(gen_64):
            t2 = time.time()
            delta = t2 - t
            t = t2

            ts = self.nsteps_64
            fs = self.frameskip_64
            prefix = ""
            count += 1
            if count % fs == (fs - 1):
                t_start_render = time.time()
                s, xs = self._lower(s), self._lower(xs)
                t_end_render = time.time()
                t_last_render = t_end_render
                denoised_images.append((s[0], xs[0]))
                pbar.update(1)
    return denoised_images

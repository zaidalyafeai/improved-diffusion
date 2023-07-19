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

class Model:

  def __init__(self, ckpt_path, nsteps_64 = 250, frameskip_64 = 25):

    config_path = '/'.join(ckpt_path.split('/')[:-1])+"/config.json"
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

  def generate(self, prompt, capt,
              uneven_steps_for_upsamplers = True,
              guidance_scale_64 = 2,
              dynamic_threshold_p_64 = 0.99,
              seed = -1):
    t = time.time()
    t_last_render = t
    
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
    return denoised_images
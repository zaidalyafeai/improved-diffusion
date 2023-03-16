from  improved_diffusion import pipeline
import matplotlib.pyplot as plt
model_part = pipeline.SamplingModel.from_config(
        checkpoint_path="/content/improved-diffusion/logs/model004000.pt",
        config_path="logsconfig.json",
        timestep_respacing = "250"
    )

import time
import ipywidgets
import json, random
from textwrap import fill
import numpy as np
from PIL import Image
from datetime import datetime

model_64 = model_part

# @markdown #### Sampling steps


steps_64 =    250# @param {type:"number"}
steps_128 =    250# @param {type:"number"}
steps_256 =    250# @param {type:"number"}
steps_512 =    250# @param {type:"number"}

uneven_steps_for_upsamplers = True # @param {type:"boolean"}


# @markdown #### Guidance

guidance_scale_64 =   2# @param {type:"number"}
guidance_scale_128 =   2# @param {type:"number"}
guidance_scale_256 =   0# @param {type:"number"}

# @markdown #### Noise conditioning

aug_level_128 = 0.150 # @param {type:"number"}
aug_level_256 = 0.100 # @param {type:"number"}
aug_level_512 = 0.075 # @param {type:"number"}

noise_cond_ts_128 = int(1000 * aug_level_128)
noise_cond_ts_256 = int(1000 * aug_level_256)
noise_cond_ts_512 = int(1000 * aug_level_512)

# @markdown #### Thresholding

dynamic_threshold_p_64 =  0.995# @param {type:"number"}
dynamic_threshold_p_128 =  0.995# @param {type:"number"}
dynamic_threshold_p_256 =  0# @param {type:"number"}
dynamic_threshold_p_512 =  0# @param {type:"number"}

# @markdown #### Display options

frameskip_64 =     25# @param {type:"number"}
frameskip_128 =     25# @param {type:"number"}
frameskip_256 =     20# @param {type:"number"}
frameskip_512 =     20# @param {type:"number"}

# avoids weirdness w/ colab's param type integer
frameskip_64 = int(frameskip_64)
frameskip_128 = int(frameskip_128)
frameskip_256 = int(frameskip_256)
frameskip_512 = int(frameskip_512)

# @markdown #### **What should we draw?**

seed =   -1# @param {type:"number"}
seed = int(seed)  # avoids weirdness w/ colab's param type integer

transcription = "That was amazing"  # @param {type:"string"}

description =   "a smiling metal robot with a speech bubble next to it" # @param {type:"string"}

mask_transcription = False
mask_description = False


if seed < 0:
    seed = random.randint(0, 999999)
    print(f"Using seed: {seed} (Use this value to reproduce the same image if you like it.)")
    
seed_64 = seed_128 = seed_256 = seed_512 = seed

# internal terminology
prompt = transcription
capt = description

if mask_transcription:
    prompt = "<mask><mask><mask><mask>"
if mask_description:
    capt = "unknown"

prompt = prompt.replace('\\n', '\n')

if isinstance(steps_64, int):
    steps_64 = str(steps_64)

if isinstance(steps_64, tuple) or isinstance(steps_64, list):
    steps_64 = ','.join(str(e) for e in steps_64)

nsteps_64 = sum(int(e) for e in str(steps_64).split(','))

if isinstance(steps_128, int):
    steps_128 = str(steps_128)

if isinstance(steps_128, tuple) or isinstance(steps_128, list):
    steps_128 = ','.join(str(e) for e in steps_128)

nsteps_128 = sum(int(e) for e in str(steps_128).split(','))

if isinstance(steps_256, int):
    steps_256 = str(steps_256)

if isinstance(steps_256, tuple) or isinstance(steps_256, list):
    steps_256 = ','.join(str(e) for e in steps_256)

nsteps_256 = sum(int(e) for e in str(steps_256).split(','))

if isinstance(steps_512, int):
    steps_512 = str(steps_512)

if isinstance(steps_512, tuple) or isinstance(steps_512, list):
    steps_512 = ','.join(str(e) for e in steps_512)

nsteps_512 = sum(int(e) for e in str(steps_512).split(','))

frameskip_64 = min(frameskip_64, nsteps_64//2)
frameskip_128 = min(frameskip_128, nsteps_128//2)
frameskip_256 = min(frameskip_256, nsteps_256//2)
frameskip_512 = min(frameskip_512, nsteps_512//2)

model_64.set_timestep_respacing(steps_64)

import warnings



t = time.time()
t_last_render = t

import torch

def _lower(t):
    if isinstance(t, torch.Tensor):
        return t.cpu().numpy()
    return np.asarray(t)

with torch.cuda.amp.autocast():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        count = 0
        partname = 'part 1'

        guidance_scale_64_ = guidance_scale_64

        dynamic_threshold_p_64_ = dynamic_threshold_p_64

        if True:
            model_64.cuda();
        
        gen_64 = model_64.sample(
            prompt, 1, 1, 
            yield_intermediates=True,
            clip_denoised=True,
            clf_free_guidance=True,
            guidance_scale=guidance_scale_64_,
            dynamic_threshold_p=dynamic_threshold_p_64_,
            capt=capt,
            seed=seed_64,
            verbose=False,
            )
        
        for i, (s, xs) in enumerate(gen_64):
            t2 = time.time()
            delta = t2 - t
            t = t2

            ts = nsteps_64
            fs = frameskip_64
            prefix = ""

            count += 1

            plms = False
            count_ = count*2 if plms else count
            if plms:
                fs /= 2
            
            if (count % fs == (fs-1) or count == ts):

                t_start_render = time.time()

                s, xs = _lower(s), _lower(xs)

                frame = np.concatenate([s[0], xs[0]], axis=1)
                img = Image.fromarray(frame)
                if img.size[1] < 128:
                    img = img.resize((256, 128), resample=Image.NEAREST)
                img_last = s[0]
                img.save(f'logs/{i}.png')

                t_end_render = time.time()

                frac = (t_end_render - t_start_render) / (t_start_render - t_last_render)
                t_last_render = t_end_render
                
                time.sleep(0.1)

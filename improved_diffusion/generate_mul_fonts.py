from improved_diffusion import pipeline
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

# font_names = ['ديواني جلي', 'ديواني مشكل', 'ديواني طويل', 'ديواني بسيط', 'كوفي بسيط', 'كوفي منحني', 'فارسي بسيط', 'مغربي اندلس', 'رقعة مدبب', 'رقعة بسيط', 'رقعة سريع', 'ثلث ديواني', 'ثلث بسيط', 'مربع بسيط', 'حر مدبب', 'حر بسيط', 'حر طويل', 'موبايلي', 'منجا', 'الجزيرة']

# font_names += ["فارسي مدبب", "جلي رقعة", "كوفي سريع", "رقعة مشكل", "كوفي طويل"]

font_names = ['diwani decorated', 'diwani diacritized', 'diwani long', 'diwani standard',
                'kufi standard', 'kufi curved suqare', 'farisi standard', 'morrocan andulus',
                'rukaa bold', 'rukaa standard', 'rukaa fast', 'thuluth diwani', 'thuluth standard'
                ,'square standard', 'free bold', 'free standard', 'free long', 'mobili', 'managa'
                , 'aljazeera']
font_names += ['kufi long', 'kufi diacritized', 'kufi fast', 'kufi decorated', 'mobili managa']

def generate(
    model_64, 
    transcription, 
    description,
    image_logs_path,
    steps_64 = 250,  
    uneven_steps_for_upsamplers = True,  
    guidance_scale_64 = 2,  
    dynamic_threshold_p_64 = 0.99,  
    frameskip_64 = 25,
    mask_transcription = False,
    mask_description = False,
    seed = -1
    
): 
    if seed < 0:
        seed = random.randint(0, 999999)
        print(
            f"Using seed: {seed} (Use this value to reproduce the same image if you like it.)"
        )

    os.makedirs(image_logs_path, exist_ok = True)
    seed_64 = seed_128 = seed_256 = seed_512 = seed

    # internal terminology
    prompt = transcription
    capt = description

    if mask_transcription:
        prompt = "<mask><mask><mask><mask>"
    if mask_description:
        capt = "unknown"

    prompt = prompt.replace("\\n", "\n")

    if isinstance(steps_64, int):
        steps_64 = str(steps_64)

    if isinstance(steps_64, tuple) or isinstance(steps_64, list):
        steps_64 = ",".join(str(e) for e in steps_64)

    nsteps_64 = sum(int(e) for e in str(steps_64).split(","))

    frameskip_64 = min(frameskip_64, nsteps_64 // 2)

    model_64.set_timestep_respacing(steps_64)

    t = time.time()
    t_last_render = t

    def _lower(t):
        if isinstance(t, torch.Tensor):
            return t.cpu().numpy()
        return np.asarray(t)

    with torch.cuda.amp.autocast():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            count = 0
            partname = "part 1"

            guidance_scale_64_ = guidance_scale_64

            dynamic_threshold_p_64_ = dynamic_threshold_p_64

            if True:
                model_64.cuda()

            gen_64 = model_64.sample(
                prompt,
                1,
                1,
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
                count_ = count * 2 if plms else count
                if plms:
                    fs /= 2

                if count % fs == (fs - 1) or count == ts:

                    t_start_render = time.time()

                    s, xs = _lower(s), _lower(xs)

                    frame = np.concatenate([s[0], xs[0]], axis=1)
                    img = Image.fromarray(frame)
                    if img.size[1] < 128:
                        img = img.resize((256, 128), resample=Image.NEAREST)
                    img_last = s[0]
                    print(f'saving image {image_logs_path}/{i}.png ...')
                    img.save(f"{image_logs_path}/{i}.png")

                    t_end_render = time.time()

                    frac = (t_end_render - t_start_render) / (
                        t_start_render - t_last_render
                    )
                    t_last_render = t_end_render

                    time.sleep(0.1)
def run(args,):
    ckpt_path = args.path
    image_logs_path = f"logs/generated_" + '_'.join(ckpt_path.split("/")[1:]).split(".")[0]
    i = 0
    while True:
        if os.path.isdir(image_logs_path+f"_{i}"):
            i += 1
        else:
            image_logs_path += f"_{i}"
            print(f'saving to {image_logs_path}')
            break
    config_path = '/'.join(ckpt_path.split('/')[:-1])+"/config.json"
    print("saving to ", image_logs_path)
    clipname = "t5-v1_1-xxl"
    if args.text_encoder_type == "clip":
        clipname = "ViT-L/14@336px"
    model_part = pipeline.SamplingModel.from_config(
        checkpoint_path=ckpt_path,
        config_path=config_path,
        timestep_respacing="250",
        clipname=clipname
    )

    model_64 = model_part
    transcription = args.text  # @param {type:"string"}
    if args.capt != "":
        descriptions = [args.capt] * args.n  # @param {type:"string"}
        for i, description in enumerate(descriptions):
            generate(model_64, transcription, description, f"{image_logs_path}/{description}_{i}")
    else:
        descriptions = font_names
        for i, description in enumerate(descriptions):
            generate(model_64, transcription, description, f"{image_logs_path}/{description}")

    


    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="generate new images")
    parser.add_argument("-path", "--path", type=str, required=True)
    parser.add_argument("-text", "--text", type=str, required=True)
    parser.add_argument("-text_encoder_type", "--text_encoder_type", type=str, required=True)
    parser.add_argument("-capt", "--capt", default = "", type=str, required=False)
    parser.add_argument("-n", "--n", type=int, required=True, default = 25)
    args = parser.parse_args()
    run(args)

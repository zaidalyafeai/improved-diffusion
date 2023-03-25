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


def run(args):
    ckpt_path = args.path
    image_logs_path = "logs/generated_" + (ckpt_path.split("/")[-1].split(".")[0])
    print("saving to ", image_logs_path)
    os.makedirs(image_logs_path, exist_ok=True)
    model_part = pipeline.SamplingModel.from_config(
        checkpoint_path=ckpt_path,
        config_path="logsconfig.json",
        timestep_respacing="250",
    )

    model_64 = model_part

    # @markdown #### Sampling steps

    steps_64 = 250  # @param {type:"number"}

    uneven_steps_for_upsamplers = True  # @param {type:"boolean"}
    guidance_scale_64 = 2  # @param {type:"number"}
    dynamic_threshold_p_64 = 0.995  # @param {type:"number"}
    frameskip_64 = 25  # @param {type:"number"}

    frameskip_64 = int(frameskip_64)

    # @markdown #### **What should we draw?**

    seed = -1  # @param {type:"number"}
    seed = int(seed)  # avoids weirdness w/ colab's param type integer

    transcription = args.txt  # @param {type:"string"}

    description = "unkown"  # @param {type:"string"}

    mask_transcription = False
    mask_description = False

    if seed < 0:
        seed = random.randint(0, 999999)
        print(
            f"Using seed: {seed} (Use this value to reproduce the same image if you like it.)"
        )

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
                    img.save(f"{image_logs_path}/{i}.png")

                    t_end_render = time.time()

                    frac = (t_end_render - t_start_render) / (
                        t_start_render - t_last_render
                    )
                    t_last_render = t_end_render

                    time.sleep(0.1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="generate new images")
    parser.add_argument("-path", "--path", type=str, required=True)
    parser.add_argument("-text", "--text", type=str, required=True)

    args = parser.parse_args()
    run(args)

"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import blobfile as bf

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from improved_diffusion.image_datasets import load_tokenizer, tokenize


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    n_texts = args.num_samples // args.batch_size

    using_text_dir = False
    batch_texts = n_texts * [args.text_input]
    # noise = None

    if args.text_dir and os.path.exists(args.text_dir):
        using_text_dir = True
        data_dir = args.text_dir

        text_files = [
            bf.join(data_dir, entry)
            for entry in sorted(bf.listdir(data_dir))
            if entry.endswith('.txt')
        ][args.text_dir_offset:args.text_dir_offset + n_texts]

        # debug
        # text_files = n_texts * [text_files[0]]

        batch_texts = []
        for i, path_txt in enumerate(text_files):
            with bf.BlobFile(path_txt, "r") as f:
                text = f.read()
            batch_texts.append(text)
            print(f"text {i}: {repr(text)}")

        # # constant noise
        # shape = (1, 3, args.image_size, args.image_size)
        # device = next(model.parameters()).device
        # noise = th.randn(*shape, device=device)
        # noise = th.tile(noise, (args.batch_size, 1, 1, 1))
    else:
        print(f"text_input: {args.text_input}")

    text_gen = (x for x in batch_texts)

    logger.log("sampling...")
    if args.seed > -1:
        print(f"setting seed to {args.seed}")
        th.manual_seed(args.seed)
    all_images = []
    all_labels = []

    while len(all_images) * args.batch_size < args.num_samples:
        if (args.seed > -1) and using_text_dir:
            print(f"setting seed to {args.seed}")
            th.manual_seed(args.seed)

        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        if args.txt:
            this_text = args.batch_size * [next(text_gen)]
            tokenizer = load_tokenizer(max_seq_len=model.text_encoder.pos_emb.emb.num_embeddings)
            txt = th.as_tensor(tokenize(tokenizer, this_text)).to(dist_util.dev())
            model_kwargs["txt"] = txt
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        image_channels = 1 if args.monochrome else 3
        sample = sample_fn(
            model,
            (args.batch_size, image_channels, args.image_size, args.image_size),
            # noise=noise,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        text_input="",
        text_dir="",
        text_dir_offset=0,
        log_interval=10,  # ignored
        seed=-1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

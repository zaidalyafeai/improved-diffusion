"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.image_datasets import load_superres_data, load_tokenizer, tokenize


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("loading data...")
    using_ground_truth = args.base_data_dir != "" and os.path.exists(args.base_data_dir)
    print(f"args.base_data_dir: {args.base_data_dir} | using_ground_truth: {using_ground_truth}")
    n_texts = args.num_samples // args.batch_size
    if n_texts > 1:
        raise ValueError("num_samples != bs TODO")

    if using_ground_truth:
        data = load_superres_data(
            args.base_data_dir,
            batch_size=n_texts,
            large_size=args.large_size,
            small_size=args.small_size,
            class_cond=args.class_cond,
            txt=args.txt,
            monochrome=args.monochrome,
            deterministic=True,
            offset=args.base_data_offset,
        )
        data = (model_kwargs for _, model_kwargs in data)
        tokenizer = load_tokenizer(max_seq_len=model.text_encoder.pos_emb.emb.num_embeddings)
    else:
        data = load_data_for_worker(args.base_samples, args.batch_size, args.class_cond, args.txt)
        tokenizer = None

    logger.log("creating samples...")
    if args.seed > -1:
        print(f"setting seed to {args.seed}")
        th.manual_seed(args.seed)

    all_images = []
    image_channels = 1 if args.monochrome else 3
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = next(data)
        if using_ground_truth:
            print(f"text: {repr(model_kwargs['txt'])}")
            if args.txt_override != "":
                model_kwargs['txt'] = [args.txt_override for _ in model_kwargs['txt']]
                print(f"overridden with: {repr(model_kwargs['txt'])}")
            txt = tokenize(tokenizer, model_kwargs["txt"])
            txt = th.as_tensor(txt).to(dist_util.dev())
            model_kwargs["txt"] = txt

            for k, v in model_kwargs.items():
                print((k, v.shape))
            model_kwargs['low_res'] = th.cat([model_kwargs['low_res'] for _ in range(args.batch_size)])
            model_kwargs['txt'] = th.cat([model_kwargs['txt'] for _ in range(args.batch_size)])
            for k, v in model_kwargs.items():
                print((k, v.shape))
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, image_channels, args.large_size, args.large_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        all_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(all_samples, sample)  # gather not supported with NCCL
        for sample in all_samples:
            all_images.append(sample.cpu().numpy())
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def load_data_for_worker(base_samples, batch_size, class_cond, txt):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]
        if class_cond or txt:
            label_arr = obj["arr_1"]
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer = []
    label_buffer = []
    while True:
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i])
            if class_cond or txt:
                label_buffer.append(label_arr[i])
            if len(buffer) == batch_size:
                batch = th.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                res = dict(low_res=batch)
                if class_cond or txt:
                    key = "txt" if txt else "y"
                    res[key] = th.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer = [], []


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        base_samples="",
        base_data_dir="",
        base_data_offset=0,
        model_path="",
        log_interval=None,  # ignored
        seed=-1,
        txt_override="",
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

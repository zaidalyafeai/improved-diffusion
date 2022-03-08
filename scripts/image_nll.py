"""
Approximate the bits/dimension for an image model.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data, tokenize, load_tokenizer
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_config_to_model,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    config_path = args.config_path
    have_config_path = config_path != ""
    using_config = have_config_path and os.path.exists(config_path)

    logger.log("creating model and diffusion...")

    if using_config:
        model, diffusion_factory, tokenizer, _ = load_config_to_model(config_path, args)
        diffusion = diffusion_factory()

    else:
        tokenizer = None
        tokenizer_config = dict(
            max_seq_len=getattr(args, 'max_seq_len', None),
            char_level=getattr(args, 'char_level', None),
        )

        if args.txt:
            tokenizer = load_tokenizer(**tokenizer_config)


        model_diffusion_args = args_to_dict(args, model_and_diffusion_defaults().keys())
        model_diffusion_args['tokenizer'] = tokenizer
        model, diffusion = create_model_and_diffusion(
            **model_diffusion_args
        )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        txt=args.txt,
        deterministic=True,
    )

    logger.log("evaluating...")
    run_bpd_evaluation(model, diffusion, data, args.num_samples, args.clip_denoised, tokenizer=tokenizer)


def run_bpd_evaluation(model, diffusion, data, num_samples, clip_denoised, tokenizer=None):
    all_bpd = []
    all_metrics = {"vb": [], "mse": [], "xstart_mse": []}
    num_complete = 0
    while num_complete < num_samples:
        batch, model_kwargs = next(data)
        batch = batch.to(dist_util.dev())
        if 'txt' in model_kwargs:
            model_kwargs['txt'] = th.as_tensor(tokenize(tokenizer, model_kwargs['txt']), device=dist_util.dev())
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        minibatch_metrics = diffusion.calc_bpd_loop(
            model, batch, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )

        for key, term_list in all_metrics.items():
            terms = minibatch_metrics[key].mean(dim=0) / dist.get_world_size()
            dist.all_reduce(terms)
            term_list.append(terms.detach().cpu().numpy())

        total_bpd = minibatch_metrics["total_bpd"]
        total_bpd = total_bpd.mean() / dist.get_world_size()
        dist.all_reduce(total_bpd)
        all_bpd.append(total_bpd.item())
        num_complete += dist.get_world_size() * batch.shape[0]

        logger.log(f"done {num_complete} samples: bpd={np.mean(all_bpd)}")

    if dist.get_rank() == 0:
        for name, terms in all_metrics.items():
            out_path = os.path.join(logger.get_dir(), f"{name}_terms.npz")
            logger.log(f"saving {name} terms to {out_path}")
            np.savez(out_path, np.mean(np.stack(terms), axis=0))

    dist.barrier()
    logger.log("evaluation complete")


def create_argparser():
    defaults = dict(
        data_dir="", clip_denoised=True, num_samples=1000, batch_size=1, model_path="",
        config_path="",
        char_level=True,
        max_seq_len=384,

    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

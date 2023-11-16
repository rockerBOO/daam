from collections import defaultdict
from pathlib import Path
import argparse
import json
import random
import sys
import time

import pandas as pd
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverSinglestepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    AutoencoderKL
)
from tqdm import tqdm
import numpy as np
import torch
import accelerate

from daam import trace
from daam.experiment import GenerationExperiment, build_word_list_coco80
from daam.utils import set_seed, cached_nlp, auto_device, auto_autocast

actions = [
    "quickgen",
    "prompt",
    "coco",
    "template",
    "cconj",
    "coco-unreal",
    "stdin",
    "regenerate",
]
model_id_map = {
    "v1": "runwayml/stable-diffusion-v1-5",
    "v2-base": "stabilityai/stable-diffusion-2-base",
    "v2-large": "stabilityai/stable-diffusion-2",
    "v2-1-base": "stabilityai/stable-diffusion-2-1-base",
    "v2-1-large": "stabilityai/stable-diffusion-2-1",
}

samplers = [
    "ddim",
    "pndm",
    "lms",
    "euler",
    "euler_a",
    "heun",
    "dpm_2",
    "dpm_2_a",
    "dpmsolver",
    "dpmsolver++",
    "dpmsingle",
    "k_lms",
    "k_euler",
    "k_euler_a",
    "k_dpm_2",
    "k_dpm_2_a",
]

# scheduler:
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"


def main(args):
    # args.lemma = cached_nlp(args.word)[0].lemma_ if args.word else None
    model_id = model_id_map[args.model]
    seeds = []

    if args.output_folder is None:
        args.output_folder = "."

    prompts = [(".", args.prompt)]

    if args.output_folder is None:
        args.output_folder = "output"

    sched_init_args = {}
    if args.sampler == "ddim":
        scheduler_cls = DDIMScheduler
    elif args.sampler == "ddpm":  # ddpmはおかしくなるのでoptionから外してある
        scheduler_cls = DDPMScheduler
    elif args.sampler == "pndm":
        scheduler_cls = PNDMScheduler
    elif args.sampler == "lms" or args.sampler == "k_lms":
        scheduler_cls = LMSDiscreteScheduler
    elif args.sampler == "euler" or args.sampler == "k_euler":
        scheduler_cls = EulerDiscreteScheduler
    elif args.sampler == "euler_a" or args.sampler == "k_euler_a":
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif args.sampler == "dpmsolver" or args.sampler == "dpmsolver++":
        scheduler_cls = DPMSolverMultistepScheduler
        sched_init_args["algorithm_type"] = args.sampler
    elif args.sampler == "dpmsingle":
        scheduler_cls = DPMSolverSinglestepScheduler
    elif args.sampler == "heun":
        scheduler_cls = HeunDiscreteScheduler
    elif args.sampler == "dpm_2" or args.sampler == "k_dpm_2":
        scheduler_cls = KDPM2DiscreteScheduler
    elif args.sampler == "dpm_2_a" or args.sampler == "k_dpm_2_a":
        scheduler_cls = KDPM2AncestralDiscreteScheduler
    else:
        scheduler_cls = DDIMScheduler

    scheduler = scheduler_cls(
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDLER_SCHEDULE,
        **sched_init_args,
    )
    
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        safety_checker=None,
        scheduler=scheduler,
        vae=vae,
    )
    pipe = auto_device(pipe)

    accelerator = accelerate.Accelerator()

    with accelerator.autocast(), torch.no_grad():
        for gen_idx, (prompt_id, prompt) in enumerate(tqdm(prompts)):
            seed = int(time.time()) if args.random_seed else args.seed

            if seeds and gen_idx < len(seeds):
                seed = seeds[gen_idx]

            gen = set_seed(seed)

            prompt_id = str(prompt_id)

            with trace(
                pipe,
                # 512,
                # 512,
                args.width,
                args.height,
                low_memory=args.low_memory,
                save_heads=args.save_heads,
                load_heads=args.load_heads,
            ) as tc:

                out = pipe(
                    prompt,
                    width=args.width,
                    height=args.height,
                    num_inference_steps=args.steps,
                    guidance_scale=7.0,
                    generator=gen,
                )

                img_filename = f"{prompt_id}-{prompt}.png"
                print(f"Saving image to {img_filename}")
                out.images[0].save(img_filename)

                global_heat_map = tc.compute_global_heat_map(prompt=prompt)
                for word in args.words.split(","):
                    word = word.strip()
                    heat_map = global_heat_map.compute_word_heat_map(word)
                    daam_img_filename = f"{word}-daam.png"
                    print(f"Saving daam heatmap to {daam_img_filename}")
                    heat_map.plot_overlay(out.images[0], out_file=daam_img_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
    Test DAAM

    accelerate launch test.py "a woman with blonde hair standing on the street" --words "woman,blonde hair,street" --width 384 --height 640 --sampler=dpmsolver++ --steps=15
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("prompt", type=str)
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument(
        "--model", type=str, default="v1", choices=list(model_id_map.keys())
    )
    parser.add_argument("--output-folder", "-o", type=str)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--steps", "-n", type=int, default=50)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--all-heads", action="store_true")
    parser.add_argument("--words", type=str)
    parser.add_argument("--random-seed", action="store_true")
    parser.add_argument("--save-heads", action="store_true")
    parser.add_argument("--load-heads", action="store_true")

    parser.add_argument(
        "--sampler",
        type=str,
        default="ddim",
        choices=samplers,
        help=f"sampler (scheduler) type for images / サンプル出力時のサンプラー（スケジューラ）の種類",
    )
    args = parser.parse_args()

    main(args)

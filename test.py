from collections import defaultdict
from pathlib import Path
import argparse
import json
import random
import sys
import time

import pandas as pd
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
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


def main(args):
    # args.lemma = cached_nlp(args.word)[0].lemma_ if args.word else None
    model_id = model_id_map[args.model]
    seeds = []

    if args.output_folder is None:
        args.output_folder = "."

    prompts = [(".", args.prompt)]

    if args.output_folder is None:
        args.output_folder = "output"

    # new_prompts = []
    #
    # if args.lemma is not None:
    #     for prompt_id, prompt in tqdm(prompts):
    #         if args.lemma not in prompt.lower():
    #             continue
    #
    #         doc = cached_nlp(prompt)
    #         found = False
    #
    #         for tok in doc:
    #             if tok.lemma_.lower() == args.lemma and not found:
    #                 found = True
    #             elif (
    #                 tok.lemma_.lower() == args.lemma
    #             ):  # filter out prompts with multiple instances of the word
    #                 found = False
    #                 break
    #
    #         if found:
    #             new_prompts.append((prompt_id, prompt))
    #
    #     prompts = new_prompts

    # scheduler = DPMSolverMultistepScheduler()
    # prompts = prompts[: args.gen_limit]
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        safety_checker=None,
        # scheduler=scheduler,
    )
    pipe = auto_device(pipe)

    accelerator = accelerate.Accelerator()

    with accelerator.autocast(), torch.no_grad():
        for gen_idx, (prompt_id, prompt) in enumerate(tqdm(prompts)):
            # seed = int(time.time()) if args.random_seed else args.seed
            # prompt = prompt.replace(",", " ,").replace(".", " .").strip()

            # if seeds and gen_idx < len(seeds):
            #     seed = seeds[gen_idx]
            #
            # gen = set_seed(seed)
            #
            # if args.action == "cconj":
            #     seed = int(prompt_id.split("-")[1]) + args.seed_offset
            #     gen = set_seed(seed)

            # prompt_id = str(prompt_id)
            print(prompt)

            with trace(
                pipe,
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
                    num_inference_steps=args.num_timesteps,
                    # generator=gen,
                    callback=tc.time_callback,
                )

                out_image_filename = f"{prompt}.png"
                out.images[0].save(out_image_filename)
                print(f"Save generated image to {out_image_filename}")

                global_heat_map = tc.compute_global_heat_map(prompt=prompt)
                for word in args.words.split(","):
                    heat_map = global_heat_map.compute_word_heat_map(word)
                    heat_map.plot_overlay(out.images[0], out_file=f"{word}-daam.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
    Test DAAM

    python test.py 
    """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("prompt", type=str)
    parser.add_argument("--action", "-a", type=str, choices=actions, default=actions[0])
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument(
        "--model", type=str, default="v1", choices=list(model_id_map.keys())
    )
    parser.add_argument("--output-folder", "-o", type=str)
    parser.add_argument("--input-folder", "-i", type=str, default="input")
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--gen-limit", type=int, default=1000)
    parser.add_argument("--template", type=str, default="{numeral} {noun}")
    parser.add_argument(
        "--template-data-file", "-tdf", type=str, default="template.tsv"
    )
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--num-timesteps", "-n", type=int, default=15)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--all-heads", action="store_true")
    parser.add_argument("--words", type=str)
    parser.add_argument("--random-seed", action="store_true")
    parser.add_argument("--truth-only", action="store_true")
    parser.add_argument("--save-heads", action="store_true")
    parser.add_argument("--load-heads", action="store_true")
    args = parser.parse_args()

    main(args)

import os
import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import TypeVar, Tuple, List, Union

import matplotlib.pyplot as plt
import numpy as np

# import PIL.Image
from PIL import Image
import spacy
import torch
import torch.nn.functional as F

from transformers.image_transforms import to_pil_image


__all__ = [
    "set_seed",
    "compute_token_merge_indices",
    "plot_mask_heat_map",
    "cached_nlp",
    "cache_dir",
    "auto_device",
    "auto_autocast",
]


T = TypeVar("T")


def auto_device(obj: T = torch.device("cpu")) -> T:
    if isinstance(obj, torch.device):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        return obj.to("cuda")

    return obj


def auto_autocast(*args, **kwargs):
    if not torch.cuda.is_available():
        kwargs["enabled"] = False

    return torch.cuda.amp.autocast(*args, **kwargs)


def plot_mask_heat_map(im: Image.Image, heat_map: torch.Tensor, threshold: float = 0.4):
    im = torch.from_numpy(np.array(im)).float() / 255
    mask = (heat_map.squeeze() > threshold).float()
    im = im * mask.unsqueeze(-1)
    plt.imshow(im)


def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    gen = torch.Generator(device=auto_device())
    gen.manual_seed(seed)

    return gen


def cache_dir() -> Path:
    # *nix
    if os.name == "posix" and sys.platform != "darwin":
        xdg = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        return Path(xdg, "daam")
    elif sys.platform == "darwin":
        # Mac OS
        return Path(os.path.expanduser("~"), "Library/Caches/daam")
    else:
        # Windows
        local = os.environ.get("LOCALAPPDATA", None) or os.path.expanduser(
            "~\\AppData\\Local"
        )
        return Path(local, "daam")


def compute_token_merge_indices(
    tokenizer, prompt: str, word: str, word_idx: int = None, offset_idx: int = 0
) -> Tuple[List[int], Union[None, int]]:
    merge_idxs = []

    if prompt == "":
        raise ValueError("Prompt cannot be empty")
    tokens = tokenizer.tokenize(prompt.lower())
    if word_idx is None:
        word = word.lower()
        search_tokens = tokenizer.tokenize(word)
        start_indices = [
            x + offset_idx
            for x in range(len(tokens))
            if tokens[x : x + len(search_tokens)] == search_tokens
        ]
        for indice in start_indices:
            merge_idxs += [i + indice for i in range(0, len(search_tokens))]
        if not merge_idxs:
            raise ValueError(f"Word '{word}' not found in the prompt.")
    else:
        merge_idxs.append(word_idx)

    return [x + 1 for x in merge_idxs], word_idx  # Offset by 1.


nlp = None


@lru_cache(maxsize=100000)
def cached_nlp(prompt: str, type="en_core_web_md"):
    global nlp

    if nlp is None:
        try:
            nlp = spacy.load(type)
        except OSError:
            import os

            os.system(f"python -m spacy download {type}")
            nlp = spacy.load(type)

    return nlp(prompt)


@torch.no_grad()
def expand_image(
    heatmap, image, absolute=False, threshold=None, plot=False, **plot_kwargs
):
    # type: (Image.Image, bool, float, bool, Dict[str, Any]) -> torch.Tensor

    # # remove batch and channel dimensions
    # # TODO maybe handle batch more appropriately
    # h = self.img_height // 8
    # w = self.img_width // 8

    if isinstance(image, Image.Image):
        w = image.size[0]
        h = image.size[1]
    elif isinstance(image, torch.Tensor):
        w = image.size(2)
        h = image.size(1)

    # shape 77, 1, 48, 80
    heatmap = heatmap.unsqueeze(0).unsqueeze(0)

    # The clamping fixes undershoot.
    im = F.interpolate(heatmap, size=(w, h), mode="bicubic").clamp_(min=0)

    # im = heatmap.unsqueeze(0).unsqueeze(0)
    # im = F.interpolate(
    #     im.float().detach(), size=(image.size[0], image.size[1]), mode="bicubic"
    # )

    if not absolute:
        im = (im - im.min()) / (im.max() - im.min() + 1e-8)

    if threshold is not None:
        im = (im > threshold).float()

    im = im.cpu().detach().squeeze()

    return im


def tensor2img(tensor):
    if len(tensor.size()) == 2:
        # add a channel
        tensor = tensor.unsqueeze(-1)

    if tensor.get_device() != -1:
        tensor = tensor.float().detach().cpu()

    if tensor.size(2) > 3:
        print(f"invalid number of channels to tensor2img {tensor.size()}")

    if len(tensor.size()) > 3:
        print(f"invalid size to tensor2img {tensor.size()}")

    return to_pil_image(tensor, do_rescale=True)


# Measure the dimensions of the tensor and get the largest one by dimension size
# tensor.Size([77, 29, 40]) tensor.Size([77, 22, 38]) tensor.Size([77, 29, 18])
def get_max_tensor_width_height(tensors) -> Tuple[int, int]:
    maxes = {(sum(m.size()), m.size(1), m.size(2)) for m in tensors}

    max_sum = -1
    max = (-1, -1)
    for m, w, h in maxes:
        if m > max_sum:
            max = (w, h)
            max_sum = m

    return max

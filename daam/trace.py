import math
from pathlib import Path
from typing import List, Type, Union

import PIL.Image as Image
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from transformers.image_transforms import to_pil_image
from matplotlib import pyplot as plt

from .experiment import GenerationExperiment
from .heatmap import GlobalHeatMap, RawHeatMapCollection, AggregateCollection
from .hook import (
    AggregateHooker,
    ObjectHooker,
    UNetCrossAttentionLocator,
)
from .utils import auto_autocast, cache_dir, tensor2img, get_max_tensor_width_height

__all__ = ["trace", "DiffusionHeatMapHooker", "GlobalHeatMap"]


class DiffusionHeatMapHooker(AggregateHooker):
    def __init__(
        self,
        pipeline: StableDiffusionPipeline = None,
        width: int = 512,
        height: int = 512,
        low_memory: bool = False,
        load_heads: bool = False,
        save_heads: bool = False,
        data_dir: str = None,
        context_size: int = 77,
        unet=None,  # UNet/DiffusionModel
        vae=None,  # VAE Model
        tokenizer=None,  # CLIPTokenizer
        image_processor=None,
        vae_scale_factor: Union[None, int] = None,  #
        sample_size: Union[None, int] = None,  # sample_size for the UNet
        batch_size: Union[None, int] = None,
    ):
        self.img_width = width
        self.img_height = height
        self.all_heat_maps = (
            AggregateCollection([RawHeatMapCollection() for _ in range(batch_size)])
            if batch_size is not None
            else AggregateCollection([RawHeatMapCollection()])
        )

        if pipeline is not None and isinstance(pipeline, StableDiffusionPipeline):
            self.unet = pipeline.unet
            self.vae = pipeline.vae
            self.tokenizer = pipeline.tokenizer
            self.vae_scale_factor = pipeline.vae_scale_factor

            self.sample_size = (
                sample_size
                if sample_size is not None
                else pipeline.unet.config.sample_size
            )
            self.pipe = pipeline
            self.image_processor = pipeline.feature_extractor
        else:
            # Allow one to to pass unet, tokenizer and vae_scale_factor separately
            assert unet is not None
            assert vae is not None
            assert tokenizer is not None
            assert vae_scale_factor is not None
            assert image_processor is not None
            assert sample_size is not None

            self.unet = unet
            self.tokenizer = tokenizer
            self.vae = vae
            self.vae_scale_factor = vae_scale_factor
            self.sample_size = sample_size
            self.image_processor = image_processor

        h = self.sample_size * self.vae_scale_factor
        self.latent_hw = (
            4096 if h == 512 else 9216
        )  # 64x64 or 96x96 depending on if it's 2.0-v or 2.0
        locate_middle = load_heads or save_heads
        self.locator = UNetCrossAttentionLocator(
            restrict={0} if low_memory else None,
            locate_middle_block=locate_middle,
        )
        self.last_prompts: List[str] = [""]
        self.last_images: List[Image] = [None]
        self.time_idx = 0
        self._gen_idx = 0

        modules = [
            UNetCrossAttentionHooker(
                x,
                self,
                img_width=width,
                img_height=height,
                layer_idx=idx,
                latent_hw=self.latent_hw,
                load_heads=load_heads,
                save_heads=save_heads,
                data_dir=data_dir,
                context_size=context_size,
            )
            for idx, x in enumerate(self.locator.locate(self.unet))
        ]

        if pipeline is not None:
            modules.append(PipelineHooker(pipeline, self))

        modules.append(VAEHooker(self.vae, self, self.image_processor))

        super().__init__(modules)

    def heatmaps(self):
        return self.all_heat_maps

    def set_heatmaps(self, heatmaps):
        self.all_heat_maps = heatmaps

    def time_callback(self, *args, **kwargs):
        self.time_idx += 1

    @property
    def layer_names(self):
        return self.locator.layer_names

    def to_experiments(self, path, seed=None, id=".", subtype=".", **compute_kwargs):
        # type: (Union[Path, str], int, str, str, Dict[str, Any]) -> GenerationExperiment
        """Exports the last generation call to a serializable generation experiment."""

        return [
            GenerationExperiment(
                self.last_images,
                self.compute_global_heat_map(**compute_kwargs).heat_maps,
                self.last_prompt,
                seed=seed,
                id=id,
                subtype=subtype,
                path=path,
                tokenizer=self.tokenizer,
            )
            for image in self.last_images
        ]

    def compute_global_heat_map(
        self,
        prompt=None,
        batch_idx=None,
        head_idx=None,
        layer_idx=None,
        normalize=False,
    ):
        # type: (str, List[float],  int, int, bool) -> GlobalHeatMap
        """
        Compute the global heat map for the given prompt, aggregating across time (inference steps) and space (different
        spatial transformer block heat maps).

        Args:
            prompt: The prompt to compute the heat map for. If none, uses the last prompt that was used for generation.
            factors: Restrict the application to heat maps with spatial factors in this set. If `None`, use all sizes.
            head_idx: Restrict the application to heat maps with this head index. If `None`, use all heads.
            layer_idx: Restrict the application to heat maps with this layer index. If `None`, use all layers.

        Returns:
            A heat map object for computing word-level heat maps.
        """
        batch_heat_maps = self.heatmaps()

        assert prompt is not None or len(self.last_prompt) > 0

        if prompt is None:
            prompt = self.last_prompt

        all_merges = []

        with auto_autocast(dtype=torch.float32):
            batch_merges = []
            for heat_maps in batch_heat_maps:
                merges = []
                # print(f"heamaps inside batch {len(heat_maps)}")
                for (layer, head), heat_map in heat_maps:
                    # print(f"layer {layer} head {head} heat_map size {heat_map.size()}")
                    if (head_idx is None or head_idx == head) and (
                        layer_idx is None or layer_idx == layer
                    ):
                        merges.append(heat_map)

                w, h = get_max_tensor_width_height(merges)

                # we want to interpolate the dimensions so they are all the same size
                for i, merge in enumerate(merges):
                    # The clamping fixes undershoot.
                    heat_map = F.interpolate(
                        merge.unsqueeze(0).permute(1, 0, 3, 2),
                        size=(w, h),
                        mode="bicubic",
                    ).clamp_(min=0)
                    merges[i] = heat_map

                all_merges.append(merges)

        try:
            all_maps = torch.stack([torch.stack(x, 0) for x in all_merges], dim=0)
        except RuntimeError:
            if head_idx is not None or layer_idx is not None:
                raise RuntimeError("No heat maps found for the given parameters.")
            else:
                raise RuntimeError(
                    "No heat maps found. Did you forget to call `with trace(...)` during generation?"
                )

        # print("all maps" , all_maps.size())

        all_all_maps = []
        for i, maps in enumerate(all_maps):
            maps = maps.mean(0)[:, 0]
            maps = maps[
                : len(self.tokenizer.tokenize(prompt)) + 2
            ]  # 1 for SOS and 1 for padding

            if normalize:
                maps = maps / (
                    maps[1:-1].sum(0, keepdim=True) + 1e-6
                )  # drop out [SOS] and [PAD] for proper probabilities

            # print("merging maps", maps.size())
            all_all_maps.append(maps)

        # print("all all maps", len(all_all_maps), [m.size() for m in all_all_maps])
        return GlobalHeatMap(self.tokenizer, prompt, all_all_maps)


class VAEHooker(ObjectHooker[AutoencoderKL]):
    def __init__(self, vae: AutoencoderKL, parent_trace: "trace", image_processor):
        super().__init__(vae)
        self.parent_trace = parent_trace

    def _hooked_decode(
        hk_self, self: AutoencoderKL, z: torch.FloatTensor, *args, **kwargs
    ):
        output = hk_self.monkey_super("decode", z, *args, **kwargs)

        images = []
        # Outputs are all the possible batches being decoded
        for imgs in output:
            # Each batch can have multiple images
            for img in imgs:
                if len(img.size()) == 2:
                    images.append(
                        to_pil_image(
                            img.unsqueeze(0).permute(1, 2, 0).cpu(), do_rescale=True
                        )
                    )
                else:
                    images.append(to_pil_image(img.squeeze().cpu(), do_rescale=True))

        hk_self.parent_trace.last_images = images
        return output

    def _hook_impl(self):
        self.monkey_patch("decode", self._hooked_decode)


class PipelineHooker(ObjectHooker[StableDiffusionPipeline]):
    def __init__(self, pipeline: StableDiffusionPipeline, parent_trace: "trace"):
        super().__init__(pipeline)
        # self.heat_maps = parent_trace.all_heat_maps
        self.parent_trace = parent_trace

    def _hooked_encode_prompt(
        hk_self,
        _: StableDiffusionPipeline,
        prompt: Union[str, List[str]],
        device,
        num_images_per_prompt,
        *args,
        **kwargs,
    ):
        if not isinstance(prompt, str) and len(prompt) > 1:
            raise ValueError(
                "Only single prompt generation is supported for heat map computation."
            )
        elif not isinstance(prompt, str):
            last_prompt = prompt[0]
        else:
            last_prompt = prompt

        # # create batched heatmaps
        # hk_self.parent_trace.all_heat_maps =
        hk_self.parent_trace.set_heatmaps(
            AggregateCollection(
                [RawHeatMapCollection() for _ in range(num_images_per_prompt)]
            )
        )

        # print("setup heatmaps collection", hk_self.parent_trace.heatmaps())

        hk_self.parent_trace.last_prompt = last_prompt
        ret = hk_self.monkey_super(
            "encode_prompt", prompt, device, num_images_per_prompt, *args, **kwargs
        )

        return ret

    def _hook_impl(self):
        self.monkey_patch("encode_prompt", self._hooked_encode_prompt)


class UNetCrossAttentionHooker(ObjectHooker[Attention]):
    def __init__(
        self,
        module: Attention,
        parent_trace: "trace",
        context_size: int = 77,
        img_width: int = 512,
        img_height: int = 512,
        layer_idx: int = 0,
        latent_hw: int = 9216,
        load_heads: bool = False,
        save_heads: bool = False,
        data_dir: Union[str, Path] = None,
    ):
        super().__init__(module)
        self.parent_trace = parent_trace
        self.context_size = context_size
        self.layer_idx = layer_idx
        self.latent_hw = latent_hw

        self.load_heads = load_heads
        self.save_heads = save_heads
        self.trace = parent_trace

        self.img_height = img_height
        self.img_width = img_width

        self.original_processor = None

        if data_dir is not None:
            data_dir = Path(data_dir)
        else:
            data_dir = cache_dir() / "heads"

        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def _unravel_attn(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        # x shape: (heads, height * width, tokens)
        """
        Unravels the attention, returning it as a collection of heat maps.

        Args:
            x (`torch.Tensor`): cross attention slice/map between the words and the tokens.
            value (`torch.Tensor`): the value tensor.

        Returns:
            `List[Tuple[int, torch.Tensor]]`: the list of heat maps across heads.
        """
        h = int(math.ceil(math.sqrt(x.size(1) * self.img_width / self.img_height)))
        w = int(math.ceil(math.sqrt(x.size(1) * self.img_height / self.img_width)))
        maps = []
        x = x.permute(2, 0, 1)

        with auto_autocast(dtype=torch.float32):
            for i, map_ in enumerate(x):
                map_ = map_.view(map_.size(0), w, h)
                map_ = map_[map_.size(0) // 2 :]  # Filter out unconditional
                maps.append(map_)

        maps = torch.stack(maps, 0)  # shape: (tokens, heads, height, width)
        return maps.permute(
            1, 0, 2, 3
        ).contiguous()  # shape: (heads, tokens, width, height)

    def _save_attn(self, attn_slice: torch.Tensor):
        torch.save(attn_slice, self.data_dir / f"{self.trace._gen_idx}.pt")

    def _load_attn(self) -> torch.Tensor:
        return torch.load(self.data_dir / f"{self.trace._gen_idx}.pt")

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        """Capture attentions and aggregate them."""
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = prepare_attention_mask(
            attn, attention_mask, sequence_length, batch_size
        )
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif hasattr(attn, "norm_cross") and attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = head_to_batch_dim(attn, query)
        key = head_to_batch_dim(attn, key)
        value = head_to_batch_dim(attn, value)

        attention_probs = get_attention_scores(attn, query, key, attention_mask)

        # DAAM save heads
        if self.save_heads:
            self._save_attn(attention_probs)
        elif self.load_heads:
            attention_probs = self._load_attn()

        # compute shape factor
        self.trace._gen_idx += 1

        if attention_probs.shape[-1] == self.context_size:
            maps = self._unravel_attn(attention_probs)

            # print(query.shape, attention_probs.shape, batch_size, maps.size())
            # print(attention_probs.shape[0], "//", batch_size, " > ", maps.shape[0])
            # for head_idx, heatmap in enumerate(maps):
            # breakpoint()
            for batch_idx, batch in enumerate(
                maps.vsplit(attention_probs.shape[0] // maps.shape[0])
                if attention_probs.shape[0] // batch_size < maps.shape[0]
                else [maps]
            ):
                for head_idx, heatmap in enumerate(batch):
                    # print("heatmaps: ", len(self.parent_trace.heatmaps()))
                    self.parent_trace.heatmaps()[batch_idx].update(
                        self.layer_idx, head_idx, heatmap
                    )

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = batch_to_head_dim(attn, hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    def _hook_impl(self):
        if hasattr(self.module, "processor") and callable(
            getattr(self.module, "processor")
        ):
            self.original_processor = self.module.processor
            self.module.set_processor(self)
        else:
            self.monkey_patch("forward", comp_vis_attention(self, self.module))

    def _unhook_impl(self):
        if self.original_processor is not None:
            self.module.set_processor(self.original_processor)
        else:
            self.unhook()

    @property
    def num_heat_maps(self):
        return len(next(iter(self.heat_maps.values())))


def comp_vis_attention(_call, attn):
    def attention_call(*args, **kwargs):
        attention_mask = kwargs.mask if "mask" in kwargs else None
        encoder_hidden_states = kwargs["context"] if "context" in kwargs else None

        # Arguments come in all weird at times so we are breaking up a pattern here
        if len(args) == 2:
            return _call(
                attn=args[0],
                hidden_states=args[1],
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )
        else:
            return _call(
                attn=attn,
                hidden_states=args[0],
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )

    return attention_call


def prepare_attention_mask(
    attn,
    attention_mask: torch.Tensor,
    target_length: int,
    batch_size: int,
    out_dim: int = 3,
) -> torch.Tensor:
    r"""
    Prepare the attention mask for the attention computation.

    Args:
        attention_mask (`torch.Tensor`):
            The attention mask to prepare.
        target_length (`int`):
            The target length of the attention mask. This is the length of the attention mask after padding.
        batch_size (`int`):
            The batch size, which is used to repeat the attention mask.
        out_dim (`int`, *optional*, defaults to `3`):
            The output dimension of the attention mask. Can be either `3` or `4`.

    Returns:
        `torch.Tensor`: The prepared attention mask.
    """
    head_size = attn.heads
    if attention_mask is None:
        return attention_mask

    current_length: int = attention_mask.shape[-1]
    if current_length != target_length:
        if attention_mask.device.type == "mps":
            # HACK: MPS: Does not support padding by greater than dimension of input tensor.
            # Instead, we can manually construct the padding tensor.
            padding_shape = (
                attention_mask.shape[0],
                attention_mask.shape[1],
                target_length,
            )
            padding = torch.zeros(
                padding_shape, dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask = torch.cat([attention_mask, padding], dim=2)
        else:
            # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
            #       we want to instead pad by (0, remaining_length), where remaining_length is:
            #       remaining_length: int = target_length - current_length
            # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
            attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

    if out_dim == 3:
        if attention_mask.shape[0] < batch_size * head_size:
            attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
    elif out_dim == 4:
        attention_mask = attention_mask.unsqueeze(1)
        attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

    return attention_mask


def batch_to_head_dim(attn, tensor: torch.Tensor) -> torch.Tensor:
    r"""
    Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
    is the number of heads initialized while constructing the `Attention` class.

    Args:
        tensor (`torch.Tensor`): The tensor to reshape.

    Returns:
        `torch.Tensor`: The reshaped tensor.
    """
    head_size = attn.heads
    batch_size, seq_len, dim = tensor.shape
    tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
    tensor = tensor.permute(0, 2, 1, 3).reshape(
        batch_size // head_size, seq_len, dim * head_size
    )
    return tensor


def head_to_batch_dim(attn, tensor: torch.Tensor, out_dim: int = 3) -> torch.Tensor:
    r"""
    Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
    the number of heads initialized while constructing the `Attention` class.

    Args:
        tensor (`torch.Tensor`): The tensor to reshape.
        out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
            reshaped to `[batch_size * heads, seq_len, dim // heads]`.

    Returns:
        `torch.Tensor`: The reshaped tensor.
    """
    head_size = attn.heads
    batch_size, seq_len, dim = tensor.shape
    tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
    tensor = tensor.permute(0, 2, 1, 3)

    if out_dim == 3:
        tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)

    return tensor


def get_attention_scores(
    attn, query: torch.Tensor, key: torch.Tensor, attention_mask: torch.Tensor = None
) -> torch.Tensor:
    r"""
    Compute the attention scores.

    Args:
        query (`torch.Tensor`): The query tensor.
        key (`torch.Tensor`): The key tensor.
        attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

    Returns:
        `torch.Tensor`: The attention probabilities/scores.
    """
    dtype = query.dtype
    if hasattr(attn, "upcast_attention"):
        query = query.float()
        key = key.float()

    if attention_mask is None:
        baddbmm_input = torch.empty(
            query.shape[0],
            query.shape[1],
            key.shape[1],
            dtype=query.dtype,
            device=query.device,
        )
        beta = 0
    else:
        baddbmm_input = attention_mask
        beta = 1

    attention_scores = torch.baddbmm(
        baddbmm_input,
        query,
        key.transpose(-1, -2),
        beta=beta,
        alpha=attn.scale,
    )
    del baddbmm_input

    if hasattr(attn, "upcast_softmax") and attn.upcast_softmax:
        attention_scores = attention_scores.float()

    attention_probs = attention_scores.softmax(dim=-1)
    del attention_scores

    attention_probs = attention_probs.to(dtype)

    return attention_probs


trace: Type[DiffusionHeatMapHooker] = DiffusionHeatMapHooker

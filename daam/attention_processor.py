import torch


# Translating attention from from CompVis codebase model to Hugging Face attention format
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


# Source: huggingface/diffusers
# Apache License 2.0
# Modified to be used outside the class for AttentionProcessor


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

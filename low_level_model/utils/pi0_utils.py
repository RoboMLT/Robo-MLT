"""PI0 Utility Functions.

This module provides utility functions for the PI0 model:
- Dtype handling for device compatibility
- Sinusoidal positional embeddings for flow matching
- Vector padding for variable-length inputs
- Attention mask construction
- Image resizing with aspect-ratio-preserving padding
"""

import logging
import math

import torch
import torch.nn.functional as F
from torch import Tensor


def get_safe_dtype(dtype: torch.dtype, device: str | torch.device) -> torch.dtype:
    """Get a device-compatible dtype, falling back to float32 if necessary.

    Args:
        dtype: Requested dtype.
        device: Target device.

    Returns:
        The original dtype if supported, otherwise float32.
    """
    if isinstance(device, torch.device):
        device = device.type

    if device == "mps" and dtype == torch.float64:
        return torch.float32

    if device == "xpu" and dtype == torch.float64:
        if hasattr(torch.xpu, "get_device_capability"):
            cap = torch.xpu.get_device_capability()
            if not cap.get("has_fp64", False):
                logging.warning("Device xpu does not support float64, using float32.")
                return torch.float32
        else:
            logging.warning("xpu capability check failed; assuming no float64 support.")
            return torch.float32

    return dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Create sinusoidal positional embeddings for flow-matching timesteps.

    Used in flow matching to encode the diffusion timestep t ∈ [0, 1]. Supports
    an arbitrary number of leading dims, so ``time`` may be a per-sample scalar
    ``[B]`` (baseline pi0) *or* a per-token timestep ``[B, T]`` (TTRTC, where each
    action token carries its own flow timestep). The embedding dimension is always
    appended as a new trailing axis.

    Args:
        time: Timesteps with any leading shape ``[*L]`` (e.g. ``[B]`` or ``[B, T]``).
        dimension: Embedding dimension (must be even).
        min_period: Minimum sinusoidal period.
        max_period: Maximum sinusoidal period.
        device: Target device.

    Returns:
        Positional embeddings ``[*L, dimension]``.
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    dtype = get_safe_dtype(torch.float64, device.type if isinstance(device, torch.device) else device)

    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    # Broadcast the frequency axis against ``time``'s trailing dim so any leading
    # shape (``[B]`` or ``[B, T]``) is handled uniformly.
    sin_input = scaling_factor * time.to(dtype)[..., None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)
    return pos_emb


def pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    """Pad the last dimension of a vector to a target size.

    Args:
        vector: Input tensor [..., features].
        new_dim: Target size for the last dimension.

    Returns:
        Padded tensor with zeros.
    """
    if vector.shape[-1] == new_dim:
        return vector

    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim

    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def build_attention_mask_and_position_ids(
    pad_masks: torch.Tensor,
    att_masks: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build 4D attention mask and position IDs for transformer.

    Args:
        pad_masks: Boolean mask [B, N], True for real tokens.
        att_masks: Block structure mask [B, N].
        dtype: Output dtype for attention mask.

    Returns:
        attention_mask: Additive mask [B, 1, N, N] (0 or -inf).
        position_ids: Position indices [B, N].
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks

    position_ids = torch.cumsum(pad_masks, dim=1) - 1

    mask_value = torch.finfo(dtype).min
    attention_mask = torch.where(
        att_2d_masks,
        torch.zeros_like(att_2d_masks, dtype=dtype),
        torch.full_like(att_2d_masks, mask_value, dtype=dtype),
    )
    attention_mask = attention_mask.unsqueeze(1)

    return attention_mask, position_ids


def resize_with_pad(
    img: Tensor,
    width: int,
    height: int,
    pad_value: float = -1,
) -> Tensor:
    """Resize image preserving aspect ratio with center padding.

    Args:
        img: 4-D image tensor [B, C, H, W] or [B, H, W, C].
        width: Target width.
        height: Target height.
        pad_value: Padding fill value (default -1 for SigLIP after [-1,1] norm).

    Returns:
        Resized and center-padded image tensor.
    """
    if img.ndim != 4:
        raise ValueError(f"4-D tensor expected, got shape {img.shape}")

    channels_last = img.shape[-1] <= 4
    if channels_last:
        img = img.permute(0, 3, 1, 2)

    cur_height, cur_width = img.shape[2], img.shape[3]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_h = max(0, height - resized_height)
    pad_w = max(0, width - resized_width)
    pad_h0, pad_h1 = pad_h // 2, pad_h - pad_h // 2
    pad_w0, pad_w1 = pad_w // 2, pad_w - pad_w // 2

    padded_img = F.pad(resized_img, (pad_w0, pad_w1, pad_h0, pad_h1), value=pad_value)

    if channels_last:
        padded_img = padded_img.permute(0, 2, 3, 1)

    return padded_img


def build_shared_obs_attention_mask_and_position_ids(
    prefix_pad_masks: torch.Tensor,
    prefix_att_masks: torch.Tensor,
    suffix_pad_masks: torch.Tensor,
    suffix_att_masks: torch.Tensor,
    num_offsets: int,
    offset_mask: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build attention mask and position IDs for shared observation training.

    Args:
        prefix_pad_masks: Padding mask for prefix [B, prefix_length].
        prefix_att_masks: Attention structure mask for prefix [B, prefix_length].
        suffix_pad_masks: Padding mask for one suffix [B, suffix_length].
        suffix_att_masks: Attention structure mask for one suffix [B, suffix_length].
        num_offsets: Number of offset branches.
        offset_mask: Boolean mask [B, num_offsets] indicating valid offsets.
        dtype: Output dtype for attention mask.

    Returns:
        attention_mask: Additive mask [B, 1, total_length, total_length].
        position_ids: Position indices [B, total_length].
    """
    batch_size = prefix_pad_masks.shape[0]
    prefix_length = prefix_pad_masks.shape[1]
    suffix_length = suffix_pad_masks.shape[1]
    total_length = prefix_length + suffix_length * num_offsets
    device = prefix_pad_masks.device
    mask_value = torch.finfo(dtype).min

    full_pad_masks = torch.zeros(batch_size, total_length, dtype=torch.bool, device=device)
    full_att_masks = torch.zeros(batch_size, total_length, dtype=prefix_att_masks.dtype, device=device)

    full_pad_masks[:, :prefix_length] = prefix_pad_masks
    full_att_masks[:, :prefix_length] = prefix_att_masks

    suffix_pad_tiled = suffix_pad_masks.unsqueeze(1).expand(-1, num_offsets, -1).reshape(batch_size, -1)
    suffix_att_tiled = suffix_att_masks.unsqueeze(1).expand(-1, num_offsets, -1).reshape(batch_size, -1)
    full_pad_masks[:, prefix_length:] = suffix_pad_tiled
    full_att_masks[:, prefix_length:] = suffix_att_tiled

    cumsum = torch.cumsum(full_att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = full_pad_masks[:, None, :] * full_pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks

    suffix_positions = torch.arange(num_offsets * suffix_length, device=device)
    offset_ids = suffix_positions // suffix_length
    query_offset_ids = offset_ids.unsqueeze(1)
    key_offset_ids = offset_ids.unsqueeze(0)
    cross_offset_mask = query_offset_ids == key_offset_ids

    suffix_start = prefix_length
    att_2d_masks[:, suffix_start:, suffix_start:] = (
        att_2d_masks[:, suffix_start:, suffix_start:] & cross_offset_mask
    )

    offset_validity = offset_mask.unsqueeze(2).expand(-1, -1, suffix_length).reshape(batch_size, -1)
    att_2d_masks[:, suffix_start:, :] = att_2d_masks[:, suffix_start:, :] & offset_validity.unsqueeze(2)
    att_2d_masks[:, :, suffix_start:] = att_2d_masks[:, :, suffix_start:] & offset_validity.unsqueeze(1)

    position_ids = torch.zeros(batch_size, total_length, dtype=torch.long, device=device)
    prefix_pos = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1
    position_ids[:, :prefix_length] = prefix_pos

    last_prefix_pos = prefix_pos[:, -1]
    suffix_pos_base = torch.cumsum(suffix_pad_masks.long(), dim=1)
    suffix_pos_tiled = suffix_pos_base.unsqueeze(1).expand(-1, num_offsets, -1).reshape(batch_size, -1)
    position_ids[:, prefix_length:] = last_prefix_pos[:, None] + suffix_pos_tiled

    attention_mask = torch.where(
        att_2d_masks,
        torch.zeros_like(att_2d_masks, dtype=dtype),
        torch.full_like(att_2d_masks, mask_value, dtype=dtype),
    )
    attention_mask = attention_mask.unsqueeze(1)

    return attention_mask, position_ids


__all__ = [
    "get_safe_dtype",
    "create_sinusoidal_pos_embedding",
    "pad_vector",
    "build_attention_mask_and_position_ids",
    "build_shared_obs_attention_mask_and_position_ids",
    "resize_with_pad",
]

"""Rotary position embeddings."""

from __future__ import annotations

import torch
from torch import Tensor


def apply_rope(tensor: Tensor, position_offset: int = 0, base: float = 10000.0) -> Tensor:
    """Rotate adjacent feature pairs of a ``(B, H, T, D)`` tensor."""
    head_dim = tensor.size(-1)
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    positions = torch.arange(
        position_offset,
        position_offset + tensor.size(-2),
        device=tensor.device,
        dtype=torch.float32,
    )
    inv_frequency = base ** (
        -torch.arange(0, head_dim, 2, device=tensor.device, dtype=torch.float32)
        / head_dim
    )
    angles = torch.outer(positions, inv_frequency)
    cos = angles.cos().to(tensor.dtype)[None, None, :, :]
    sin = angles.sin().to(tensor.dtype)[None, None, :, :]
    even = tensor[..., 0::2]
    odd = tensor[..., 1::2]
    output = torch.empty_like(tensor)
    output[..., 0::2] = even * cos - odd * sin
    output[..., 1::2] = even * sin + odd * cos
    return output

"""Attention layers implemented without ``nn.MultiheadAttention``."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
    *,
    return_attention: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Compute scaled dot-product attention.

    Args:
        query, key, value: tensors whose final dimensions are respectively
            ``(..., query_length, d_k)``, ``(..., key_length, d_k)`` and
            ``(..., key_length, d_v)``.
        mask: a tensor broadcastable to the attention score shape. Boolean
            ``True`` entries are blocked. A floating-point mask is treated as
            an additive score mask.
        return_attention: also return the softmax attention probabilities.
    """
    if query.size(-1) != key.size(-1):
        raise ValueError("query and key must have the same head dimension")
    if key.size(-2) != value.size(-2):
        raise ValueError("key and value must have the same sequence length")

    scale = 1.0 / math.sqrt(query.size(-1))
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    if mask is not None:
        mask = mask.to(device=scores.device)
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(mask, float("-inf"))
        else:
            scores = scores + mask.to(dtype=scores.dtype)

    attention = torch.softmax(scores, dim=-1)
    # A fully masked query row has no valid probability distribution. Returning
    # zeros is safer than allowing NaNs to propagate through later layers.
    attention = torch.nan_to_num(attention, nan=0.0)
    output = torch.matmul(attention, value)
    if return_attention:
        return output, attention
    return output


class MultiHeadAttention(nn.Module):
    """Multi-head attention using explicit projections and head reshaping."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def _split_heads(self, tensor: Tensor) -> Tensor:
        batch_size, seq_len, _ = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def _merge_heads(self, tensor: Tensor) -> Tensor:
        batch_size, _, seq_len, _ = tensor.shape
        tensor = tensor.transpose(1, 2).contiguous()
        return tensor.reshape(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor | None = None,
        mask: Tensor | None = None,
        *,
        need_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Apply self-attention, or cross-attention when ``key_value`` is set."""
        if query.ndim != 3:
            raise ValueError("query must have shape (batch, sequence, d_model)")
        if key_value is None:
            key_value = query
        if key_value.ndim != 3:
            raise ValueError("key_value must have shape (batch, sequence, d_model)")

        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key_value))
        v = self._split_heads(self.v_proj(key_value))
        context, weights = scaled_dot_product_attention(
            q, k, v, mask=mask, return_attention=True
        )
        # Dropout is applied to probabilities only during training. The weights
        # returned for visualization remain normalized and easy to interpret.
        if self.training:
            context = torch.matmul(self.attention_dropout(weights), v)

        output = self.output_dropout(self.out_proj(self._merge_heads(context)))
        if need_weights:
            return output, weights
        return output

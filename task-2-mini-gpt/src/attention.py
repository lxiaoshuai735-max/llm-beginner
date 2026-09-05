"""Causal multi-head attention with a per-layer KV cache."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .rope import apply_rope

KVCache = tuple[Tensor, Tensor]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2:
            raise ValueError("head dimension must be even for RoPE")
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def _split(self, tensor: Tensor) -> Tensor:
        batch, length, width = tensor.shape
        return tensor.reshape(batch, length, self.n_heads, width // self.n_heads).transpose(1, 2)

    def forward(
        self,
        inputs: Tensor,
        kv_cache: KVCache | None = None,
        *,
        return_cache: bool = False,
    ) -> Tensor | tuple[Tensor, KVCache]:
        q, k, v = self.qkv(inputs).chunk(3, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)
        past_length = 0 if kv_cache is None else kv_cache[0].size(-2)
        q = apply_rope(q, past_length)
        k = apply_rope(k, past_length)
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        query_positions = torch.arange(
            past_length, past_length + inputs.size(1), device=inputs.device
        )[:, None]
        key_positions = torch.arange(k.size(-2), device=inputs.device)[None, :]
        causal_mask = key_positions > query_positions
        scores = scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        probabilities = self.attention_dropout(probabilities)
        context = torch.matmul(probabilities, v)
        context = context.transpose(1, 2).contiguous().reshape(inputs.shape)
        output = self.residual_dropout(self.output(context))
        if return_cache:
            return output, (k, v)
        return output

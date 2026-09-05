"""Transformer encoder block."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer encoder block with residual connections."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        inputs: Tensor,
        mask: Tensor | None = None,
        *,
        need_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        normalized = self.attention_norm(inputs)
        if need_weights:
            attended, weights = self.attention(
                normalized, mask=mask, need_weights=True
            )
        else:
            attended = self.attention(normalized, mask=mask)

        hidden = inputs + attended
        output = hidden + self.ffn(self.ffn_norm(hidden))
        if need_weights:
            return output, weights
        return output


def make_causal_mask(length: int, device: torch.device | None = None) -> Tensor:
    """Return a ``(length, length)`` boolean mask; ``True`` blocks the future."""
    return torch.triu(
        torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
    )

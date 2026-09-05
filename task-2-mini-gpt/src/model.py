"""Decoder-only mini-GPT with RoPE and incremental KV caching."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from .attention import CausalSelfAttention, KVCache
from .sampling import sample_next_token
from .tokenizer import BPETokenizer


@dataclass
class GPTConfig:
    vocab_size: int
    max_seq_len: int = 129
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 6
    d_ff: int = 768
    dropout: float = 0.1


class DecoderBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(
            config.d_model, config.n_heads, config.dropout
        )
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        inputs: Tensor,
        kv_cache: KVCache | None = None,
        *,
        return_cache: bool = False,
    ):
        if return_cache:
            attended, cache = self.attention(
                self.attention_norm(inputs), kv_cache, return_cache=True
            )
        else:
            attended = self.attention(self.attention_norm(inputs), kv_cache)
        hidden = inputs + attended
        output = hidden + self.ffn(self.ffn_norm(hidden))
        return (output, cache) if return_cache else output


class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len
        # eval/run.py creates a block+1-token window for next-token targets.
        self.block_size = config.max_seq_len - 1
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        ids: Tensor,
        kv_cache: list[KVCache] | None = None,
        return_cache: bool = False,
    ):
        if ids.ndim != 2:
            raise ValueError("ids must have shape (batch, sequence)")
        past_length = 0 if not kv_cache else kv_cache[0][0].size(-2)
        if past_length + ids.size(1) > self.max_seq_len:
            raise ValueError("sequence plus cache exceeds max_seq_len")
        hidden = self.dropout(self.embedding(ids))
        new_cache: list[KVCache] = []
        for index, layer in enumerate(self.layers):
            layer_cache = None if kv_cache is None else kv_cache[index]
            if return_cache:
                hidden, updated = layer(hidden, layer_cache, return_cache=True)
                new_cache.append(updated)
            else:
                hidden = layer(hidden, layer_cache)
        logits = self.lm_head(self.final_norm(hidden))
        return (logits, new_cache) if return_cache else logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids,
        max_new_tokens: int,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float = 1.0,
    ) -> Tensor:
        self.eval()
        device = next(self.parameters()).device
        ids = torch.as_tensor(prompt_ids, dtype=torch.long, device=device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.size(1) == 0:
            raise ValueError("prompt must contain at least one token")
        ids = ids[:, -self.block_size :]
        logits, cache = self(ids, return_cache=True)
        for _ in range(max_new_tokens):
            next_id = sample_next_token(
                logits[:, -1], temperature=temperature, top_k=top_k, top_p=top_p
            )
            ids = torch.cat((ids, next_id), dim=1)
            if cache[0][0].size(-2) >= self.max_seq_len:
                context = ids[:, -self.block_size :]
                logits, cache = self(context, return_cache=True)
            else:
                logits, cache = self(next_id, kv_cache=cache, return_cache=True)
        return ids


def save_checkpoint(path: str | Path, model: MiniGPT, tokenizer: BPETokenizer, **metadata) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path = path.parent / "tokenizer.json"
    tokenizer.save_pretrained(tokenizer_path)
    torch.save(
        {
            "config": asdict(model.config),
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "tokenizer_file": tokenizer_path.name,
            **metadata,
        },
        path,
    )


def load_for_eval(ckpt_path: str):
    path = Path(ckpt_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tokenizer = BPETokenizer.from_pretrained(path.parent / payload.get("tokenizer_file", "tokenizer.json"))
    model = MiniGPT(GPTConfig(**payload["config"]))
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, tokenizer

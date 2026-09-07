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
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        tokenizer: BPETokenizer | None = None,
        enforce_utf8: bool = False,
        complete_utf8_at_end: bool = True,
        stop_after_newline: bool = False,
        min_new_tokens: int = 1,
    ) -> Tensor:
        if repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be at least 1.0")
        if no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size cannot be negative")
        if (enforce_utf8 or stop_after_newline) and tokenizer is None:
            raise ValueError("tokenizer is required for UTF-8 or newline-aware generation")
        if min_new_tokens < 1:
            raise ValueError("min_new_tokens must be at least 1")
        self.eval()
        device = next(self.parameters()).device
        ids = torch.as_tensor(prompt_ids, dtype=torch.long, device=device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.size(1) == 0:
            raise ValueError("prompt must contain at least one token")
        ids = ids[:, -self.block_size :]
        logits, cache = self(ids, return_cache=True)
        utf8_states: list[int] | None = None
        if enforce_utf8:
            assert tokenizer is not None
            utf8_states = []
            for row in ids.tolist():
                state = tokenizer.utf8_state_for_ids(row)
                if state is None:
                    raise ValueError("prompt does not form a valid UTF-8 prefix")
                utf8_states.append(state)
        generated_bytes = [bytearray() for _ in range(ids.size(0))]
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1].clone()
            if repetition_penalty > 1.0:
                for batch_index in range(ids.size(0)):
                    seen = ids[batch_index].unique()
                    values = next_logits[batch_index, seen]
                    next_logits[batch_index, seen] = torch.where(
                        values < 0,
                        values * repetition_penalty,
                        values / repetition_penalty,
                    )
            if no_repeat_ngram_size > 0:
                for batch_index in range(ids.size(0)):
                    sequence = ids[batch_index].tolist()
                    prefix_size = no_repeat_ngram_size - 1
                    if prefix_size == 0:
                        banned = set(sequence)
                    elif len(sequence) >= prefix_size:
                        prefix = tuple(sequence[-prefix_size:])
                        banned = {
                            sequence[index + prefix_size]
                            for index in range(len(sequence) - prefix_size)
                            if tuple(sequence[index : index + prefix_size]) == prefix
                        }
                    else:
                        banned = set()
                    if banned and len(banned) < next_logits.size(-1):
                        next_logits[batch_index, list(banned)] = float("-inf")
            if enforce_utf8:
                assert tokenizer is not None and utf8_states is not None
                next_states: list[dict[int, int]] = []
                for batch_index, state in enumerate(utf8_states):
                    allowed, transitions = tokenizer.valid_next_tokens(state)
                    # A byte-BPE token may end halfway through a multi-byte
                    # Chinese character.  On the last decoding step retain
                    # only tokens that close the UTF-8 sequence, so the saved
                    # sample is an exact, complete UTF-8 string rather than a
                    # lossy decoded prefix.
                    if complete_utf8_at_end and _ == max_new_tokens - 1:
                        allowed = [token for token in allowed if transitions[token] == 0]
                    if not allowed:
                        raise RuntimeError("no token can complete the UTF-8 prefix")
                    mask = torch.ones_like(next_logits[batch_index], dtype=torch.bool)
                    mask[allowed] = False
                    next_logits[batch_index, mask] = float("-inf")
                    next_states.append(transitions)
            next_id = sample_next_token(
                next_logits, temperature=temperature, top_k=top_k, top_p=top_p
            )
            if enforce_utf8:
                assert utf8_states is not None
                utf8_states = [
                    next_states[index][int(token.item())]
                    for index, token in enumerate(next_id[:, 0])
                ]
            if stop_after_newline:
                assert tokenizer is not None
                for index, token in enumerate(next_id[:, 0]):
                    generated_bytes[index].extend(tokenizer.token_bytes(int(token.item())))
            ids = torch.cat((ids, next_id), dim=1)
            if (
                stop_after_newline
                and _ + 1 >= min_new_tokens
                and all(b"\n" in payload for payload in generated_bytes)
                and (utf8_states is None or all(state == 0 for state in utf8_states))
            ):
                break
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

"""Character tokenizer and Transformer classifier for ChnSentiCorp."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn

from .block import TransformerBlock


@dataclass
class ModelConfig:
    vocab_size: int
    pad_token_id: int = 0
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    max_len: int = 256
    num_classes: int = 2
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class CharacterTokenizer:
    """A deterministic tokenizer learned only from the task's training split."""

    PAD = "[PAD]"
    UNK = "[UNK]"
    CLS = "[CLS]"
    SEP = "[SEP]"
    SPECIAL_TOKENS = (PAD, UNK, CLS, SEP)

    def __init__(self, id_to_token: Sequence[str]) -> None:
        if list(id_to_token[: len(self.SPECIAL_TOKENS)]) != list(self.SPECIAL_TOKENS):
            raise ValueError("special tokens must occupy the first four vocabulary ids")
        self.id_to_token = list(id_to_token)
        self.token_to_id = {token: index for index, token in enumerate(id_to_token)}

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.PAD]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        max_vocab_size: int = 6000,
        min_frequency: int = 2,
    ) -> "CharacterTokenizer":
        if max_vocab_size <= len(cls.SPECIAL_TOKENS):
            raise ValueError("max_vocab_size is too small for the special tokens")

        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(str(text))
        candidates = [item for item in counts.items() if item[1] >= min_frequency]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        remaining = max_vocab_size - len(cls.SPECIAL_TOKENS)
        vocabulary = list(cls.SPECIAL_TOKENS) + [char for char, _ in candidates[:remaining]]
        return cls(vocabulary)

    def encode(
        self,
        text: str,
        max_length: int | None = None,
        *,
        add_special_tokens: bool = True,
    ) -> list[int]:
        tokens = list(str(text))
        if max_length is not None:
            reserved = 2 if add_special_tokens else 0
            tokens = tokens[: max(0, max_length - reserved)]
        ids = [self.token_to_id.get(token, self.token_to_id[self.UNK]) for token in tokens]
        if add_special_tokens:
            ids = [self.token_to_id[self.CLS], *ids, self.token_to_id[self.SEP]]
        return ids

    def convert_ids_to_tokens(self, ids: Iterable[int]) -> list[str]:
        return [self.id_to_token[int(index)] for index in ids]

    def to_dict(self) -> dict:
        return {"id_to_token": self.id_to_token}

    @classmethod
    def from_dict(cls, payload: dict) -> "CharacterTokenizer":
        return cls(payload["id_to_token"])


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        even_indices = torch.arange(0, d_model, 2, dtype=torch.float32)
        rates = torch.exp(-even_indices * torch.log(torch.tensor(10000.0)) / d_model)
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(positions * rates)
        if d_model > 1:
            encoding[:, 1::2] = torch.cos(positions * rates[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, inputs: Tensor) -> Tensor:
        seq_len = inputs.size(1)
        if seq_len > self.encoding.size(0):
            raise ValueError(
                f"sequence length {seq_len} exceeds max_len {self.encoding.size(0)}"
            )
        return inputs + self.encoding[:seq_len].to(dtype=inputs.dtype).unsqueeze(0)


class TransformerClassifier(nn.Module):
    """A Transformer encoder with ``[CLS]`` pooling for binary classification."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.position_encoding = SinusoidalPositionEncoding(
            config.d_model, config.max_len
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, config.num_classes)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        return_attentions: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        if input_ids.size(1) > self.config.max_len:
            raise ValueError("input sequence is longer than the configured max_len")

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        # Broadcast over heads and query positions. True means blocked.
        padding_mask = ~attention_mask.to(dtype=torch.bool, device=input_ids.device)
        padding_mask = padding_mask[:, None, None, :]

        hidden = self.token_embedding(input_ids) * (self.config.d_model**0.5)
        hidden = self.embedding_dropout(self.position_encoding(hidden))
        all_attentions: list[Tensor] = []
        for layer in self.layers:
            if return_attentions:
                hidden, weights = layer(
                    hidden, mask=padding_mask, need_weights=True
                )
                all_attentions.append(weights)
            else:
                hidden = layer(hidden, mask=padding_mask)

        hidden = self.final_norm(hidden)
        logits = self.classifier(hidden[:, 0])
        if return_attentions:
            return logits, all_attentions
        return logits


def save_checkpoint(
    path: str | Path,
    model: TransformerClassifier,
    tokenizer: CharacterTokenizer,
    **metadata,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_config": model.config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        **metadata,
    }
    torch.save(payload, path)


def load_checkpoint(
    ckpt_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[TransformerClassifier, CharacterTokenizer, dict]:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ModelConfig(**payload["model_config"])
    tokenizer = CharacterTokenizer.from_dict(payload["tokenizer"])
    model = TransformerClassifier(config)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"model_state", "model_config", "tokenizer"}
    }
    return model, tokenizer, metadata


def load_for_eval(ckpt_path: str) -> tuple[TransformerClassifier, callable]:
    """Factory required by ``eval/run.py``."""
    model, tokenizer, _ = load_checkpoint(ckpt_path, device="cpu")
    model.eval()
    max_len = model.config.max_len

    def tokenize_fn(text: str) -> Tensor:
        return torch.tensor(
            tokenizer.encode(text, max_length=max_len), dtype=torch.long
        )

    return model, tokenize_fn

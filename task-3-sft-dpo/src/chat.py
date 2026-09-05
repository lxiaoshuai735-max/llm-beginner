"""Qwen chat formatting and assistant-only loss masking."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor


def format_messages(messages: list[dict]) -> str:
    chunks = []
    for message in messages:
        role = str(message["role"])
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported chat role: {role}")
        chunks.append(
            f"<|im_start|>{role}\n{str(message['content'])}<|im_end|>\n"
        )
    return "".join(chunks)


def _tokenizer():
    from transformers import AutoTokenizer

    model_path = Path(__file__).resolve().parents[1] / "models/Qwen2.5-0.5B"
    return AutoTokenizer.from_pretrained(str(model_path))


def build_labels(input_ids: Tensor, messages: list[dict], tokenizer=None) -> Tensor:
    """Mask templates and non-assistant turns with ``-100``."""
    if input_ids.ndim != 1:
        raise ValueError("input_ids must be one-dimensional")
    tokenizer = tokenizer or _tokenizer()
    labels = torch.full_like(input_ids, -100)
    prefix = ""
    full_ids = tokenizer(format_messages(messages), add_special_tokens=False).input_ids
    offset = max(0, input_ids.numel() - len(full_ids))
    for message in messages:
        role = str(message["role"])
        content = str(message["content"])
        header = f"<|im_start|>{role}\n"
        chunk = f"{header}{content}<|im_end|>\n"
        if role == "assistant":
            start = len(tokenizer(prefix + header, add_special_tokens=False).input_ids) + offset
            end = len(tokenizer(prefix + header + content, add_special_tokens=False).input_ids) + offset
            end = min(end, input_ids.numel())
            if start < end:
                labels[start:end] = input_ids[start:end]
        prefix += chunk
    return labels


def tokenize_conversation(tokenizer, messages: list[dict], max_length: int = 512):
    text = format_messages(messages)
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).input_ids[0]
    labels = build_labels(encoded, messages, tokenizer=tokenizer)
    return encoded, labels

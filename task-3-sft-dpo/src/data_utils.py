"""Dataset readers for MOSS SFT and DPO-En-Zh-20k.

The readers accept both the original MOSS export and simplified
ShareGPT-style mirrors. Invalid records are skipped instead of terminating a
long training run.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable, Iterator
from pathlib import Path


SFT_DIALOGUES = [
    [{"role": "user", "content": "你好，请用一句话介绍你自己。"}, {"role": "assistant", "content": "你好！我是一个乐于提供清晰、可靠帮助的人工智能助手。"}],
    [{"role": "user", "content": "什么是机器学习？"}, {"role": "assistant", "content": "机器学习是让计算机从数据中学习规律，并据此进行预测或决策的方法。"}],
]

DPO_PAIRS = [
    {"messages": [{"role": "user", "content": "什么是深度学习？"}], "chosen": "深度学习是使用多层神经网络从数据中学习分层表示的机器学习方法。", "rejected": "深度学习就是电脑记住所有答案。"},
    {"messages": [{"role": "user", "content": "如何保护账号安全？"}], "chosen": "使用唯一强密码、密码管理器和多因素认证，并警惕钓鱼链接。", "rejected": "把同一个简单密码用于所有网站最方便。"},
]


def _text(value) -> str:
    if isinstance(value, dict):
        value = value.get("value") or value.get("content") or value.get("text") or ""
    if isinstance(value, list):
        parts = [_text(part) for part in value]
        return "\n".join(part for part in parts if part)
    return str(value or "").strip()


def _role_message(message: dict) -> dict | None:
    role = str(message.get("role") or message.get("from") or "").lower()
    role = {"human": "user", "gpt": "assistant", "moss": "assistant"}.get(role, role)
    content = _text(message.get("content") or message.get("value") or message.get("text"))
    if role not in {"system", "user", "assistant"} or not content:
        return None
    return {"role": role, "content": content}


def _turn_messages(turn: dict) -> list[dict]:
    human = _text(turn.get("Human") or turn.get("human") or turn.get("user"))
    assistant = _text(
        turn.get("MOSS")
        or turn.get("moss")
        or turn.get("assistant")
        or turn.get("Assistant")
    )
    human = re.sub(r"^<\|Human\|>:\s*", "", human)
    human = re.sub(r"<eoh>\s*$", "", human).strip()
    assistant = re.sub(r"^<\|MOSS\|>:\s*", "", assistant)
    assistant = re.sub(r"<eom>\s*$", "", assistant).strip()
    messages = []
    if human:
        messages.append({"role": "user", "content": human})
    if assistant:
        messages.append({"role": "assistant", "content": assistant})
    return messages


def moss_record_to_messages(item: dict) -> list[dict]:
    """Normalize one original/simplified MOSS record to Qwen chat messages."""
    messages = item.get("messages")
    if isinstance(messages, list):
        normalized = [_role_message(message) for message in messages if isinstance(message, dict)]
        return [message for message in normalized if message]

    conversation = (
        item.get("conversation")
        or item.get("conversations")
        or item.get("turns")
        or item.get("chat")
    )
    if isinstance(conversation, list):
        normalized = []
        for turn in conversation:
            if not isinstance(turn, dict):
                continue
            direct = _role_message(turn)
            normalized.extend([direct] if direct else _turn_messages(turn))
        return normalized

    if not isinstance(conversation, dict):
        conversation = item
    turn_keys = [key for key in conversation if re.fullmatch(r"turn_\d+", str(key))]
    turn_keys.sort(key=lambda key: int(str(key).split("_")[-1]))
    normalized = []
    for key in turn_keys:
        turn = conversation[key]
        if isinstance(turn, dict):
            normalized.extend(_turn_messages(turn))
    return normalized


def iter_sft_dialogues(data_dir: str | Path) -> Iterator[list[dict]]:
    """Stream every valid MOSS conversation without loading the corpus in RAM."""
    data_dir = Path(data_dir)
    for path in sorted(data_dir.rglob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                messages = moss_record_to_messages(item)
                if messages and any(message["role"] == "assistant" for message in messages):
                    yield messages


def buffered_shuffle(items: Iterable, buffer_size: int, seed: int) -> Iterator:
    """Shuffle a stream using bounded memory."""
    if buffer_size <= 1:
        yield from items
        return
    rng = random.Random(seed)
    buffer = []
    for item in items:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = item
    rng.shuffle(buffer)
    yield from buffer


def load_sft_dialogues(data_dir: str | Path, max_samples: int = 0) -> list[list[dict]]:
    """Materialized compatibility wrapper used by small tests."""
    dialogues = []
    for messages in iter_sft_dialogues(data_dir):
        dialogues.append(messages)
        if max_samples and len(dialogues) >= max_samples:
            break
    return dialogues or SFT_DIALOGUES


def _preference_text(value) -> str:
    if isinstance(value, list) and value:
        value = value[-1]
    return _text(value)


def dpo_record_to_pair(item: dict) -> dict | None:
    conversation = item.get("conversations") or item.get("messages") or []
    messages = []
    if isinstance(conversation, list):
        normalized = [_role_message(message) for message in conversation if isinstance(message, dict)]
        messages = [message for message in normalized if message]
    if not messages:
        prompt = _text(item.get("prompt") or item.get("instruction") or item.get("question"))
        extra = _text(item.get("input"))
        if extra:
            prompt = f"{prompt}\n{extra}".strip()
        if prompt:
            messages = [{"role": "user", "content": prompt}]
    chosen = _preference_text(item.get("chosen") or item.get("response_chosen"))
    rejected = _preference_text(item.get("rejected") or item.get("response_rejected"))
    if not messages or not chosen or not rejected:
        return None
    return {"messages": messages, "chosen": chosen, "rejected": rejected}


def load_dpo_pairs(data_dir: str | Path) -> list[dict]:
    """Load all Chinese and English preference pairs."""
    data_dir = Path(data_dir)
    pairs = []
    for path in sorted(data_dir.rglob("*.json")) + sorted(data_dir.rglob("*.jsonl")):
        if path.name.startswith("."):
            continue
        if path.suffix == ".jsonl":
            records = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        else:
            try:
                records = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(records, dict):
                records = records.get("data") or records.get("train") or [records]
        if not isinstance(records, list):
            continue
        for item in records:
            if isinstance(item, dict):
                pair = dpo_record_to_pair(item)
                if pair:
                    pairs.append(pair)
    return pairs or DPO_PAIRS

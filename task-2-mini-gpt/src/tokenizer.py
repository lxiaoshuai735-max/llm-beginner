"""A small byte-level BPE tokenizer with lossless UTF-8 round trips."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _merge_pair(sequence: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    merged: list[int] = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
            merged.append(new_id)
            index += 2
        else:
            merged.append(sequence[index])
            index += 1
    return merged


class BPETokenizer:
    """Byte-level BPE; every possible input remains exactly decodable."""

    def __init__(self, merges: list[tuple[int, int]] | None = None) -> None:
        self.merges = list(merges or [])
        self._token_bytes: list[bytes] = [bytes([value]) for value in range(256)]
        for left, right in self.merges:
            if left >= len(self._token_bytes) or right >= len(self._token_bytes):
                raise ValueError("invalid merge table: token referenced before creation")
            self._token_bytes.append(self._token_bytes[left] + self._token_bytes[right])

    @property
    def vocab_size(self) -> int:
        return len(self._token_bytes)

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int = 512,
        min_pair_frequency: int = 2,
    ) -> "BPETokenizer":
        if vocab_size < 256:
            raise ValueError("byte-level BPE requires vocab_size >= 256")
        sequence = list(text.encode("utf-8"))
        merges: list[tuple[int, int]] = []
        while 256 + len(merges) < vocab_size and len(sequence) > 1:
            counts = Counter(zip(sequence, sequence[1:]))
            if not counts:
                break
            pair, frequency = min(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
            if frequency < min_pair_frequency:
                break
            new_id = 256 + len(merges)
            sequence = _merge_pair(sequence, pair, new_id)
            merges.append(pair)
        return cls(merges)

    def encode(self, text: str) -> list[int]:
        sequence = list(str(text).encode("utf-8"))
        for rank, pair in enumerate(self.merges):
            sequence = _merge_pair(sequence, pair, 256 + rank)
        return sequence

    def decode(self, ids: list[int]) -> str:
        try:
            payload = b"".join(self._token_bytes[int(index)] for index in ids)
        except IndexError as exc:
            raise ValueError("token id is outside the vocabulary") from exc
        # Valid encoded text still round-trips exactly. During unconstrained
        # byte-level generation the final token can stop midway through a UTF-8
        # code point, so use a replacement marker instead of failing the run.
        return payload.decode("utf-8", errors="replace")

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "type": "byte_bpe",
                    "vocab_size": self.vocab_size,
                    "merges": [list(pair) for pair in self.merges],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([tuple(map(int, pair)) for pair in payload["merges"]])

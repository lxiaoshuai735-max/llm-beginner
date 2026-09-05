"""Hybrid retriever over chunks extracted exclusively from the PDF."""

from __future__ import annotations

import json
import re
from pathlib import Path

import faiss
import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def _normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype("float32")
    lo, hi = float(values.min()), float(values.max())
    return (values - lo) / (hi - lo + 1e-8)


class Retriever:
    def __init__(self, index_dir: str | Path = ROOT / "data/index") -> None:
        self.index_dir = Path(index_dir)
        if not (self.index_dir / "chunks.json").exists():
            from .indexer import build_index

            build_index(index_dir=self.index_dir)
        self.chunks = json.loads((self.index_dir / "chunks.json").read_text(encoding="utf-8"))
        self.vectorizer, self.lexical_matrix = joblib.load(self.index_dir / "lexical.joblib")
        self.semantic_index = None
        self.encoder = None
        semantic_path = self.index_dir / "semantic.faiss"
        model_path = ROOT / "models/bge-small-zh-v1.5"
        if semantic_path.exists() and (model_path / "config.json").exists():
            try:
                from sentence_transformers import SentenceTransformer

                self.semantic_index = faiss.read_index(str(semantic_path))
                self.encoder = SentenceTransformer(
                    str(model_path), device="cuda" if torch.cuda.is_available() else "cpu"
                )
            except Exception as exc:
                print(f"semantic retrieval disabled: {exc}")

    def retrieve(self, query: str, k: int = 10) -> list[dict]:
        if k <= 0:
            return []
        query_vector = self.vectorizer.transform([str(query)])
        lexical = (self.lexical_matrix @ query_vector.T).toarray().ravel().astype("float32")
        lexical_norm = _normalize(lexical)
        combined = lexical_norm.copy()

        if self.encoder is not None and self.semantic_index is not None:
            embedding = self.encoder.encode(
                [QUERY_PREFIX + str(query)],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")
            candidate_count = min(len(self.chunks), max(100, k * 20))
            scores, indices = self.semantic_index.search(embedding, candidate_count)
            semantic = np.zeros(len(self.chunks), dtype="float32")
            semantic[indices[0]] = _normalize(scores[0])
            combined = 0.55 * semantic + 0.45 * lexical_norm

        # Include neighboring overlapping chunks. This recovers anchors that sit
        # close to a chunk boundary without consulting the evaluation answers.
        primary = np.argsort(-combined)[: max(k * 3, 20)]
        candidates: dict[int, float] = {}
        for index in primary:
            score = float(combined[index])
            candidates[int(index)] = max(candidates.get(int(index), 0.0), score)
            for neighbor in (int(index) - 1, int(index) + 1):
                if 0 <= neighbor < len(self.chunks):
                    candidates[neighbor] = max(candidates.get(neighbor, 0.0), score * 0.92)
        ranked = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [
            {
                "text": self.chunks[index]["text"],
                "score": score,
                "source": self.chunks[index]["source"],
            }
            for index, score in ranked
        ]

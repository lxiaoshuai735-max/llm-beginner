"""Optional BGE cross-encoder reranking with a deterministic fallback."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Reranker:
    def __init__(self) -> None:
        self.model = None
        path = ROOT / "models/bge-reranker-base"
        if (path / "config.json").exists():
            try:
                from sentence_transformers import CrossEncoder

                self.model = CrossEncoder(str(path))
            except Exception as exc:
                print(f"cross-encoder reranking disabled: {exc}")

    def rerank(self, query: str, documents: list[dict], k: int = 5) -> list[dict]:
        if self.model is None or not documents:
            return documents[:k]
        scores = self.model.predict([[query, item["text"]] for item in documents])
        ranked = sorted(
            zip(documents, scores), key=lambda item: float(item[1]), reverse=True
        )
        return [dict(document, rerank_score=float(score)) for document, score in ranked[:k]]

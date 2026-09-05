"""End-to-end RAG entry point required by the evaluator."""

from __future__ import annotations

from .generator import Generator
from .reranker import Reranker
from .retriever import Retriever

_retriever = None
_reranker = None
_generator = None


def answer(query: str) -> dict:
    global _retriever, _reranker, _generator
    _retriever = _retriever or Retriever()
    _reranker = _reranker or Reranker()
    _generator = _generator or Generator()
    recalled = _retriever.retrieve(query, k=20)
    sources = _reranker.rerank(query, recalled, k=5)
    response = _generator.generate(query, sources)
    return {"answer": response, "sources": sources}

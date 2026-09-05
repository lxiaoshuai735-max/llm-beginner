"""Build a hybrid BGE/FAISS and character TF-IDF index from ``kb.pdf``."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from .chunker import chunk_text, extract_pdf_text

ROOT = Path(__file__).resolve().parents[1]


def build_index(
    pdf_path: str | Path = ROOT / "data/kb.pdf",
    index_dir: str | Path = ROOT / "data/index",
    chunk_size: int = 768,
    overlap: int = 192,
) -> dict:
    pdf_path, index_dir = Path(pdf_path), Path(index_dir)
    if not pdf_path.exists():
        raise FileNotFoundError(f"knowledge-base PDF not found: {pdf_path}")
    index_dir.mkdir(parents=True, exist_ok=True)
    text = extract_pdf_text(pdf_path)
    raw_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    chunks = [
        {"id": index, "text": chunk, "source": f"kb.pdf#chunk-{index}"}
        for index, chunk in enumerate(raw_chunks)
    ]
    (index_dir / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
    )

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=1,
        max_features=60000,
        sublinear_tf=True,
        norm="l2",
    )
    lexical_matrix = vectorizer.fit_transform(raw_chunks)
    joblib.dump((vectorizer, lexical_matrix), index_dir / "lexical.joblib")

    model_path = ROOT / "models/bge-small-zh-v1.5"
    backend = "lexical"
    if (model_path / "config.json").exists():
        try:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(
                str(model_path), device="cuda" if torch.cuda.is_available() else "cpu"
            )
            embeddings = encoder.encode(
                raw_chunks,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
            ).astype("float32")
            faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
            faiss_index.add(embeddings)
            faiss.write_index(faiss_index, str(index_dir / "semantic.faiss"))
            backend = "bge+faiss+tfidf"
        except Exception as exc:
            print(f"BGE indexing unavailable; lexical index remains usable: {exc}")
    config = {
        "source": str(pdf_path),
        "chunks": len(chunks),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "backend": backend,
    }
    (index_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False))
    return config

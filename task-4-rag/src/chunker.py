"""PDF extraction and overlapping character chunks."""

from __future__ import annotations

import re
from pathlib import Path


def extract_pdf_text(pdf_path: str | Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages = []
    for page_number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        pages.append(f"\n[第{page_number}页]\n{text.strip()}")
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split by character count while retaining overlap across boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    text = str(text).strip()
    if not text:
        return []
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks

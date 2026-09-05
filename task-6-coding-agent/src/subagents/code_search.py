"""Read-only code-search subagent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..mcp_server import read_file
from ..model_client import LocalQwenClient


class CodeSearchSubagent:
    """Inspect candidate source files, then return only a compact summary."""

    def __init__(self, client: LocalQwenClient | None = None) -> None:
        self.client = client or LocalQwenClient()

    def run(self, repo_path: str, issue: str) -> dict[str, Any]:
        root = Path(repo_path).resolve()
        candidates = [
            path.relative_to(root).as_posix()
            for path in sorted(root.rglob("*.py"))
            if ".git" not in path.parts and not path.name.startswith("test_")
        ][:12]
        excerpts = []
        for relative in candidates:
            try:
                excerpts.append(f"## {relative}\n{read_file(repo_path, relative)[:3000]}")
            except Exception as exc:
                excerpts.append(f"## {relative}\nERROR: {exc}")
        prompt = (
            "You are a read-only code-search subagent. Identify likely files and suspicious lines. "
            "Do not propose changes to tests. Return a concise plain-text summary.\n\n"
            f"Issue:\n{issue}\n\nFiles:\n" + "\n\n".join(excerpts)
        )
        try:
            summary = self.client.chat([{"role": "user", "content": prompt}], max_tokens=350)
        except Exception as exc:
            summary = f"Model summary unavailable ({type(exc).__name__}). Candidate implementation files: {', '.join(candidates)}"
        return {"summary": summary, "candidate_files": candidates}

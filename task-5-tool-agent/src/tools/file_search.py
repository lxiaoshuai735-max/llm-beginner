"""Search filenames and text beneath the Task 5 directory."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "file_search",
        "description": "Search safe local project files by filename/glob or text content.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Filename, glob, or text to find"},
                "dir": {"type": "string", "description": "Directory under the Task 5 project"},
            },
            "required": ["pattern"],
        },
    },
}

_ROOT = Path(__file__).resolve().parents[2]
_TEXT_SUFFIXES = {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".toml", ".csv"}


def _safe_directory(value: str) -> Path:
    candidate = Path(value) if value else _ROOT
    if not candidate.is_absolute():
        candidate = _ROOT / candidate
    candidate = candidate.resolve()
    if candidate != _ROOT and _ROOT not in candidate.parents:
        raise ValueError("search directory must stay inside the Task 5 project")
    if not candidate.is_dir():
        raise ValueError(f"directory does not exist: {candidate}")
    return candidate


def run(args: dict[str, Any]) -> str:
    pattern = str(args.get("pattern", "")).strip()
    if not pattern or len(pattern) > 200:
        raise ValueError("pattern must contain 1-200 characters")
    directory = _safe_directory(str(args.get("dir", "")))
    name_query = any(char in pattern for char in "*?[]") or "." in pattern
    matches: list[str] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.stat().st_size > 1024 * 1024:
            continue
        relative = path.relative_to(_ROOT).as_posix()
        if fnmatch.fnmatch(path.name.lower(), pattern.lower()) or fnmatch.fnmatch(relative.lower(), pattern.lower()):
            preview = ""
            if path.suffix.lower() in _TEXT_SUFFIXES:
                try:
                    preview = " ".join(path.read_text(encoding="utf-8", errors="replace")[:500].split())
                except OSError:
                    pass
            matches.append(f"FILE {relative}" + (f": {preview}" if preview else ""))
            continue
        if not name_query and path.suffix.lower() in _TEXT_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            index = text.lower().find(pattern.lower())
            if index >= 0:
                snippet = " ".join(text[max(0, index - 80): index + len(pattern) + 120].split())
                matches.append(f"CONTENT {relative}: {snippet}")
        if len(matches) >= 50:
            break
    if not matches:
        return f"No matches for {pattern!r} under {directory.relative_to(_ROOT).as_posix() or '.'}."
    return f"Found {len(matches)} match(es):\n" + "\n".join(matches)

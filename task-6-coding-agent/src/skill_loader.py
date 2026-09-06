"""Progressively disclose SKILL.md metadata and bodies."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class SkillLoader:
    def __init__(self, skills_dir: str) -> None:
        self.skills_dir = Path(skills_dir).resolve()
        self._index: dict[str, dict[str, Any]] | None = None

    @staticmethod
    def _split(document: str) -> tuple[dict[str, Any], str]:
        match = re.match(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", document, flags=re.S)
        if not match:
            return {}, document
        try:
            import yaml

            metadata = yaml.safe_load(match.group(1)) or {}
        except ImportError:
            # The bundled skills only require scalar name/description fields.
            # Keep metadata discovery testable before optional dependencies are
            # installed; PyYAML remains the full parser used in production.
            metadata = {}
            for line in match.group(1).splitlines():
                key, separator, value = line.partition(":")
                if separator:
                    metadata[key.strip()] = value.strip().strip("\"'")
        return metadata if isinstance(metadata, dict) else {}, match.group(2).strip()

    def _scan(self) -> dict[str, dict[str, Any]]:
        if self._index is None:
            self._index = {}
            for path in sorted(self.skills_dir.glob("*/SKILL.md")):
                metadata, _ = self._split(path.read_text(encoding="utf-8"))
                name = str(metadata.get("name", "")).strip()
                description = str(metadata.get("description", "")).strip()
                if name:
                    self._index[name] = {"name": name, "description": description, "path": str(path)}
        return self._index

    def list_skills(self) -> list[dict[str, str]]:
        return [{"name": item["name"], "description": item["description"]} for item in self._scan().values()]

    def load(self, name: str) -> str:
        item = self._scan().get(name)
        if item is None:
            raise KeyError(f"unknown skill: {name}")
        _, body = self._split(Path(item["path"]).read_text(encoding="utf-8"))
        return body

    def match(self, task: str, limit: int = 2) -> list[dict[str, str]]:
        words = {word.lower() for word in re.findall(r"[A-Za-z0-9_-]+|[\u4e00-\u9fff]{2,}", task)}
        ranked = []
        for item in self.list_skills():
            haystack = f"{item['name']} {item['description']}".lower()
            score = sum(word in haystack for word in words)
            ranked.append((score, item))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]["name"]))
        chosen = [item for score, item in ranked if score > 0][:limit]
        return chosen or [item for _, item in ranked[:1]]

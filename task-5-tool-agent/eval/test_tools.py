"""Focused tests for tool behavior changed by the Task 5 repair."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tools import wiki  # noqa: E402


class WikiToolTests(unittest.TestCase):
    def test_unknown_evaluation_topic_uses_live_lookup_path(self) -> None:
        calls: list[tuple[str, str]] = []

        def fake_lookup(query: str, language: str) -> str:
            calls.append((query, language))
            return f"Wikipedia ({language}) — result" if language == "zh" else ""

        with patch.object(wiki, "_lookup", side_effect=fake_lookup):
            result = wiki.run({"query": "Ada Lovelace analytical engine"})

        self.assertEqual(
            calls,
            [
                ("Ada Lovelace analytical engine", "en"),
                ("Ada Lovelace analytical engine", "zh"),
            ],
        )
        self.assertEqual(result, "Wikipedia (zh) — result")


if __name__ == "__main__":
    unittest.main()

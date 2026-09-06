"""Read-only code-search subagent whose repository access goes through MCP."""

from __future__ import annotations

import re
from typing import Any

from ..model_client import LocalQwenClient


class CodeSearchSubagent:
    """Inspect candidate source files, then return only a compact summary."""

    def __init__(self, client: LocalQwenClient | None = None) -> None:
        self.client = client or LocalQwenClient()

    async def run(self, repo_path: str, issue: str, tool_client: Any) -> dict[str, Any]:
        search_outputs: list[str] = []
        for query in ("def ", "class ", "import "):
            output = await tool_client.call_tool(
                "search_files",
                {"query": query, "glob": "*.py"},
                repo_path=repo_path,
            )
            search_outputs.append(output)

        candidates: list[str] = []
        for output in search_outputs:
            if output.startswith("ToolError"):
                continue
            for line in output.splitlines():
                match = re.match(r"^(.+\.py):\d+: ", line)
                if not match:
                    continue
                relative = match.group(1)
                lowered_parts = {part.lower() for part in relative.replace("\\", "/").split("/")}
                name = relative.rsplit("/", 1)[-1].lower()
                if (
                    lowered_parts.intersection({"test", "tests"})
                    or name.startswith("test_")
                    or name.endswith("_test.py")
                ):
                    continue
                if relative not in candidates:
                    candidates.append(relative)
                if len(candidates) >= 12:
                    break
            if len(candidates) >= 12:
                break

        excerpts: list[str] = []
        for relative in candidates:
            content = await tool_client.call_tool(
                "read_file", {"path": relative}, repo_path=repo_path
            )
            excerpts.append(f"## {relative}\n{content[:3000]}")
        prompt = (
            "You are a read-only code-search subagent. Identify likely files and suspicious lines. "
            "Do not propose changes to tests. Return a concise plain-text summary.\n\n"
            f"Issue:\n{issue}\n\nFiles:\n" + "\n\n".join(excerpts)
        )
        try:
            summary = self.client.chat([{"role": "user", "content": prompt}], max_tokens=350)
        except Exception as exc:
            summary = (
                f"Model summary unavailable ({type(exc).__name__}). "
                f"Candidate implementation files: {', '.join(candidates) or '(none found)'}"
            )
        return {
            "summary": summary,
            "candidate_files": candidates,
            "mcp_calls": 3 + len(candidates),
        }

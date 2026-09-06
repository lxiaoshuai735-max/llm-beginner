"""Test-execution and failure-diagnosis subagent using MCP."""

from __future__ import annotations

import re
from typing import Any

from ..model_client import LocalQwenClient


class TestDiagnosisSubagent:
    """Run tests over MCP and summarize output in an isolated model context."""

    def __init__(self, client: LocalQwenClient | None = None) -> None:
        self.client = client or LocalQwenClient()

    async def run(self, repo_path: str, issue: str, tool_client: Any) -> dict[str, Any]:
        output = await tool_client.call_tool("run_tests", {}, repo_path=repo_path)
        prompt = (
            "You are a test-diagnosis subagent. Summarize the failing behavior and the likely "
            "implementation contract in at most six lines. Never recommend weakening tests.\n\n"
            f"Issue:\n{issue}\n\nPytest output:\n{output}"
        )
        try:
            summary = self.client.chat([{"role": "user", "content": prompt}], max_tokens=300)
        except Exception as exc:
            summary = f"Model diagnosis unavailable ({type(exc).__name__}); inspect raw pytest output."
        match = re.match(r"\Aexit_code=(\d+)(?:\r?\n|\Z)", output)
        return {
            "summary": summary,
            "test_output": output,
            "tests_passed": match is not None and int(match.group(1)) == 0,
            "mcp_calls": 1,
        }

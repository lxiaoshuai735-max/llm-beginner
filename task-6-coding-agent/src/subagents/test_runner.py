"""Test-execution and failure-diagnosis subagent."""

from __future__ import annotations

from typing import Any

from ..mcp_server import run_tests
from ..model_client import LocalQwenClient


class TestDiagnosisSubagent:
    """Run tests and summarize output in an isolated model context."""

    def __init__(self, client: LocalQwenClient | None = None) -> None:
        self.client = client or LocalQwenClient()

    def run(self, repo_path: str, issue: str) -> dict[str, Any]:
        output = run_tests(repo_path)
        prompt = (
            "You are a test-diagnosis subagent. Summarize the failing behavior and the likely "
            "implementation contract in at most six lines. Never recommend weakening tests.\n\n"
            f"Issue:\n{issue}\n\nPytest output:\n{output}"
        )
        try:
            summary = self.client.chat([{"role": "user", "content": prompt}], max_tokens=300)
        except Exception as exc:
            summary = f"Model diagnosis unavailable ({type(exc).__name__}); inspect raw pytest output."
        return {"summary": summary, "test_output": output, "tests_passed": "exit_code=0" in output}

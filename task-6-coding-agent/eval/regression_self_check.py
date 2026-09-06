"""Dependency-light regression check for the agent's completion semantics.

The pytest suite additionally exercises real pytest and, when installed, a real
MCP stdio round trip.  This script uses a transport double so it can validate
the varied-bug and disabled-model branches before optional packages are set up.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import tool_helpers
from src.agent import CodingAgent


class ToolClientDouble:
    transport_name = "mcp-test-double"

    async def __aenter__(self) -> "ToolClientDouble":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def list_tools(self) -> list[dict[str, Any]]:
        schemas = {
            "read_file": {"path": {"type": "string"}},
            "write_file": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "run_tests": {},
            "git_diff": {},
            "git_apply": {"patch": {"type": "string"}},
            "search_files": {
                "query": {"type": "string"},
                "glob": {"type": "string"},
            },
        }
        return [
            {
                "name": name,
                "description": name,
                "input_schema": {"type": "object", "properties": properties},
            }
            for name, properties in schemas.items()
        ]

    async def call_tool(
        self, name: str, arguments: dict[str, Any], repo_path: str
    ) -> str:
        if name == "run_tests":
            source = (Path(repo_path) / "worker.py").read_text(encoding="utf-8")
            return "exit_code=0\n1 passed" if "left * right" in source else "exit_code=1\n1 failed"
        function = getattr(tool_helpers, name)
        try:
            return function(repo_path=repo_path, **arguments)
        except Exception as exc:
            return f"ToolError: {type(exc).__name__}: {exc}"


class ScriptedModel:
    model = "scripted-self-check"

    def __init__(self, decisions: list[dict[str, Any]] | None = None) -> None:
        self.decisions = iter(
            decisions
            or [
                {"tool": "read_file", "arguments": {"path": "worker.py"}},
                {"tool": "run_tests", "arguments": {}},
                {
                    "tool": "write_file",
                    "arguments": {
                        "path": "worker.py",
                        "content": "def multiply(left, right):\n    return left * right\n",
                    },
                },
                {"tool": "run_tests", "arguments": {}},
                {"tool": "git_diff", "arguments": {}},
            ]
        )

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 500) -> str:
        prompt = messages[-1]["content"]
        if "code-search subagent" in prompt:
            return "Inspect worker.py."
        if "test-diagnosis subagent" in prompt:
            return "The multiplication assertion fails."
        return json.dumps(next(self.decisions))


class DisabledModel:
    model = "disabled-self-check"

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 500) -> str:
        raise ConnectionError("disabled for regression check")


def prepare_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir(parents=True)
    (repo / "worker.py").write_text(
        "def multiply(left, right):\n    return left + right\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "self-check@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Self Check"], cwd=repo, check=True)
    subprocess.run(["git", "add", "worker.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "bug baseline"], cwd=repo, check=True)
    return repo


def prepare_dirty_passing_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir(parents=True)
    (repo / "worker.py").write_text(
        "def multiply(left, right):\n    return left * right\n", encoding="utf-8"
    )
    (repo / "notes.txt").write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "self-check@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Self Check"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "clean baseline"], cwd=repo, check=True)
    (repo / "notes.txt").write_text("pre-existing dirty change\n", encoding="utf-8")
    return repo


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="task6-self-check-") as temporary:
        success_repo = prepare_repo(Path(temporary) / "success")
        success = CodingAgent(
            client=ScriptedModel(),
            tool_client_factory=ToolClientDouble,
            max_steps=7,
        ).run(str(success_repo), "multiply incorrectly performs addition")
        assert success["success"] is True
        assert success["tests_passed"] is True
        assert "left * right" in success["patch"]

        failure_repo = prepare_repo(Path(temporary) / "failure")
        before = (failure_repo / "worker.py").read_text(encoding="utf-8")
        failure = CodingAgent(
            client=DisabledModel(),
            tool_client_factory=ToolClientDouble,
            max_steps=6,
        ).run(str(failure_repo), "multiply incorrectly performs addition")
        assert failure["success"] is False
        assert failure["status"] == "failed"
        assert failure["patch"] == ""
        assert (failure_repo / "worker.py").read_text(encoding="utf-8") == before

        dirty_repo = prepare_dirty_passing_repo(Path(temporary) / "dirty")
        original = "def multiply(left, right):\n    return left * right\n"
        dirty_decisions = [
            {"tool": "read_file", "arguments": {"path": "worker.py"}},
            {"tool": "run_tests", "arguments": {}},
            {
                "tool": "write_file",
                "arguments": {"path": "worker.py", "content": original + "# temporary\n"},
            },
            {"tool": "write_file", "arguments": {"path": "worker.py", "content": original}},
            {"tool": "run_tests", "arguments": {}},
            {"tool": "git_diff", "arguments": {}},
            {"done": True},
        ]
        dirty = CodingAgent(
            client=ScriptedModel(dirty_decisions),
            tool_client_factory=ToolClientDouble,
            max_steps=7,
        ).run(str(dirty_repo), "review multiplication")
        assert dirty["status"] == "failed"
        assert dirty["baseline_patch"] not in {"", "(no diff)"}
        assert dirty["patch"] == ""

        ordered_repo = prepare_repo(Path(temporary) / "ordered")
        ordered_decisions = [
            {"tool": "read_file", "arguments": {"path": "worker.py"}},
            {"tool": "run_tests", "arguments": {}},
            {
                "tool": "write_file",
                "arguments": {
                    "path": "worker.py",
                    "content": "def multiply(left, right):\n    return left * right\n",
                },
            },
            {"tool": "git_diff", "arguments": {}},
            {"tool": "run_tests", "arguments": {}},
            {"done": True},
        ]
        ordered = CodingAgent(
            client=ScriptedModel(ordered_decisions),
            tool_client_factory=ToolClientDouble,
            max_steps=7,
        ).run(str(ordered_repo), "repair multiplication")
        assert ordered["status"] == "succeeded"
        main_names = [step["tool_call"]["name"] for step in ordered["steps"][2:]]
        assert main_names.count("git_diff") == 2
        assert main_names[-1] == "git_diff"

    print("REGRESSION_SELF_CHECK_OK")


if __name__ == "__main__":
    main()

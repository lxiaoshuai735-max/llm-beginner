"""Regression checks for truthful completion and generic bug handling."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from src import tool_helpers
from src.agent import CodingAgent
from src.mcp_client import MCPStdioClient


TOOLS = [
    ("read_file", {"path": {"type": "string"}}, ["path"]),
    (
        "write_file",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ["path", "content"],
    ),
    ("run_tests", {}, []),
    ("git_diff", {}, []),
    ("git_apply", {"patch": {"type": "string"}}, ["patch"]),
    (
        "search_files",
        {"query": {"type": "string"}, "glob": {"type": "string"}},
        ["query"],
    ),
]


class InProcessMCPDouble:
    """A small transport double; production uses MCPStdioClient."""

    transport_name = "mcp-test-double"

    async def __aenter__(self) -> "InProcessMCPDouble":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "description": name,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    **({"required": required} if required else {}),
                },
            }
            for name, properties, required in TOOLS
        ]

    async def call_tool(
        self, name: str, arguments: dict[str, Any], repo_path: str
    ) -> str:
        function = getattr(tool_helpers, name)
        try:
            return function(repo_path=repo_path, **arguments)
        except Exception as exc:
            return f"ToolError: {type(exc).__name__}: {exc}"


class ScriptedModel:
    model = "scripted-regression-model"

    def __init__(self, decisions: list[dict[str, Any] | str]) -> None:
        self.decisions = iter(decisions)

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 500) -> str:
        prompt = messages[-1]["content"]
        if "read-only code-search subagent" in prompt:
            return "worker.py contains the implementation under test."
        if "test-diagnosis subagent" in prompt:
            return "The multiply contract fails for nontrivial operands."
        decision = next(self.decisions)
        return decision if isinstance(decision, str) else json.dumps(decision)


class DisabledModel:
    model = "disabled-model"

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 500) -> str:
        raise ConnectionError("model service intentionally disabled")


def make_bug_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "varied-bug-repo"
    repo.mkdir()
    (repo / "worker.py").write_text(
        "def multiply(left, right):\n    return left + right\n", encoding="utf-8"
    )
    (repo / "test_worker.py").write_text(
        "from worker import multiply\n\n"
        "def test_multiply():\n"
        "    assert multiply(3, 4) == 12\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Regression Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "bug baseline"], cwd=repo, check=True)
    return repo


def make_dirty_passing_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "dirty-passing-repo"
    repo.mkdir()
    (repo / "worker.py").write_text(
        "def multiply(left, right):\n    return left * right\n", encoding="utf-8"
    )
    (repo / "test_worker.py").write_text(
        "from worker import multiply\n\n"
        "def test_multiply():\n"
        "    assert multiply(3, 4) == 12\n",
        encoding="utf-8",
    )
    (repo / "notes.txt").write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Regression Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "clean baseline"], cwd=repo, check=True)
    (repo / "notes.txt").write_text("pre-existing dirty change\n", encoding="utf-8")
    return repo


def varied_bug_decisions() -> list[dict[str, Any]]:
    return [
        {"thought": "inspect implementation", "tool": "read_file", "arguments": {"path": "worker.py"}},
        {"thought": "confirm baseline", "tool": "run_tests", "arguments": {}},
        {
            "thought": "repair multiplication",
            "tool": "write_file",
            "arguments": {
                "path": "worker.py",
                "content": "def multiply(left, right):\n    return left * right\n",
            },
        },
        {"thought": "verify repair", "tool": "run_tests", "arguments": {}},
        {"thought": "record patch", "tool": "git_diff", "arguments": {}},
    ]


def test_varied_bug_closes_only_after_model_authored_edit(tmp_path: Path) -> None:
    repo = make_bug_repo(tmp_path)
    trace = CodingAgent(
        client=ScriptedModel(varied_bug_decisions()),
        tool_client_factory=InProcessMCPDouble,
        max_steps=7,
    ).run(str(repo), "multiply returns addition instead of a product")

    assert trace["status"] == "succeeded", trace
    assert trace["success"] is True
    assert trace["tests_passed"] is True
    assert "return left * right" in trace["patch"]
    assert (repo / "worker.py").read_text(encoding="utf-8").endswith("left * right\n")
    main_steps = trace["steps"][2:]
    assert main_steps
    assert all(step["tool_call"]["transport"] == "mcp-test-double" for step in main_steps)
    assert trace["tool_transport"] == "mcp-test-double"


def test_disabled_model_is_an_explicit_failure_and_never_edits(tmp_path: Path) -> None:
    repo = make_bug_repo(tmp_path)
    before = (repo / "worker.py").read_text(encoding="utf-8")
    trace = CodingAgent(
        client=DisabledModel(),
        tool_client_factory=InProcessMCPDouble,
        max_steps=6,
    ).run(str(repo), "multiply returns addition instead of a product")

    assert trace["status"] == "failed"
    assert trace["success"] is False
    assert trace["tests_passed"] is False
    assert trace["patch"] == ""
    assert "deterministic repair is forbidden" in trace["failure_reason"]
    assert (repo / "worker.py").read_text(encoding="utf-8") == before
    assert not any(
        step["tool_call"]["name"] in {"write_file", "git_apply"}
        for step in trace["steps"]
    )


def test_completion_parser_uses_the_reported_exit_code_only() -> None:
    assert CodingAgent._tests_succeeded("exit_code=0\n3 passed") is True
    assert CodingAgent._tests_succeeded("exit_code=1\ntest printed exit_code=0") is False


def test_test_directories_are_protected_from_both_write_tools() -> None:
    assert CodingAgent._protected_write(
        "write_file", {"path": "tests/helpers.py", "content": ""}
    )


def test_preexisting_diff_plus_write_then_restore_cannot_report_success(tmp_path: Path) -> None:
    repo = make_dirty_passing_repo(tmp_path)
    original = "def multiply(left, right):\n    return left * right\n"
    decisions = [
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
    trace = CodingAgent(
        client=ScriptedModel(decisions),
        tool_client_factory=InProcessMCPDouble,
        max_steps=7,
    ).run(str(repo), "review multiplication without changing existing behavior")

    assert trace["baseline_patch"] not in {"", "(no diff)"}
    assert trace["status"] == "failed"
    assert trace["patch"] == ""
    assert (repo / "worker.py").read_text(encoding="utf-8") == original


def test_diff_must_be_collected_after_the_final_successful_test(tmp_path: Path) -> None:
    repo = make_bug_repo(tmp_path)
    decisions = [
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
    trace = CodingAgent(
        client=ScriptedModel(decisions),
        tool_client_factory=InProcessMCPDouble,
        max_steps=7,
    ).run(str(repo), "multiply returns addition instead of a product")

    assert trace["status"] == "succeeded", trace
    main_names = [step["tool_call"]["name"] for step in trace["steps"][2:]]
    assert main_names.count("git_diff") == 2
    last_test_index = max(index for index, name in enumerate(main_names) if name == "run_tests")
    last_diff_index = max(index for index, name in enumerate(main_names) if name == "git_diff")
    assert last_test_index < last_diff_index
    assert CodingAgent._protected_write(
        "git_apply",
        {"patch": "--- a/tests/helpers.py\n+++ b/tests/helpers.py\n@@ -1 +1 @@\n-a\n+b\n"},
    )


def test_invalid_json_and_one_repeated_read_only_action_can_recover(tmp_path: Path) -> None:
    repo = make_bug_repo(tmp_path)
    decisions: list[dict[str, Any] | str] = [
        {"tool": "read_file", "arguments": {"path": "worker.py"}},
        '{"thought":"collect evidence","tool":"run_tests","arguments":',
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
    trace = CodingAgent(
        client=ScriptedModel(decisions),
        tool_client_factory=InProcessMCPDouble,
        max_steps=8,
    ).run(str(repo), "multiply returns addition instead of a product")

    assert trace["status"] == "succeeded"
    assert "return left * right" in trace["patch"]
    main_steps = trace["steps"][2:]
    assert sum(step["tool_call"]["name"] == "run_tests" for step in main_steps) == 3
    assert any(
        "model returned invalid JSON" in step["thought"]
        for step in main_steps
    )


@pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="MCP runtime is not installed")
def test_real_mcp_stdio_round_trip(tmp_path: Path) -> None:
    repo = make_bug_repo(tmp_path)

    async def round_trip() -> tuple[list[dict[str, Any]], str]:
        async with MCPStdioClient() as client:
            tools = await client.list_tools()
            content = await client.call_tool("read_file", {"path": "worker.py"}, str(repo))
            return tools, content

    tools, content = asyncio.run(round_trip())
    assert {tool["name"] for tool in tools} == {name for name, _, _ in TOOLS}
    assert all("repo_path" not in tool["input_schema"].get("properties", {}) for tool in tools)
    assert "def multiply" in content


@pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="MCP runtime is not installed")
def test_varied_bug_runs_end_to_end_over_real_mcp_stdio(tmp_path: Path) -> None:
    repo = make_bug_repo(tmp_path)
    trace = CodingAgent(
        client=ScriptedModel(varied_bug_decisions()),
        max_steps=7,
    ).run(str(repo), "multiply returns addition instead of a product")

    assert trace["status"] == "succeeded"
    assert trace["tool_transport"] == "mcp-stdio"
    assert "return left * right" in trace["patch"]
    assert all(
        step["tool_call"]["transport"] == "mcp-stdio"
        for step in trace["steps"][2:]
    )

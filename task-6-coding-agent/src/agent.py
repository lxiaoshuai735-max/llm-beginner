"""Handwritten coding-agent loop using MCP tools, Skills and Subagents."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from .mcp_client import MCPStdioClient
from .model_client import LocalQwenClient
from .skill_loader import SkillLoader
from .subagents import CodeSearchSubagent, TestDiagnosisSubagent


class Trace(dict):
    """Dictionary trace with stable evaluator-facing keys."""


_SYSTEM = """You are the main coding agent. Work only inside the supplied repository.
Return exactly one JSON object per turn, with no markdown fence:
{{"thought":"brief reasoning","tool":"tool_name","arguments":{{...}}}}
When tests have passed and the diff has been inspected, return:
{{"thought":"brief summary","done":true}}

Rules:
- Read relevant code and use pytest evidence before editing.
- Never modify tests, snapshots, .git, or paths outside the repository.
- Prefer write_file for a small complete-file edit. Preserve unrelated behavior.
- A task is not done until an implementation change was made, a later run_tests
  reports exit_code=0, and a later git_diff records a non-empty patch.
- Tool errors are observations to correct on the next turn.
- If you cannot derive a justified edit, do not claim success.

Available tools (repo_path is supplied by the host and must not be included):
{tools}
"""


class CodingAgent:
    def __init__(
        self,
        api_url: str | None = None,
        model: str | None = None,
        max_steps: int = 10,
        max_protocol_retries: int = 3,
        *,
        client: Any | None = None,
        tool_client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.client = client or LocalQwenClient(api_url=api_url, model=model)
        self.max_steps = max_steps
        self.max_protocol_retries = max_protocol_retries
        self.skills_dir = Path(__file__).resolve().parent / "skills"
        self.tool_client_factory = tool_client_factory or MCPStdioClient

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
        start = cleaned.find("{")
        if start < 0:
            raise ValueError("model response contains no JSON object")
        value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        if not isinstance(value, dict):
            raise ValueError("model response must be a JSON object")
        return value

    @staticmethod
    def _successful_tool(steps: list[dict[str, Any]], name: str) -> bool:
        return any(
            step.get("tool_call", {}).get("name") == name
            and not str(step.get("observation", "")).startswith("ToolError")
            for step in steps
        )

    @staticmethod
    def _tests_succeeded(observation: str) -> bool:
        match = re.match(r"\Aexit_code=(\d+)(?:\r?\n|\Z)", observation)
        return match is not None and int(match.group(1)) == 0

    @staticmethod
    def _completion_ready(
        *,
        last_mutation_step: int,
        last_test_step: int,
        last_diff_step: int,
        patch: str,
        baseline_patch: str,
    ) -> bool:
        return (
            0 < last_mutation_step < last_test_step < last_diff_step
            and bool(patch)
            and patch != "(no diff)"
            and patch != baseline_patch
        )

    def _fallback_action(
        self,
        fallback_source: str,
        steps: list[dict[str, Any]],
        tests_passed: bool,
        patch: str,
        last_mutation_step: int,
        last_test_step: int,
        last_diff_step: int,
    ) -> tuple[str, dict[str, Any], str] | None:
        """Return only evidence-gathering recovery actions, never a guessed edit."""
        if not self._successful_tool(steps, "read_file"):
            return (
                "read_file",
                {"path": fallback_source},
                "Recover from the model protocol error by reading likely source code.",
            )
        if not self._successful_tool(steps, "run_tests"):
            return (
                "run_tests",
                {},
                "Recover from the model protocol error by collecting test evidence.",
            )
        if last_mutation_step > 0 and (
            not tests_passed or last_test_step <= last_mutation_step
        ):
            return (
                "run_tests",
                {},
                "Verify the latest model-authored change with the full test suite.",
            )
        if (
            last_mutation_step > 0
            and tests_passed
            and last_test_step > last_mutation_step
            and (not patch or last_diff_step <= last_test_step)
        ):
            return (
                "git_diff",
                {},
                "Record the tested model-authored patch before stopping.",
            )
        return None

    @staticmethod
    def _normalize_arguments(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        args = dict(arguments)
        args.pop("repo_path", None)
        if tool_name in {"read_file", "write_file"}:
            raw = str(args.get("path", ""))
            if Path(raw).is_absolute():
                args["path"] = Path(raw).name
        if tool_name == "search_files":
            args.setdefault("glob", "*.py")
        return args

    @staticmethod
    def _protected_write(tool_name: str, arguments: dict[str, Any]) -> str | None:
        if tool_name not in {"write_file", "git_apply"}:
            return None
        if tool_name == "write_file":
            path = Path(str(arguments.get("path", "")))
            lowered_parts = {part.lower() for part in path.parts}
            if (
                lowered_parts.intersection(
                    {".git", "test", "tests", "snapshot", "snapshots", "__snapshots__"}
                )
                or path.name.lower().startswith("test_")
                or path.name.lower().endswith("_test.py")
                or path.name.lower() == "conftest.py"
                or path.name.lower().endswith(".orig")
                or "snapshot" in path.name.lower()
            ):
                return "refusing to modify a test, snapshot, baseline, or .git file"
        else:
            patch = str(arguments.get("patch", ""))
            for line in patch.splitlines():
                if not (line.startswith("+++ b/") or line.startswith("--- a/")):
                    continue
                raw_path = line[6:].split("\t", 1)[0]
                path = PurePosixPath(raw_path)
                lowered_parts = {part.lower() for part in path.parts}
                if (
                    lowered_parts.intersection(
                        {".git", "test", "tests", "snapshot", "snapshots", "__snapshots__"}
                    )
                    or path.name.lower().startswith("test_")
                    or path.name.lower().endswith("_test.py")
                    or path.name.lower() == "conftest.py"
                    or path.name.lower().endswith(".orig")
                    or "snapshot" in path.name.lower()
                ):
                    return "refusing a patch that modifies a protected file"
        return None

    @staticmethod
    def _trace(
        *,
        steps: list[dict[str, Any]],
        patch: str,
        tests_passed: bool,
        model: str,
        skills: list[str],
        subagents: dict[str, str],
        failure_reason: str | None,
        success: bool,
        baseline_patch: str,
        tool_transport: str,
    ) -> Trace:
        return Trace(
            status="succeeded" if success else "failed",
            success=success,
            steps=steps,
            patch=patch,
            baseline_patch=baseline_patch,
            tests_passed=tests_passed,
            failure_reason=None if success else (failure_reason or "completion criteria were not met"),
            model=model,
            tool_transport=tool_transport,
            skills=skills,
            subagents=subagents,
        )

    def run(self, repo_path: str, issue: str) -> Trace:
        """Run the async MCP loop from a normal synchronous entry point."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_async(repo_path, issue))
        raise RuntimeError("CodingAgent.run() cannot be used inside an active event loop; await run_async()")

    async def run_async(self, repo_path: str, issue: str) -> Trace:
        root = Path(repo_path).resolve()
        if not root.is_dir():
            raise ValueError(f"repository does not exist: {root}")

        loader = SkillLoader(str(self.skills_dir))
        selected = loader.match(issue, limit=3)
        skill_context = "\n\n".join(
            f"## Skill: {item['name']}\n{loader.load(item['name'])}" for item in selected
        )
        model_name = str(getattr(self.client, "model", "unknown"))
        steps: list[dict[str, Any]] = []
        subagent_summaries: dict[str, str] = {}
        baseline_patch = ""
        transport_name = "mcp-stdio"

        try:
            async with self.tool_client_factory() as tool_client:
                transport_name = str(getattr(tool_client, "transport_name", "unknown"))
                advertised_tools = await tool_client.list_tools()
                allowed_tools = {tool["name"] for tool in advertised_tools}
                required_tools = {
                    "read_file",
                    "write_file",
                    "run_tests",
                    "git_diff",
                    "git_apply",
                    "search_files",
                }
                missing_tools = required_tools - allowed_tools
                if missing_tools:
                    raise RuntimeError(f"MCP server is missing tools: {sorted(missing_tools)}")

                baseline_patch = await tool_client.call_tool(
                    "git_diff", {}, repo_path=str(root)
                )
                if baseline_patch.startswith("ToolError"):
                    raise RuntimeError(f"cannot capture baseline diff: {baseline_patch}")

                search_result = await CodeSearchSubagent(client=self.client).run(
                    str(root), issue, tool_client
                )
                test_result = await TestDiagnosisSubagent(client=self.client).run(
                    str(root), issue, tool_client
                )
                fallback_source = (
                    search_result["candidate_files"][0]
                    if search_result["candidate_files"]
                    else "README.md"
                )
                subagent_summaries = {
                    "code_search": search_result["summary"],
                    "test_diagnosis": test_result["summary"],
                }
                steps.extend(
                    [
                        {
                            "step": 1,
                            "thought": "Delegate read-only repository discovery to an isolated context.",
                            "tool_call": {
                                "name": "delegate_code_search",
                                "arguments": {"issue": issue},
                                "transport": transport_name,
                            },
                            "observation": (
                                f"MCP calls={search_result['mcp_calls']}\n"
                                f"{search_result['summary']}"
                            ),
                        },
                        {
                            "step": 2,
                            "thought": "Delegate baseline test diagnosis to an isolated context.",
                            "tool_call": {
                                "name": "delegate_test_diagnosis",
                                "arguments": {},
                                "transport": transport_name,
                            },
                            "observation": (
                                f"MCP calls={test_result['mcp_calls']}\n"
                                f"{test_result['summary']}\n{test_result['test_output']}"
                            ),
                        },
                    ]
                )

                tool_prompt = "\n".join(
                    f"- {tool['name']}: {tool['description']} "
                    f"schema={json.dumps(tool['input_schema'])}"
                    for tool in advertised_tools
                )
                messages = [
                    {"role": "system", "content": _SYSTEM.format(tools=tool_prompt)},
                    {
                        "role": "user",
                        "content": (
                            f"Repository: {root.name}\nIssue:\n{issue}\n\nLoaded skills:\n{skill_context}"
                            f"\n\nCode-search summary:\n{search_result['summary']}"
                            f"\n\nTest summary:\n{test_result['summary']}\nBegin with one tool call."
                        ),
                    },
                ]
                tests_passed = False
                patch = ""
                last_mutation_step = 0
                last_test_step = 0
                last_diff_step = 0
                last_signature: str | None = None
                repeat_streak = 0
                protocol_retries = 0
                failure_reason: str | None = None

                def completion_ready() -> bool:
                    return self._completion_ready(
                        last_mutation_step=last_mutation_step,
                        last_test_step=last_test_step,
                        last_diff_step=last_diff_step,
                        patch=patch,
                        baseline_patch=baseline_patch,
                    )

                def fallback_action() -> tuple[str, dict[str, Any], str] | None:
                    return self._fallback_action(
                        fallback_source=fallback_source,
                        steps=steps,
                        tests_passed=tests_passed,
                        patch=patch,
                        last_mutation_step=last_mutation_step,
                        last_test_step=last_test_step,
                        last_diff_step=last_diff_step,
                    )

                def request_protocol_retry(reason: str, response: str) -> None:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": response or "(no usable model response)",
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Your previous response could not be executed: {reason}. "
                                    "Return exactly one valid JSON object and nothing else. "
                                    "Use a listed tool name, an arguments object, and escape all "
                                    "newlines inside JSON strings as \\n. Do not repeat the same "
                                    "action unless repository state changed."
                                ),
                            },
                        ]
                    )

                for _ in range(self.max_steps):
                    model_text = ""
                    model_call_error: Exception | None = None
                    protocol_error: Exception | None = None
                    try:
                        model_text = self.client.chat(messages, max_tokens=550)
                    except Exception as exc:
                        model_call_error = exc
                        decision = {}
                    else:
                        try:
                            decision = self._parse_json(model_text)
                        except Exception as exc:
                            protocol_error = exc
                            decision = {}

                    fallback_reason = ""
                    if decision.get("done"):
                        if completion_ready():
                            break
                        fallback_reason = "model marked done before the verified completion criteria"
                        action = fallback_action()
                        if action is None:
                            failure_reason = fallback_reason
                            break
                        tool_name, arguments, thought = action
                    else:
                        tool_name = str(decision.get("tool", ""))
                        arguments = decision.get("arguments", {})
                        thought = str(decision.get("thought", ""))
                        if model_call_error is not None:
                            fallback_reason = (
                                f"model call failed: {type(model_call_error).__name__}: "
                                f"{model_call_error}"
                            )
                        elif protocol_error is not None:
                            protocol_retries += 1
                            fallback_reason = (
                                f"model returned invalid JSON: {type(protocol_error).__name__}: "
                                f"{protocol_error}"
                            )
                        elif tool_name not in allowed_tools or not isinstance(arguments, dict):
                            protocol_retries += 1
                            fallback_reason = "model returned an invalid tool decision"
                        else:
                            protocol_retries = 0
                        if fallback_reason:
                            if (
                                model_call_error is None
                                and protocol_retries > self.max_protocol_retries
                            ):
                                failure_reason = (
                                    f"{fallback_reason}; protocol retry limit exceeded"
                                )
                                break
                            action = fallback_action()
                            if action is None:
                                if model_call_error is None:
                                    request_protocol_retry(fallback_reason, model_text)
                                    continue
                                else:
                                    failure_reason = (
                                        fallback_reason
                                        + "; no model-authored edit exists, so deterministic repair is forbidden"
                                    )
                                    break
                            tool_name, arguments, recovery = action
                            thought = f"{recovery} ({fallback_reason})"

                    arguments = self._normalize_arguments(tool_name, arguments)
                    signature = json.dumps([tool_name, arguments], ensure_ascii=False, sort_keys=True)
                    if signature == last_signature:
                        repeat_streak += 1
                    else:
                        last_signature = signature
                        repeat_streak = 1
                    if repeat_streak > 2:
                        protocol_retries += 1
                        repeated_reason = "model repeatedly selected the same action"
                        if protocol_retries > self.max_protocol_retries:
                            failure_reason = f"{repeated_reason}; protocol retry limit exceeded"
                            break
                        action = fallback_action()
                        if action is None:
                            request_protocol_retry(repeated_reason, model_text)
                            continue
                        tool_name, arguments, recovery = action
                        replacement_signature = json.dumps(
                            [tool_name, arguments], ensure_ascii=False, sort_keys=True
                        )
                        if replacement_signature == signature:
                            request_protocol_retry(repeated_reason, model_text)
                            continue
                        last_signature = replacement_signature
                        repeat_streak = 1
                        thought = f"{thought} {recovery}".strip()

                    protection_error = self._protected_write(tool_name, arguments)
                    if protection_error:
                        observation = f"ToolError: {protection_error}"
                    else:
                        try:
                            observation = await tool_client.call_tool(
                                tool_name, arguments, repo_path=str(root)
                            )
                        except Exception as exc:
                            observation = f"ToolError: {type(exc).__name__}: {exc}"

                    current_step = len(steps) + 1
                    steps.append(
                        {
                            "step": current_step,
                            "thought": thought or "Execute the next evidence-based action.",
                            "tool_call": {
                                "name": tool_name,
                                "arguments": arguments,
                                "transport": getattr(tool_client, "transport_name", "mcp-stdio"),
                            },
                            "observation": observation,
                        }
                    )

                    changed = (
                        tool_name == "write_file" and observation.startswith("wrote ")
                    ) or (
                        tool_name == "git_apply" and observation.strip() == "patch applied"
                    )
                    if changed:
                        last_mutation_step = current_step
                        last_test_step = 0
                        last_diff_step = 0
                        tests_passed = False
                        patch = ""
                    elif tool_name == "run_tests":
                        tests_passed = (
                            last_mutation_step > 0
                            and current_step > last_mutation_step
                            and self._tests_succeeded(observation)
                        )
                        last_test_step = current_step if tests_passed else 0
                    elif tool_name == "git_diff":
                        valid_diff = (
                            last_mutation_step > 0
                            and observation not in {"(no diff)", ""}
                            and observation != baseline_patch
                            and not observation.startswith("ToolError")
                        )
                        patch = observation if valid_diff else ""
                        last_diff_step = current_step if valid_diff else 0

                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": model_text or json.dumps(decision),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Observation:\n{observation}\nChoose the next tool or mark done "
                                    "only if a change was made, later tests pass, and the diff is recorded."
                                    + (
                                        f" Protocol correction: {fallback_reason}. Return strict JSON."
                                        if fallback_reason and model_call_error is None
                                        else ""
                                    )
                                ),
                            },
                        ]
                    )
                    if completion_ready():
                        break
                else:
                    failure_reason = "maximum steps reached before verified completion"

                success = completion_ready()
                if not success:
                    failure_reason = failure_reason or "verified completion criteria were not met"

                return self._trace(
                    steps=steps,
                    patch=patch,
                    tests_passed=tests_passed,
                    model=model_name,
                    skills=[item["name"] for item in selected],
                    subagents=subagent_summaries,
                    failure_reason=failure_reason,
                    success=success,
                    baseline_patch=baseline_patch,
                    tool_transport=transport_name,
                )
        except Exception as exc:
            steps.append(
                {
                    "step": len(steps) + 1,
                    "thought": "The MCP transport must be available before any main-agent tool runs.",
                    "tool_call": {
                        "name": "connect_mcp_stdio",
                        "arguments": {},
                        "transport": transport_name,
                    },
                    "observation": f"ToolError: {type(exc).__name__}: {exc}",
                }
            )
            return self._trace(
                steps=steps,
                patch="",
                tests_passed=False,
                model=model_name,
                skills=[item["name"] for item in selected],
                subagents=subagent_summaries,
                failure_reason=f"unable to establish or use MCP stdio transport: {type(exc).__name__}",
                success=False,
                baseline_patch=baseline_patch,
                tool_transport=transport_name,
            )

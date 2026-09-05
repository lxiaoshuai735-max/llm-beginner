"""A small stdio MCP server and the same tools as directly callable functions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mini-coding-agent")

_TOOLS = [
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file inside the target repository.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write complete UTF-8 text to a file inside the target repository.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_tests",
        "description": "Run python -m pytest -q inside the target repository.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "git_diff",
        "description": "Return the repository's current git diff.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "git_apply",
        "description": "Safely apply a unified diff inside the target repository.",
        "input_schema": {
            "type": "object",
            "properties": {"patch": {"type": "string"}},
            "required": ["patch"],
        },
    },
    {
        "name": "search_files",
        "description": "Search text in repository files and return matching locations.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "glob": {"type": "string"}},
            "required": ["query"],
        },
    },
]


def list_tools() -> list[dict[str, Any]]:
    """Return serializable tool metadata for the evaluator and model prompt."""
    return [dict(item) for item in _TOOLS]


def _repo(repo_path: str) -> Path:
    root = Path(repo_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"repository does not exist: {root}")
    return root


def _safe_path(repo_path: str, relative_path: str) -> Path:
    root = _repo(repo_path)
    supplied = Path(relative_path)
    if supplied.is_absolute():
        raise ValueError("absolute paths are forbidden; use a repository-relative path")
    target = (root / supplied).resolve()
    if target != root and root not in target.parents:
        raise ValueError("path escapes the target repository")
    if ".git" in target.relative_to(root).parts:
        raise ValueError("direct access to .git is forbidden")
    return target


@mcp.tool()
def read_file(repo_path: str, path: str) -> str:
    target = _safe_path(repo_path, path)
    if not target.is_file():
        raise FileNotFoundError(path)
    if target.stat().st_size > 1024 * 1024:
        raise ValueError("file is larger than 1 MiB")
    return target.read_text(encoding="utf-8", errors="replace")


@mcp.tool()
def write_file(repo_path: str, path: str, content: str) -> str:
    target = _safe_path(repo_path, path)
    if len(content.encode("utf-8")) > 1024 * 1024:
        raise ValueError("content is larger than 1 MiB")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8", newline="\n")
    return f"wrote {target.relative_to(_repo(repo_path)).as_posix()} ({len(content)} chars)"


def _run(args: list[str], repo_path: str, timeout: int, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=_repo(repo_path),
        input=stdin,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


@mcp.tool()
def run_tests(repo_path: str) -> str:
    result = _run([sys.executable, "-m", "pytest", "-q"], repo_path, timeout=60)
    output = (result.stdout + result.stderr)[-5000:]
    return f"exit_code={result.returncode}\n{output}"


@mcp.tool()
def git_diff(repo_path: str) -> str:
    result = _run(["git", "diff", "--no-ext-diff", "--"], repo_path, timeout=20)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    return result.stdout or "(no diff)"


@mcp.tool()
def git_apply(repo_path: str, patch: str) -> str:
    if len(patch) > 200_000:
        raise ValueError("patch is too large")
    result = _run(
        ["git", "apply", "--whitespace=nowarn", "--recount", "-"],
        repo_path,
        timeout=20,
        stdin=patch,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git apply failed")
    return "patch applied"


@mcp.tool()
def search_files(repo_path: str, query: str, glob: str = "*.py") -> str:
    if not query or len(query) > 300:
        raise ValueError("query must contain 1-300 characters")
    root = _repo(repo_path)
    matches: list[str] = []
    for path in sorted(root.rglob(glob)):
        if not path.is_file() or ".git" in path.parts or path.stat().st_size > 1024 * 1024:
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if query.lower() in line.lower():
                matches.append(f"{path.relative_to(root).as_posix()}:{line_number}: {line.strip()}")
            if len(matches) >= 100:
                return "\n".join(matches)
    return "\n".join(matches) if matches else "(no matches)"


TOOL_FUNCTIONS = {
    "read_file": read_file,
    "write_file": write_file,
    "run_tests": run_tests,
    "git_diff": git_diff,
    "git_apply": git_apply,
    "search_files": search_files,
}


if __name__ == "__main__":
    mcp.run(transport="stdio")

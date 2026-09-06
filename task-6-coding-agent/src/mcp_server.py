"""stdio MCP server for repository-scoped coding tools."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from . import tool_helpers

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


@mcp.tool()
def read_file(repo_path: str, path: str) -> str:
    return tool_helpers.read_file(repo_path, path)


@mcp.tool()
def write_file(repo_path: str, path: str, content: str) -> str:
    return tool_helpers.write_file(repo_path, path, content)


@mcp.tool()
def run_tests(repo_path: str) -> str:
    return tool_helpers.run_tests(repo_path)


@mcp.tool()
def git_diff(repo_path: str) -> str:
    return tool_helpers.git_diff(repo_path)


@mcp.tool()
def git_apply(repo_path: str, patch: str) -> str:
    return tool_helpers.git_apply(repo_path, patch)


@mcp.tool()
def search_files(repo_path: str, query: str, glob: str = "*.py") -> str:
    return tool_helpers.search_files(repo_path, query, glob)


if __name__ == "__main__":
    mcp.run(transport="stdio")

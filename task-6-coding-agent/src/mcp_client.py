"""Async stdio MCP client used by the main coding-agent loop."""

from __future__ import annotations

import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any


class MCPStdioClient:
    """Own one initialized MCP stdio session for a complete agent run.

    Imports are intentionally delayed until connection time.  This keeps trace
    parsing and unit tests usable on machines that have not installed the MCP
    runtime, while a real agent run still fails explicitly if MCP is absent.
    """

    transport_name = "mcp-stdio"

    def __init__(self) -> None:
        self._stack: AsyncExitStack | None = None
        self._session: Any = None

    async def __aenter__(self) -> "MCPStdioClient":
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise RuntimeError(
                "MCP client runtime is unavailable; install requirements.txt"
            ) from exc

        project_root = Path(__file__).resolve().parents[1]
        child_env = os.environ.copy()
        old_pythonpath = child_env.get("PYTHONPATH", "")
        child_env["PYTHONPATH"] = str(project_root) + (
            os.pathsep + old_pythonpath if old_pythonpath else ""
        )
        child_env["PYTHONUNBUFFERED"] = "1"
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "src.mcp_server"],
            env=child_env,
        )

        self._stack = AsyncExitStack()
        try:
            streams = await self._stack.enter_async_context(stdio_client(params))
            read_stream, write_stream = streams
            self._session = await self._stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()
        except BaseException:
            await self._stack.aclose()
            self._stack = None
            self._session = None
            raise
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    def _require_session(self) -> Any:
        if self._session is None:
            raise RuntimeError("MCP stdio session is not connected")
        return self._session

    async def list_tools(self) -> list[dict[str, Any]]:
        response = await self._require_session().list_tools()
        tools: list[dict[str, Any]] = []
        for tool in response.tools:
            schema = dict(tool.inputSchema or {})
            properties = dict(schema.get("properties", {}))
            properties.pop("repo_path", None)
            schema["properties"] = properties
            required = [name for name in schema.get("required", []) if name != "repo_path"]
            if required:
                schema["required"] = required
            else:
                schema.pop("required", None)
            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": schema,
                }
            )
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        repo_path: str,
    ) -> str:
        # Put the host-owned value last so even direct callers cannot override
        # the repository boundary through a crafted arguments dictionary.
        payload = {**arguments, "repo_path": repo_path}
        result = await self._require_session().call_tool(name, payload)
        parts: list[str] = []
        for item in result.content:
            if getattr(item, "type", None) == "text":
                parts.append(str(item.text))
            elif hasattr(item, "model_dump"):
                parts.append(json.dumps(item.model_dump(), ensure_ascii=False))
            else:
                parts.append(str(item))
        observation = "\n".join(parts).strip()
        if result.isError:
            return f"ToolError: {observation or 'MCP tool call failed'}"
        return observation

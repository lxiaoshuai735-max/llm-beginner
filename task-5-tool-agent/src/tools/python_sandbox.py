"""Run small teaching snippets in a constrained child process."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
from typing import Any

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "python_sandbox",
        "description": "Execute a short pure-Python snippet for algorithms; imports, files and network are blocked.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code that prints its answer"}},
            "required": ["code"],
        },
    },
}

_FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.ClassDef,
    ast.Global,
    ast.Nonlocal,
)
_FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "input", "help", "breakpoint",
    "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr",
    "__import__", "memoryview",
}


def _validate(code: str) -> None:
    if not code or len(code) > 4000:
        raise ValueError("code must contain 1-4000 characters")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"invalid Python syntax: {exc.msg}") from exc
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise ValueError(f"{type(node).__name__} is not allowed")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise ValueError(f"name {node.id!r} is not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("dunder attribute access is not allowed")


def _resource_limits() -> None:
    if os.name != "posix":
        return
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))


def run(args: dict[str, Any]) -> str:
    code = str(args.get("code", ""))
    _validate(code)
    with tempfile.TemporaryDirectory(prefix="react-python-") as temp_dir:
        process = subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=temp_dir,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=4,
            preexec_fn=_resource_limits if os.name == "posix" else None,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"},
            check=False,
        )
    stdout, stderr = process.stdout.strip(), process.stderr.strip()
    if process.returncode != 0:
        raise RuntimeError(f"sandbox exited {process.returncode}: {stderr[-1200:]}")
    if not stdout:
        return "Execution succeeded (no stdout)."
    return f"stdout:\n{stdout[:4000]}"

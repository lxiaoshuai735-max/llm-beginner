"""Repository-scoped implementations used by the MCP server.

This module deliberately contains no MCP client or server objects.  Keeping the
filesystem and subprocess policy here lets read-only subagents reuse the same
checks without giving the main agent an in-process shortcut around MCP.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def repository(repo_path: str) -> Path:
    root = Path(repo_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"repository does not exist: {root}")
    return root


def safe_path(repo_path: str, relative_path: str) -> Path:
    root = repository(repo_path)
    supplied = Path(relative_path)
    if supplied.is_absolute():
        raise ValueError("absolute paths are forbidden; use a repository-relative path")
    target = (root / supplied).resolve()
    if target != root and root not in target.parents:
        raise ValueError("path escapes the target repository")
    if ".git" in target.relative_to(root).parts:
        raise ValueError("direct access to .git is forbidden")
    return target


def read_file(repo_path: str, path: str) -> str:
    target = safe_path(repo_path, path)
    if not target.is_file():
        raise FileNotFoundError(path)
    if target.stat().st_size > 1024 * 1024:
        raise ValueError("file is larger than 1 MiB")
    return target.read_text(encoding="utf-8", errors="replace")


def write_file(repo_path: str, path: str, content: str) -> str:
    target = safe_path(repo_path, path)
    if len(content.encode("utf-8")) > 1024 * 1024:
        raise ValueError("content is larger than 1 MiB")
    if target.is_file() and target.read_text(encoding="utf-8", errors="replace") == content:
        return f"unchanged {target.relative_to(repository(repo_path)).as_posix()}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8", newline="\n")
    return f"wrote {target.relative_to(repository(repo_path)).as_posix()} ({len(content)} chars)"


def _run(
    args: list[str],
    repo_path: str,
    timeout: int,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    options: dict[str, object] = {}
    if stdin is None:
        # The MCP server itself communicates over stdin.  Repository commands
        # must never inherit that protocol stream or they may consume/block it.
        options["stdin"] = subprocess.DEVNULL
    else:
        options["input"] = stdin
    # Each Python test run gets a fresh bytecode-cache prefix.  This avoids a
    # false failure when a same-size source edit happens within one filesystem
    # timestamp tick and CPython would otherwise reuse an older local .pyc.
    with tempfile.TemporaryDirectory(prefix="coding-agent-pycache-") as pycache:
        environment = os.environ.copy()
        environment["PYTHONPYCACHEPREFIX"] = pycache
        return subprocess.run(
            args,
            cwd=repository(repo_path),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            env=environment,
            **options,
        )


def run_tests(repo_path: str) -> str:
    result = _run([sys.executable, "-m", "pytest", "-q"], repo_path, timeout=60)
    output = (result.stdout + result.stderr)[-5000:]
    return f"exit_code={result.returncode}\n{output}"


def git_diff(repo_path: str) -> str:
    result = _run(["git", "diff", "--no-ext-diff", "--"], repo_path, timeout=20)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    return result.stdout or "(no diff)"


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


def search_files(repo_path: str, query: str, glob: str = "*.py") -> str:
    if not query or len(query) > 300:
        raise ValueError("query must contain 1-300 characters")
    root = repository(repo_path)
    matches: list[str] = []
    for path in sorted(root.rglob(glob)):
        if not path.is_file() or ".git" in path.parts or path.stat().st_size > 1024 * 1024:
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if query.lower() in line.lower():
                matches.append(
                    f"{path.relative_to(root).as_posix()}:{line_number}: {line.strip()}"
                )
            if len(matches) >= 100:
                return "\n".join(matches)
    return "\n".join(matches) if matches else "(no matches)"

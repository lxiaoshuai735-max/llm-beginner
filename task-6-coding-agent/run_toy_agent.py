"""Reset the toy repository, run the agent, and save a complete trace."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.agent import CodingAgent

ROOT = Path(__file__).resolve().parent


def main() -> None:
    repo = ROOT / "data" / "toy-repo"
    shutil.copyfile(repo / "calculator.py.orig", repo / "calculator.py")
    issue = (repo / "ISSUE.md").read_text(encoding="utf-8")
    trace = CodingAgent().run(str(repo), issue)
    output = ROOT / "eval" / "toy_repo_trace.json"
    output.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"tests_passed={trace['tests_passed']}")
    print(f"steps={len(trace['steps'])}")
    print(f"patch_chars={len(trace['patch'])}")
    print(f"trace={output}")


if __name__ == "__main__":
    main()

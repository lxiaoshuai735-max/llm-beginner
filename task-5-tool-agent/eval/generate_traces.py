"""Generate several complete traces for the Task 5 submission."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import ReActAgent  # noqa: E402


def main() -> None:
    tasks = json.loads((ROOT / "data" / "tasks.json").read_text(encoding="utf-8"))
    selected_ids = {1, 5, 8, 10}
    agent = ReActAgent()
    traces = []
    for task in tasks:
        if task["id"] not in selected_ids:
            continue
        result = agent.run(task["task"])
        traces.append({"id": task["id"], "task": task["task"], **result})
        print(f"task {task['id']}: {'OK' if result['success'] else 'FAIL'}")
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": agent.model,
        "api_url": agent.api_url,
        "traces": traces,
    }
    destination = ROOT / "eval" / "traces.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()

"""Generate several complete traces for the Task 5 submission."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import ReActAgent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate auditable Task 5 ReAct traces.")
    parser.add_argument("--tasks", type=Path, default=ROOT / "data" / "tasks.json")
    parser.add_argument(
        "--ids",
        default="",
        help="Comma-separated task IDs. By default every task in --tasks is run.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "traces.json")
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-steps", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks_path = args.tasks.resolve()
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    selected_ids = {int(value.strip()) for value in args.ids.split(",") if value.strip()}
    agent = ReActAgent(api_url=args.api_url, model=args.model, max_steps=args.max_steps)
    traces = []
    for task in tasks:
        if selected_ids and task["id"] not in selected_ids:
            continue
        result = agent.run(task["task"])
        traces.append({"id": task["id"], "task": task["task"], **result})
        print(f"task {task['id']}: {'OK' if result['success'] else 'FAIL'}")
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": agent.model,
        "api_url": agent.api_url,
        "tasks_file": str(tasks_path),
        "traces": traces,
    }
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()

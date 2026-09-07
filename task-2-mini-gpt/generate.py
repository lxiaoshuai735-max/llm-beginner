"""Generate fixed-prompt poetry samples for all four decoding strategies.

The script deliberately starts from an unconstrained decoding baseline:
``repetition_penalty=1.0`` and ``no_repeat_ngram_size=0``. Token-level
repetition controls are unsafe for a byte-level tokenizer because a Chinese
character can span several tokens. UTF-8 validity is instead enforced while
sampling, and raw bytes are retained for inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from src.model import load_for_eval

ROOT = Path(__file__).resolve().parent

STRATEGIES = {
    "greedy": {"temperature": 0.0},
    # Conservative settings make the comparison meaningful for a compact
    # poetry model: all four algorithms remain distinct, without deliberately
    # pushing a low-probability byte sequence into unreadable text.
    "temperature": {"temperature": 0.25},
    "top_k": {"temperature": 0.25, "top_k": 4},
    "top_p": {"temperature": 0.25, "top_p": 0.55},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="ckpt/best.pt")
    parser.add_argument("--prompts", default="data/generation_prompts.json")
    parser.add_argument("--output", default="eval/generation_quality.json")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompts(path: Path, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts", payload) if isinstance(payload, dict) else payload
    if not isinstance(prompts, list):
        raise ValueError("prompt file must be a JSON list or contain a prompts list")
    selected = [item for item in prompts if isinstance(item, dict) and item.get("prompt")]
    if len(selected) < limit:
        raise ValueError(f"need at least {limit} fixed held-out prompts, found {len(selected)}")
    return selected[:limit]


def text_metrics(text: str) -> dict[str, int | bool]:
    longest_run, current_run = 0, 0
    previous = ""
    for char in text:
        current_run = current_run + 1 if char == previous else 1
        longest_run = max(longest_run, current_run)
        previous = char
    return {
        "character_count": len(text),
        "max_identical_character_run": longest_run,
        "contains_replacement_character": "\ufffd" in text,
    }


def main() -> None:
    args = parse_args()
    checkpoint = (ROOT / args.checkpoint).resolve()
    prompt_path = (ROOT / args.prompts).resolve()
    output_path = (ROOT / args.output).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    prompts = load_prompts(prompt_path, args.limit)

    model, tokenizer = load_for_eval(str(checkpoint))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    samples: list[dict[str, Any]] = []

    for prompt_index, prompt_record in enumerate(prompts):
        prompt = str(prompt_record["prompt"])
        prompt_ids = tokenizer.encode(prompt)
        for strategy_index, (strategy, options) in enumerate(STRATEGIES.items()):
            sample_seed = args.seed + prompt_index * 100 + strategy_index
            torch.manual_seed(sample_seed)
            full_ids = model.generate(
                prompt_ids,
                args.max_new_tokens,
                tokenizer=tokenizer,
                enforce_utf8=True,
                complete_utf8_at_end=True,
                stop_after_newline=True,
                min_new_tokens=4,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
                **options,
            )[0].cpu().tolist()
            completed_text, trimmed_tokens = tokenizer.decode_complete(full_ids)
            raw_bytes = tokenizer.ids_to_bytes(full_ids)
            sample = {
                "prompt_id": prompt_record.get("id", prompt_index + 1),
                "prompt": prompt,
                "source_title": prompt_record.get("title", ""),
                "source_author": prompt_record.get("author", ""),
                "strategy": strategy,
                "seed": sample_seed,
                "parameters": {
                    **options,
                    "repetition_penalty": 1.0,
                    "no_repeat_ngram_size": 0,
                    "enforce_utf8": True,
                    "complete_utf8_at_end": True,
                    "stop_after_newline": True,
                    "min_new_tokens": 4,
                },
                "raw_token_ids": full_ids,
                "raw_utf8_bytes_hex": raw_bytes.hex(),
                "text": completed_text,
                "complete_utf8_boundary": tokenizer.utf8_state_for_ids(full_ids) == 0,
                "trimmed_trailing_token_count": trimmed_tokens,
                "automated_checks": text_metrics(completed_text),
            }
            samples.append(sample)
            print(f"[{strategy}] {completed_text}")

    tokenizer_path = checkpoint.parent / "tokenizer.json"
    payload = {
        "format": "task2_generation_quality_v2",
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "tokenizer": str(tokenizer_path.relative_to(ROOT)),
        "tokenizer_sha256": sha256(tokenizer_path),
        "prompts": str(prompt_path.relative_to(ROOT)),
        "prompt_count": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "base_seed": args.seed,
        "notes": (
            "Raw token ids and bytes are retained. Text is decoded only through "
            "a complete UTF-8 boundary; no replacement-character deletion occurs."
        ),
        "samples": samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Keep the legacy file as a compact, human-readable first-prompt snapshot.
    first_id = prompts[0].get("id", 1)
    first_prompt = [sample for sample in samples if sample["prompt_id"] == first_id]
    (ROOT / "generation_samples.json").write_text(
        json.dumps({sample["strategy"]: sample["text"] for sample in first_prompt}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

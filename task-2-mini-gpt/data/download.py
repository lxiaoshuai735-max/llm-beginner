"""Prepare reproducible corpora for Task 2.

``poetry`` keeps the small in-repository corpus for a quick smoke run.
``tang-poetry`` downloads public Complete Tang Poems JSON, deduplicates poems,
and splits by complete poem before writing train/dev/test files.  The expanded
corpus is deliberately not committed; this script records its source and split
metadata in ``data/dataset_info.json`` instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent
ROOT = DATA_DIR.parent.parent
DEFAULT_TANG_REPO = "https://github.com/chinese-poetry/chinese-poetry.git"
DEFAULT_TANG_SSH_REPO = "git@github.com:chinese-poetry/chinese-poetry.git"


def write_dataset_info(dataset: str, ppl_threshold: float, **extra: Any) -> None:
    payload = {
        "dataset": dataset,
        "train": "train.txt",
        "dev": "dev.txt",
        "test": "test.txt",
        "ppl_threshold": ppl_threshold,
        **extra,
    }
    (DATA_DIR / "dataset_info.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_text_splits(text: str, dataset: str, ppl_threshold: float, dev_ratio: float = 0.1) -> None:
    """Fallback splitter for non-poetry corpora that have no document metadata."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) < 100:
        sys.exit(f"[错误] {dataset} 文本过短，无法切分 train/dev")
    split_at = max(1, int(len(text) * (1 - dev_ratio)))
    (DATA_DIR / "train.txt").write_text(text[:split_at], encoding="utf-8")
    (DATA_DIR / "dev.txt").write_text(text[split_at:], encoding="utf-8")
    (DATA_DIR / "test.txt").write_text("", encoding="utf-8")
    write_dataset_info(dataset, ppl_threshold, split_unit="character_fallback")
    print(f"已生成 train.txt / dev.txt（{dataset}，dev_ratio={dev_ratio}）")


def clean_line(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"\s+", "", text)
    return text.translate(str.maketrans({",": "，", ".": "。", "!": "！", "?": "？", ";": "；"}))


def normalise_poem(paragraphs: object) -> str:
    if isinstance(paragraphs, str):
        paragraphs = paragraphs.splitlines()
    if not isinstance(paragraphs, list):
        return ""
    lines = [clean_line(line) for line in paragraphs]
    lines = [line for line in lines if line]
    text = "\n".join(lines).strip()
    chinese_chars = sum("\u3400" <= char <= "\u9fff" for char in text)
    return text if chinese_chars >= 8 else ""


_CLASSICAL_LINE = re.compile(r"^[\u3400-\u9fff，。！？；、】【、]+$")


def is_clean_classical_poem(text: str) -> bool:
    """Reject editorial notes, indices, and prose annotations in source JSON.

    The Complete Tang Poems repository contains valuable but non-poetic source
    notes in some ``paragraphs`` fields (for example, book references in
    parentheses).  They are legitimate source metadata, but are harmful target
    tokens for this compact poetry language model.  Keep only multi-line poems
    composed of CJK ideographs and standard Chinese poetry punctuation.
    """
    lines = [line for line in text.splitlines() if line]
    if len(lines) < 2 or len(lines) > 80:
        return False
    for line in lines:
        chinese_count = sum("\u3400" <= char <= "\u9fff" for char in line)
        if chinese_count < 4 or chinese_count > 60 or not _CLASSICAL_LINE.fullmatch(line):
            return False
    return True


def prompt_from_poem(text: str) -> str:
    """Use a held-out poem's opening line as a fixed continuation prompt."""
    lines = [line for line in text.splitlines() if line]
    if not lines:
        return ""
    # Use the entire first line rather than stopping at its first comma.  The
    # newline is part of the prompt boundary, so generation begins with a new
    # poetic line instead of trying to extend an already complete sentence.
    return lines[0][:24] + "\n"


_PROMPT_AUTHORS = {
    "李白", "杜甫", "王維", "白居易", "孟浩然", "王昌齡", "劉禹錫",
    "李商隱", "杜牧", "岑參", "韋應物", "王之渙", "高適", "韓愈",
    "柳宗元", "張籍", "元稹", "賈島", "王建",
}


def choose_generation_prompts(test_records: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    """Select ten fixed, short, held-out classical-poetry openings.

    The selection happens *after* the complete-poem split.  It avoids source
    entries with unknown authors or extremely long title lines, so the M5
    readability evaluation measures poetic continuation rather than recovery of
    an editorial fragment.  A SHA-256 ordering makes the choice stable without
    relying on file iteration order.
    """
    candidates = []
    for record in test_records:
        lines = record["text"].splitlines()
        prompt = prompt_from_poem(record["text"])
        chinese_length = sum("\u3400" <= char <= "\u9fff" for char in prompt)
        if (
            record.get("author") in _PROMPT_AUTHORS
            and 2 <= len(lines) <= 8
            and len(record.get("title", "")) <= 30
            and 6 <= chinese_length <= 18
        ):
            candidates.append((hashlib.sha256(f"{seed}:{record['text']}".encode()).hexdigest(), record, prompt))
    candidates.sort(key=lambda item: item[0])
    chosen = candidates[:10]
    if len(chosen) < 10:
        raise ValueError("could not select ten clean held-out poetry prompts")
    return [
        {
            "id": f"heldout-{index:02d}",
            "prompt": prompt,
            "title": record.get("title", ""),
            "author": record.get("author", ""),
        }
        for index, (_, record, prompt) in enumerate(chosen, start=1)
    ]


def write_poem_split(records: list[dict[str, str]], path: Path) -> None:
    boundary = "\n\n"
    path.write_text(boundary.join(record["text"] for record in records) + "\n", encoding="utf-8")


def prepare_poem_splits(
    records: list[dict[str, str]],
    *,
    dataset: str,
    seed: int,
    dev_ratio: float,
    test_ratio: float,
    ppl_threshold: float,
    source: str,
) -> None:
    unique: dict[str, dict[str, str]] = {}
    unique_openings: set[str] = set()
    for record in records:
        text = record["text"]
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        opening = prompt_from_poem(text)
        # Source shards sometimes keep textual variants with different complete
        # bodies but the same opening couplet.  Do not let one variant land in
        # train and another in test: that would make a held-out continuation
        # artificially easy despite a full-text hash deduplication.
        if key not in unique and opening not in unique_openings:
            unique[key] = record
            unique_openings.add(opening)
    poems = list(unique.values())
    if len(poems) < 30:
        raise ValueError("poem corpus is too small for disjoint train/dev/test splits")
    random.Random(seed).shuffle(poems)
    test_count = max(10, round(len(poems) * test_ratio))
    dev_count = max(10, round(len(poems) * dev_ratio))
    train_count = len(poems) - dev_count - test_count
    if train_count < 10:
        raise ValueError("split ratios leave too few poems for training")
    test_records = poems[:test_count]
    dev_records = poems[test_count : test_count + dev_count]
    train_records = poems[test_count + dev_count :]
    write_poem_split(train_records, DATA_DIR / "train.txt")
    write_poem_split(dev_records, DATA_DIR / "dev.txt")
    write_poem_split(test_records, DATA_DIR / "test.txt")
    prompts = choose_generation_prompts(test_records, seed)
    if any(not item["prompt"] for item in prompts) or len(prompts) < 10:
        raise ValueError("could not create ten held-out poetry prompts")
    (DATA_DIR / "generation_prompts.json").write_text(
        json.dumps({"split": "test", "seed": seed, "prompts": prompts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_dataset_info(
        dataset,
        ppl_threshold,
        source=source,
        split_unit="complete_poem",
        deduplication=["sha256_complete_poem", "complete_opening_line"],
        cleaning="CJK ideographs plus standard Chinese poetry punctuation only",
        split_seed=seed,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        poem_counts={"train": len(train_records), "dev": len(dev_records), "test": len(test_records)},
        char_counts={
            "train": sum(len(record["text"]) for record in train_records),
            "dev": sum(len(record["text"]) for record in dev_records),
            "test": sum(len(record["text"]) for record in test_records),
        },
        prompt_file="generation_prompts.json",
    )
    print(
        "poem-level split: "
        f"train={len(train_records)}, dev={len(dev_records)}, test={len(test_records)}"
    )


def get_poetry(args: argparse.Namespace) -> None:
    src = ROOT / "poetryFromTang.txt"
    if not src.exists():
        sys.exit(f"[错误] 找不到 {src}（应在仓库根）")
    text = src.read_text(encoding="utf-8")
    records = [{"text": normalise_poem(block)} for block in re.split(r"\n\s*\n", text)]
    records = [record for record in records if record["text"]]
    shutil.copy(src, DATA_DIR / "poetry.txt")
    prepare_poem_splits(
        records,
        dataset="poetry",
        seed=args.seed,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        ppl_threshold=50,
        source="repository:poetryFromTang.txt",
    )


def clone_tang_source(repo_url: str, destination: Path) -> Path:
    """Sparse-clone only Complete Tang Poems, with SSH fallback for slow HTTPS."""
    if not (destination / ".git").exists():
        attempts = [repo_url]
        if repo_url == DEFAULT_TANG_REPO:
            attempts.append(DEFAULT_TANG_SSH_REPO)
        errors: list[str] = []
        for url in attempts:
            shutil.rmtree(destination, ignore_errors=True)
            try:
                subprocess.run(
                    ["git", "clone", "--depth=1", "--filter=blob:none", "--sparse", url, str(destination)],
                    check=True,
                    text=True,
                    timeout=600,
                )
                break
            except (OSError, subprocess.SubprocessError) as exc:
                errors.append(f"{url}: {exc}")
        else:
            raise RuntimeError("unable to clone Complete Tang Poems source: " + " | ".join(errors))
    subprocess.run(
        ["git", "-C", str(destination), "sparse-checkout", "set", "全唐诗"],
        check=True,
        text=True,
        timeout=600,
    )
    return destination / "全唐诗"


def get_tang_poetry(args: argparse.Namespace) -> None:
    if args.source_dir:
        source_dir = Path(args.source_dir).expanduser().resolve()
    else:
        repo_url = os.getenv("TASK2_TANG_REPO", DEFAULT_TANG_REPO)
        source_dir = clone_tang_source(repo_url, DATA_DIR / "cache" / "chinese-poetry")
    if (source_dir / "全唐诗").is_dir():
        source_dir = source_dir / "全唐诗"
    files = sorted(source_dir.glob("poet.tang.*.json"))
    if not files:
        raise FileNotFoundError(f"no poet.tang.*.json files found in {source_dir}")
    records: list[dict[str, str]] = []
    for path in files:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            text = normalise_poem(row.get("paragraphs"))
            if text and is_clean_classical_poem(text):
                records.append(
                    {
                        "text": text,
                        "title": clean_line(row.get("title")),
                        "author": clean_line(row.get("author")),
                    }
                )
    if args.max_poems > 0:
        # Shuffle before limiting, so the selected subset is reproducible but
        # not biased toward the first JSON shard or one author.
        random.Random(args.seed).shuffle(records)
        records = records[: args.max_poems]
    prepare_poem_splits(
        records,
        dataset="tang-poetry",
        seed=args.seed,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        ppl_threshold=50,
        source=f"{DEFAULT_TANG_REPO} @ poem JSON shards ({len(files)} files)",
    )


def write_hf_text_split(split: Any, out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in split:
            text = str(row.get("text", "")).strip()
            if text:
                handle.write(text)
                handle.write("\n\n")


def get_tinystories() -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("[错误] pip install datasets pyarrow")
    ds = load_dataset("roneneldan/TinyStories", cache_dir=str(DATA_DIR / "cache"))
    train_split = ds["train"]
    dev_split = ds["validation"] if "validation" in ds else ds["train"].select(range(1000))
    write_hf_text_split(train_split, DATA_DIR / "train.txt")
    write_hf_text_split(dev_split, DATA_DIR / "dev.txt")
    (DATA_DIR / "test.txt").write_text("", encoding="utf-8")
    write_dataset_info("tinystories", 10, split_unit="dataset_split")


def get_skypile() -> None:
    print("SkyPile-150B 体量大，请用 streaming 子集后显式写入 train/dev/test.txt。")
    print("建议：python data/download.py --dataset tang-poetry --max-poems 20000")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["poetry", "tang-poetry", "tinystories", "skypile"], default="poetry")
    parser.add_argument("--source-dir", help="existing 全唐诗 directory or its repository root")
    parser.add_argument("--max-poems", type=int, default=20000, help="0 means all poems")
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()
    if not 0 < args.dev_ratio < 0.5 or not 0 < args.test_ratio < 0.5 or args.dev_ratio + args.test_ratio >= 0.8:
        parser.error("dev/test ratios must be positive and leave a substantial training split")
    {"poetry": get_poetry, "tang-poetry": get_tang_poetry, "tinystories": lambda _: get_tinystories(), "skypile": lambda _: get_skypile()}[args.dataset](args)


if __name__ == "__main__":
    main()

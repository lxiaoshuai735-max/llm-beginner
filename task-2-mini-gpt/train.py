"""Train mini-GPT on a poem-level split and retain comparable run metadata."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from src.model import GPTConfig, MiniGPT, save_checkpoint
from src.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent


class TokenWindows(Dataset):
    def __init__(self, ids: list[int], block_size: int, stride: int) -> None:
        self.windows = [
            torch.tensor(ids[start : start + block_size + 1], dtype=torch.long)
            for start in range(0, len(ids) - block_size, stride)
        ]
        if not self.windows:
            raise ValueError("corpus is shorter than block_size")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.windows[index]
        return window[:-1], window[1:]


@torch.no_grad()
def evaluate(
    model: MiniGPT,
    loader: DataLoader,
    tokenizer: BPETokenizer,
    device: torch.device,
) -> dict[str, float | int]:
    """Report token PPL plus NLL normalized by emitted UTF-8 bytes.

    Token PPL is retained for the task's established threshold.  Byte-normalized
    loss is the valid comparison across different BPE vocabularies because a
    larger vocabulary changes the number of tokens used to encode the same text.
    """
    model.eval()
    total_nll, total_tokens, total_bytes = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        total_nll += F.cross_entropy(
            logits.flatten(0, 1), targets.flatten(), reduction="sum"
        ).item()
        flat_targets = targets.detach().cpu().flatten().tolist()
        total_tokens += len(flat_targets)
        total_bytes += sum(len(tokenizer.token_bytes(token_id)) for token_id in flat_targets)
    token_nll = total_nll / total_tokens
    byte_nll = total_nll / total_bytes
    return {
        "token_nll": token_nll,
        "token_perplexity": math.exp(token_nll),
        "utf8_byte_nll": byte_nll,
        "utf8_byte_perplexity": math.exp(byte_nll),
        "n_tokens": total_tokens,
        "n_utf8_bytes": total_bytes,
    }


def load_fixed_prompts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("prompts", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return []
    return [item for item in records if isinstance(item, dict) and item.get("prompt")][:10]


def tokenizer_training_text(text: str, limit: int) -> str:
    """Draw deterministic, evenly spaced training-only chunks for BPE fitting.

    The model still consumes every training poem.  This cap only avoids spending
    most of an experiment in the deliberately simple educational BPE trainer,
    while sampling across the entire training split instead of using one prefix.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    chunks = min(20, max(4, limit // 4096))
    width = max(1, limit // chunks)
    starts = [round(index * (len(text) - width) / max(1, chunks - 1)) for index in range(chunks)]
    return "\n".join(text[start:start + width] for start in starts)


@torch.no_grad()
def save_periodic_samples(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    prompts: list[dict[str, Any]],
    output_path: Path,
    epoch: int,
    max_new_tokens: int,
) -> None:
    if not prompts:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else []
    samples = []
    for record in prompts:
        prompt = str(record["prompt"])
        ids = model.generate(
            tokenizer.encode(prompt),
            max_new_tokens,
            temperature=0.0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            tokenizer=tokenizer,
            enforce_utf8=True,
            complete_utf8_at_end=True,
        )[0].cpu().tolist()
        text, trimmed = tokenizer.decode_complete(ids)
        samples.append(
            {
                "prompt_id": record.get("id"),
                "prompt": prompt,
                "text": text,
                "trimmed_trailing_token_count": trimmed,
            }
        )
    history.append({"epoch": epoch, "strategy": "greedy", "samples": samples})
    output_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--char-vocab-size", type=int, default=0)
    parser.add_argument("--tokenizer-training-chars", type=int, default=250_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--output-dir", default="ckpt")
    parser.add_argument("--sample-prompts", default="data/generation_prompts.json")
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--sample-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260906)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.block_size < 8:
        raise ValueError("block_size must be at least 8")
    if args.sample_every < 1:
        raise ValueError("sample_every must be at least 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_text = (ROOT / "data/train.txt").read_text(encoding="utf-8")
    dev_text = (ROOT / "data/dev.txt").read_text(encoding="utf-8")
    bpe_text = tokenizer_training_text(train_text, args.tokenizer_training_chars)
    tokenizer = BPETokenizer.train(
        bpe_text,
        vocab_size=args.vocab_size,
        char_vocab_size=args.char_vocab_size,
    )
    train_ids, dev_ids = tokenizer.encode(train_text), tokenizer.encode(dev_text)
    train_loader = DataLoader(
        TokenWindows(train_ids, args.block_size, max(1, args.block_size // 2)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    dev_loader = DataLoader(
        TokenWindows(dev_ids, args.block_size, args.block_size),
        batch_size=args.batch_size,
    )
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.block_size + 1,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    model = MiniGPT(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader), eta_min=args.learning_rate * 0.1
    )
    fixed_prompts = load_fixed_prompts((ROOT / args.sample_prompts).resolve())
    history: list[dict[str, Any]] = []
    best_ppl, stale = float("inf"), 0
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} vocab={tokenizer.vocab_size} train_tokens={len(train_ids)} "
        f"parameters={parameter_count:,} block_size={args.block_size} "
        f"bpe_chars={len(bpe_text)} char_vocab={args.char_vocab_size}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, tokens = 0.0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item() * targets.numel()
            tokens += targets.numel()
        metrics = evaluate(model, dev_loader, tokenizer, device)
        row = {"epoch": epoch, "train_nll": running / tokens, **metrics}
        history.append(row)
        print(
            f"epoch={epoch:02d} train_nll={row['train_nll']:.4f} "
            f"dev_ppl={row['token_perplexity']:.2f} "
            f"dev_byte_nll={row['utf8_byte_nll']:.4f}"
        )
        if row["token_perplexity"] < best_ppl:
            best_ppl, stale = float(row["token_perplexity"]), 0
            save_checkpoint(
                output_dir / "best.pt",
                model,
                tokenizer,
                epoch=epoch,
                dev_token_perplexity=row["token_perplexity"],
                dev_utf8_byte_nll=row["utf8_byte_nll"],
            )
        else:
            stale += 1
        if epoch % args.sample_every == 0:
            save_periodic_samples(
                model,
                tokenizer,
                fixed_prompts,
                output_dir / "periodic_samples.json",
                epoch,
                args.sample_new_tokens,
            )
        if stale >= args.patience:
            print(f"early_stop_epoch={epoch}")
            break
    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    dataset_info_path = ROOT / "data" / "dataset_info.json"
    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8")) if dataset_info_path.exists() else {}
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "training_arguments": vars(args),
                "dataset_info": dataset_info,
                "tokenizer_vocab_size": tokenizer.vocab_size,
                "parameter_count": parameter_count,
                "best_token_perplexity": best_ppl,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (ROOT / "figures").mkdir(exist_ok=True)
    plt.plot([row["epoch"] for row in history], [row["token_perplexity"] for row in history])
    plt.xlabel("Epoch")
    plt.ylabel("Dev token perplexity")
    plt.tight_layout()
    plt.savefig(ROOT / "figures/perplexity.png", dpi=180)
    plt.close()
    print(f"best_dev_token_ppl={best_ppl:.2f}")


if __name__ == "__main__":
    main()

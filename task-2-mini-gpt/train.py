"""Train mini-GPT on the prepared poetry corpus."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

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

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]
        return window[:-1], window[1:]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="sum")
        total_loss += loss.item()
        total_tokens += targets.numel()
    average = total_loss / total_tokens
    return average, math.exp(average)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=640)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_text = (ROOT / "data/train.txt").read_text(encoding="utf-8")
    dev_text = (ROOT / "data/dev.txt").read_text(encoding="utf-8")
    tokenizer = BPETokenizer.train(train_text, vocab_size=args.vocab_size)
    tokenizer.save_pretrained(ROOT / "ckpt/tokenizer.json")
    train_ids, dev_ids = tokenizer.encode(train_text), tokenizer.encode(dev_text)
    train_loader = DataLoader(
        TokenWindows(train_ids, args.block_size, args.block_size // 2),
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
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=5e-5)
    history, best_ppl, stale = [], float("inf"), 0
    print(f"device={device} vocab={tokenizer.vocab_size} train_tokens={len(train_ids)} parameters={sum(p.numel() for p in model.parameters()):,}")
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
        train_loss = running / tokens
        dev_loss, dev_ppl = evaluate(model, dev_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss, "dev_perplexity": dev_ppl})
        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} dev_loss={dev_loss:.4f} dev_ppl={dev_ppl:.2f}")
        if dev_ppl < best_ppl:
            best_ppl, stale = dev_ppl, 0
            save_checkpoint(ROOT / "ckpt/best.pt", model, tokenizer, epoch=epoch, dev_perplexity=dev_ppl)
        else:
            stale += 1
            if stale >= args.patience:
                break
    (ROOT / "ckpt/history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (ROOT / "figures").mkdir(exist_ok=True)
    plt.plot([row["epoch"] for row in history], [row["dev_perplexity"] for row in history])
    plt.xlabel("Epoch"); plt.ylabel("Dev perplexity"); plt.tight_layout()
    plt.savefig(ROOT / "figures/perplexity.png", dpi=180); plt.close()
    print(f"best_dev_ppl={best_ppl:.2f}")


if __name__ == "__main__":
    main()

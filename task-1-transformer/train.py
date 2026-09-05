"""Train the from-scratch Transformer classifier on ChnSentiCorp."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from src.model import (
    CharacterTokenizer,
    ModelConfig,
    TransformerClassifier,
    save_checkpoint,
)

ROOT = Path(__file__).resolve().parent


class SentimentDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: CharacterTokenizer, max_len: int):
        self.texts = frame["text"].astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return (
            self.tokenizer.encode(self.texts[index], max_length=self.max_len),
            self.labels[index],
        )


def make_collate_fn(pad_token_id: int):
    def collate(batch: list[tuple[list[int], int]]) -> tuple[Tensor, Tensor, Tensor]:
        max_len = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
        labels = torch.tensor([label for _, label in batch], dtype=torch.long)
        for row, (ids, _) in enumerate(batch):
            length = len(ids)
            input_ids[row, :length] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, :length] = True
        return input_ids, attention_mask, labels

    return collate


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return device


def make_scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, lr_multiplier)


def train_epoch(
    model: TransformerClassifier,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
    grad_clip: float,
) -> EpochMetrics:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_examples += batch_size
    return EpochMetrics(total_loss / total_examples, total_correct / total_examples)


@torch.no_grad()
def evaluate(
    model: TransformerClassifier,
    loader: DataLoader,
    device: torch.device,
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        logits = model(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_examples += batch_size
    return EpochMetrics(total_loss / total_examples, total_correct / total_examples)


def save_history(history: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)

    epochs = [item["epoch"] for item in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [item["train"]["loss"] for item in history], label="train")
    plt.plot(epochs, [item["validation"]["loss"] for item in history], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.legend()
    plt.tight_layout()
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / "loss_curve.png", dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=ROOT / "data/train.parquet")
    parser.add_argument("--validation-file", type=Path, default=ROOT / "data/validation.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "ckpt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-vocab-size", type=int, default=6000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    train_frame = pd.read_parquet(args.train_file)
    validation_frame = pd.read_parquet(args.validation_file)

    tokenizer = CharacterTokenizer.build(
        train_frame["text"].astype(str),
        max_vocab_size=args.max_vocab_size,
        min_frequency=args.min_frequency,
    )
    collate_fn = make_collate_fn(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        SentimentDataset(train_frame, tokenizer, args.max_len),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        SentimentDataset(validation_frame, tokenizer, args.max_len),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_len=args.max_len,
        dropout=args.dropout,
    )
    model = TransformerClassifier(config).to(device)
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = make_scheduler(
        optimizer,
        warmup_steps=int(args.warmup_ratio * total_steps),
        total_steps=total_steps,
    )

    print(f"device={device} vocab_size={tokenizer.vocab_size} parameters={sum(p.numel() for p in model.parameters()):,}")
    history: list[dict] = []
    best_accuracy = -1.0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, args.grad_clip
        )
        validation_metrics = evaluate(model, validation_loader, device)
        record = {
            "epoch": epoch,
            "train": asdict(train_metrics),
            "validation": asdict(validation_metrics),
        }
        history.append(record)
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} "
            f"val_loss={validation_metrics.loss:.4f} val_acc={validation_metrics.accuracy:.4f}"
        )

        if validation_metrics.accuracy > best_accuracy:
            best_accuracy = validation_metrics.accuracy
            stale_epochs = 0
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                tokenizer,
                epoch=epoch,
                validation_accuracy=best_accuracy,
                train_args=vars(args),
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping after {epoch} epochs")
                break

    save_history(history, args.output_dir)
    print(f"best validation accuracy={best_accuracy:.4f}")


if __name__ == "__main__":
    main()

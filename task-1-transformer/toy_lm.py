"""A tiny causal language-model exercise using the same handwritten attention."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from src.block import TransformerBlock, make_causal_mask
from src.model import CharacterTokenizer, SinusoidalPositionEncoding

ROOT = Path(__file__).resolve().parent


class TokenWindows(Dataset):
    def __init__(self, token_ids: list[int], sequence_length: int) -> None:
        self.sequence_length = sequence_length
        stride = sequence_length
        self.windows = [
            token_ids[start : start + sequence_length + 1]
            for start in range(0, len(token_ids) - sequence_length, stride)
        ]
        if not self.windows:
            raise ValueError("the corpus is too short for the selected sequence length")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        window = torch.tensor(self.windows[index], dtype=torch.long)
        return window[:-1], window[1:]


class ToyLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_len: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = SinusoidalPositionEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden = self.position(self.embedding(input_ids))
        causal_mask = make_causal_mask(input_ids.size(1), input_ids.device)
        for layer in self.layers:
            hidden = layer(hidden, mask=causal_mask)
        return self.lm_head(self.norm(hidden))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-file", type=Path, default=ROOT.parent / "poetryFromTang.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "ckpt/toy_lm.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )

    text = args.text_file.read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer.build([text], max_vocab_size=6000, min_frequency=1)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    dataset = TokenWindows(token_ids, args.sequence_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = ToyLanguageModel(
        tokenizer.vocab_size,
        args.d_model,
        args.n_heads,
        args.n_layers,
        args.d_ff,
        args.sequence_length,
        args.dropout,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for input_ids, targets in loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch={epoch:02d} loss={total_loss / len(loader):.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "tokenizer": tokenizer.to_dict(),
            "args": vars(args),
        },
        args.output,
    )


if __name__ == "__main__":
    main()

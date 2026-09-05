"""Create attention heatmaps for positive, negative, and long dev examples."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
import torch

from src.model import load_checkpoint

ROOT = Path(__file__).resolve().parent

# The AutoDL image may not choose a CJK font automatically even when one is
# installed. Keep Chinese character labels readable in the exported heatmaps.
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
CJK_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
CJK_FONT = FontProperties(fname=CJK_FONT_PATH) if CJK_FONT_PATH.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "ckpt/best.pt")
    parser.add_argument("--data-file", type=Path, default=ROOT / "data/validation.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--max-display-tokens", type=int, default=64)
    return parser.parse_args()


def choose_examples(frame: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    positive = frame[frame["label"] == 1].iloc[0]
    negative = frame[frame["label"] == 0].iloc[0]
    lengths = frame["text"].astype(str).str.len()
    long_example = frame.loc[lengths.idxmax()]
    return [("positive", positive), ("negative", negative), ("long", long_example)]


@torch.no_grad()
def render_example(
    name: str,
    text: str,
    label: int,
    model,
    tokenizer,
    layer_index: int,
    head_index: int,
    max_tokens: int,
    output_dir: Path,
) -> None:
    max_length = min(model.config.max_len, max_tokens)
    ids = tokenizer.encode(text, max_length=max_length)
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    logits, attentions = model(input_ids, return_attentions=True)
    layer_index = layer_index % len(attentions)
    if not 0 <= head_index < model.config.n_heads:
        raise ValueError(f"head must be between 0 and {model.config.n_heads - 1}")
    weights = attentions[layer_index][0, head_index].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    prediction = int(logits.argmax(dim=-1).item())

    side = max(8.0, min(15.0, len(tokens) * 0.23))
    figure, axis = plt.subplots(figsize=(side, side))
    image = axis.imshow(weights, cmap="viridis", aspect="auto", vmin=0.0)
    axis.set_xticks(
        range(len(tokens)),
        tokens,
        rotation=90,
        fontsize=7,
        fontproperties=CJK_FONT,
    )
    axis.set_yticks(
        range(len(tokens)), tokens, fontsize=7, fontproperties=CJK_FONT
    )
    axis.set_xlabel("Key token")
    axis.set_ylabel("Query token")
    axis.set_title(
        f"{name}: gold={label}, prediction={prediction}, layer={layer_index}, head={head_index}"
    )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_dir / f"attention_{name}.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    model, tokenizer, _ = load_checkpoint(args.checkpoint, device="cpu")
    model.eval()
    frame = pd.read_parquet(args.data_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, row in choose_examples(frame):
        render_example(
            name=name,
            text=str(row["text"]),
            label=int(row["label"]),
            model=model,
            tokenizer=tokenizer,
            layer_index=args.layer,
            head_index=args.head,
            max_tokens=args.max_display_tokens,
            output_dir=args.output_dir,
        )
    print(f"saved three heatmaps to {args.output_dir}")


if __name__ == "__main__":
    main()

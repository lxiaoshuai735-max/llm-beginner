"""Generate comparable samples with four decoding strategies."""

import json
from pathlib import Path

import torch

from src.model import load_for_eval

ROOT = Path(__file__).resolve().parent


def main():
    model, tokenizer = load_for_eval(str(ROOT / "ckpt/best.pt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prompt = "床前明月光"
    strategies = {
        "greedy": {"temperature": 0.0},
        "temperature": {"temperature": 0.8},
        "top_k": {"temperature": 0.8, "top_k": 20},
        "top_p": {"temperature": 0.8, "top_p": 0.9},
    }
    samples = {}
    for name, options in strategies.items():
        output = model.generate(tokenizer.encode(prompt), 80, **options)[0].cpu().tolist()
        # A fixed generation budget can stop halfway through a UTF-8 code point.
        samples[name] = tokenizer.decode(output).replace("�", "")
        print(f"[{name}] {samples[name]}")
    (ROOT / "generation_samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

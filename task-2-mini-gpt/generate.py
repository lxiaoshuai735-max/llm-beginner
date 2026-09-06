"""Generate comparable samples with four decoding strategies."""

import json
from pathlib import Path

import torch

from src.model import load_for_eval

ROOT = Path(__file__).resolve().parent


def main():
    torch.manual_seed(42)
    model, tokenizer = load_for_eval(str(ROOT / "ckpt/best.pt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prompt = "君不见黄河之水天上来"
    repetition_controls = {
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
    }
    strategies = {
        "greedy": {"temperature": 0.0, **repetition_controls},
        "temperature": {"temperature": 0.7, **repetition_controls},
        "top_k": {"temperature": 0.7, "top_k": 12, **repetition_controls},
        "top_p": {"temperature": 0.7, "top_p": 0.85, **repetition_controls},
    }
    samples = {}
    for name, options in strategies.items():
        output = model.generate(tokenizer.encode(prompt), 48, **options)[0].cpu().tolist()
        # A fixed generation budget can stop halfway through a UTF-8 code point.
        samples[name] = tokenizer.decode(output).replace("�", "")
        print(f"[{name}] {samples[name]}")
    (ROOT / "generation_samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

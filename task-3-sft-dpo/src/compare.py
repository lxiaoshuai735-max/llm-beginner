"""Generate base/SFT/DPO outputs for the same prompts."""

import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.chat import format_messages
    from src.lora import inject_lora, load_lora
else:
    from .chat import format_messages
    from .lora import inject_lora, load_lora

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def generate(model, tokenizer, prompt):
    text = format_messages([{"role": "user", "content": prompt}]) + "<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    return tokenizer.decode(output[0, inputs.input_ids.size(1):], skip_special_tokens=True).strip()


def main():
    model_path = ROOT / "models/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    prompts = ["什么是深度学习？", "如何保护账号安全？"]
    model = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=torch.bfloat16).to("cuda").eval()
    results = {prompt: {"base": generate(model, tokenizer, prompt)} for prompt in prompts}
    inject_lora(model, ["q_proj", "v_proj"], r=8, alpha=16)
    load_lora(model, ROOT / "ckpt/sft")
    model.eval()
    for prompt in prompts:
        results[prompt]["sft"] = generate(model, tokenizer, prompt)
    load_lora(model, ROOT / "ckpt/dpo")
    model.eval()
    for prompt in prompts:
        results[prompt]["dpo"] = generate(model, tokenizer, prompt)
    (ROOT / "comparison.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

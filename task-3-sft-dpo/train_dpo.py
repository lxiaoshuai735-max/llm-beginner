"""Full DPO-En-Zh-20k training from the completed SFT adapter."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.chat import tokenize_conversation
from src.data_utils import load_dpo_pairs
from src.lora import load_lora, save_lora

ROOT = Path(__file__).resolve().parent


def sequence_log_probability(model, input_ids, labels):
    logits = model(input_ids=input_ids).logits[:, :-1]
    targets = labels[:, 1:]
    mask = targets.ne(-100)
    safe_targets = targets.masked_fill(~mask, 0)
    token_logps = logits.log_softmax(-1).gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(-1)


def encoded_pair(tokenizer, item, max_length):
    prefix = item["messages"]
    chosen = prefix + [{"role": "assistant", "content": item["chosen"]}]
    rejected = prefix + [{"role": "assistant", "content": item["rejected"]}]
    chosen_ids, chosen_labels = tokenize_conversation(tokenizer, chosen, max_length)
    rejected_ids, rejected_labels = tokenize_conversation(tokenizer, rejected, max_length)
    return (
        chosen_ids[None].cuda(),
        chosen_labels[None].cuda(),
        rejected_ids[None].cuda(),
        rejected_labels[None].cuda(),
    )


def save_checkpoint(policy, output_dir, config, args, step, history, pair_count):
    save_lora(
        policy,
        output_dir,
        target_modules=config["target_modules"],
        r=config["r"],
        alpha=config["alpha"],
        optimizer_steps=step,
        pairs_available=pair_count,
        epochs=args.epochs,
        beta=args.beta,
        full_corpus=args.max_steps == 0 and args.max_samples == 0,
    )
    (output_dir / "training.json").write_text(
        json.dumps({"config": vars(args), "history": history}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means all selected pairs")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all 20k pairs")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument(
        "--sft-dir",
        default="ckpt/sft",
        help="SFT adapter directory produced by train_sft.py",
    )
    parser.add_argument(
        "--output-dir",
        default="ckpt/dpo",
        help="Final DPO adapter directory; matches src/compare.py",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    model_path = ROOT / "models/Qwen2.5-0.5B"
    sft_dir = ROOT / args.sft_dir
    if not (sft_dir / "adapter.pt").exists():
        raise FileNotFoundError(f"completed SFT adapter not found at {sft_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    policy = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=torch.bfloat16)
    load_lora(policy, sft_dir)
    reference = copy.deepcopy(policy).eval().cuda()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    policy = policy.cuda()
    policy.config.use_cache = False
    reference.config.use_cache = False

    pairs = load_dpo_pairs(ROOT / "data/dpo")
    if args.max_samples:
        pairs = pairs[: args.max_samples]
    if not pairs:
        raise RuntimeError("no valid DPO pairs were found")
    config = json.loads((sft_dir / "adapter_config.json").read_text(encoding="utf-8"))
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate)
    history = []
    step = 0
    started = time.time()
    policy.train()

    for epoch in range(args.epochs):
        order = list(range(len(pairs)))
        random.Random(42 + epoch).shuffle(order)
        for index in order:
            item = pairs[index]
            c_ids, c_labels, r_ids, r_labels = encoded_pair(tokenizer, item, args.max_length)
            if c_labels.ne(-100).sum().item() == 0 or r_labels.ne(-100).sum().item() == 0:
                continue
            with torch.no_grad():
                ref_chosen = sequence_log_probability(reference, c_ids, c_labels)
                ref_rejected = sequence_log_probability(reference, r_ids, r_labels)
            policy_chosen = sequence_log_probability(policy, c_ids, c_labels)
            policy_rejected = sequence_log_probability(policy, r_ids, r_labels)
            margin = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
            loss = -F.logsigmoid(args.beta * margin).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            step += 1
            history.append(
                {"step": step, "loss": float(loss.item()), "reward_margin": float(margin.mean().item())}
            )
            if step == 1 or step % args.log_every == 0:
                rate = step / max(time.time() - started, 1e-6)
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={step} loss={loss.item():.4f} "
                    f"reward_margin={margin.mean().item():.4f} pairs_per_second={rate:.2f}",
                    flush=True,
                )
            if args.save_every and step % args.save_every == 0:
                save_checkpoint(
                    policy,
                    ROOT / f"ckpt/dpo-step-{step:06d}",
                    config,
                    args,
                    step,
                    history,
                    len(pairs),
                )
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    if not history:
        raise RuntimeError("DPO produced no optimizer steps")
    save_checkpoint(policy, ROOT / args.output_dir, config, args, step, history, len(pairs))
    print(f"DPO complete: {step} optimizer steps from {len(pairs)} available pairs", flush=True)


if __name__ == "__main__":
    main()

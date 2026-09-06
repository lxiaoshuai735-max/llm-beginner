"""Full-corpus handwritten-LoRA SFT training for Qwen2.5-0.5B."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.chat import tokenize_conversation
from src.data_utils import buffered_shuffle, iter_sft_dialogues
from src.lora import inject_lora, save_lora

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means the full corpus")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means every valid record")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--shuffle-buffer", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument(
        "--output-dir",
        default="ckpt/sft",
        help="Final adapter directory; matches eval/run.py and src/compare.py",
    )
    return parser.parse_args()


def collate(batch, pad_token_id):
    length = max(item[0].numel() for item in batch)
    input_ids = torch.full((len(batch), length), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), length), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), length), dtype=torch.long)
    for row, (ids, item_labels) in enumerate(batch):
        input_ids[row, : ids.numel()] = ids
        labels[row, : ids.numel()] = item_labels
        attention_mask[row, : ids.numel()] = 1
    return input_ids.cuda(), attention_mask.cuda(), labels.cuda()


def checkpoint(model, output_dir, args, optimizer_steps, samples_seen, final_loss):
    save_lora(
        model,
        output_dir,
        target_modules=["q_proj", "v_proj"],
        r=args.rank,
        alpha=args.alpha,
        optimizer_steps=optimizer_steps,
        samples_seen=samples_seen,
        epochs=args.epochs,
        full_corpus=args.max_steps == 0 and args.max_samples == 0,
        final_loss=final_loss,
    )


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation < 1:
        raise ValueError("epochs, batch-size and gradient-accumulation must be positive")
    torch.manual_seed(42)
    model_path = ROOT / "models/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=torch.bfloat16).cuda()
    model.config.use_cache = False
    inject_lora(model, ["q_proj", "v_proj"], r=args.rank, alpha=args.alpha, dropout=0.05)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate)

    history = []
    micro_batch = []
    optimizer_steps = 0
    samples_seen = 0
    accumulated = 0
    running_loss = 0.0
    started = time.time()
    stop = False
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        dialogues = buffered_shuffle(
            iter_sft_dialogues(ROOT / "data/moss-sft"),
            buffer_size=args.shuffle_buffer,
            seed=42 + epoch,
        )
        for messages in dialogues:
            if args.max_samples and samples_seen >= args.max_samples:
                stop = True
                break
            encoded = tokenize_conversation(tokenizer, messages, args.max_length)
            if encoded[1].ne(-100).sum().item() == 0:
                continue
            micro_batch.append(encoded)
            if len(micro_batch) < args.batch_size:
                continue
            input_ids, attention_mask, labels = collate(micro_batch, tokenizer.pad_token_id)
            micro_batch = []
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            (loss / args.gradient_accumulation).backward()
            accumulated += 1
            samples_seen += input_ids.size(0)
            running_loss += float(loss.item())
            if accumulated < args.gradient_accumulation:
                continue

            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            next_step = optimizer_steps + 1
            warmup_scale = min(1.0, next_step / max(1, args.warmup_steps))
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * warmup_scale
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps = next_step
            accumulated = 0
            mean_loss = running_loss / args.gradient_accumulation
            running_loss = 0.0
            history.append({"step": optimizer_steps, "loss": mean_loss, "samples": samples_seen})

            if optimizer_steps == 1 or optimizer_steps % args.log_every == 0:
                elapsed = max(time.time() - started, 1e-6)
                rate = samples_seen / elapsed
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={optimizer_steps} "
                    f"samples={samples_seen} loss={mean_loss:.4f} samples_per_second={rate:.2f}",
                    flush=True,
                )
            if args.save_every and optimizer_steps % args.save_every == 0:
                checkpoint(
                    model,
                    ROOT / f"ckpt/sft-step-{optimizer_steps:06d}",
                    args,
                    optimizer_steps,
                    samples_seen,
                    mean_loss,
                )
            if args.max_steps and optimizer_steps >= args.max_steps:
                stop = True
                break
        if stop:
            break

    if not history:
        raise RuntimeError("no valid MOSS training examples were found")
    final_dir = ROOT / args.output_dir
    checkpoint(model, final_dir, args, optimizer_steps, samples_seen, history[-1]["loss"])
    (final_dir / "training.json").write_text(
        json.dumps(
            {
                "config": vars(args),
                "optimizer_steps": optimizer_steps,
                "samples_seen": samples_seen,
                "elapsed_seconds": math.ceil(time.time() - started),
                "history": history,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"SFT complete: {samples_seen} samples, {optimizer_steps} optimizer steps", flush=True)


if __name__ == "__main__":
    main()

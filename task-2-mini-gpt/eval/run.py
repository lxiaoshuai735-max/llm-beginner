"""Task 2 self-check: tokenizer, cache, normalized loss, and M5 generation."""

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))           # from src.* -- student implementation
sys.path.insert(0, str(ROOT.parent))    # from _eval_harness -- shared runner

from _eval_harness import run_tests


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset_info():
    path = ROOT / "data" / "dataset_info.json"
    if not path.exists():
        return {"dataset": "unknown", "ppl_threshold": 200}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"dataset": "unknown", "ppl_threshold": 200}


def test_tokenizer_roundtrip():
    from src.tokenizer import BPETokenizer

    tok_path = ROOT / "ckpt" / "tokenizer.json"
    if not tok_path.exists():
        return {"test": "tokenizer_roundtrip", "pass": None,
                "skip": "ckpt/tokenizer.json does not exist"}
    tok = BPETokenizer.from_pretrained(str(tok_path))
    samples = ["床前明月光", "Hello, world!", "深度学习需要数学基础"]
    failures = []
    for text in samples:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        if text != decoded:
            failures.append({"in": text, "out": decoded})
    return {"test": "tokenizer_roundtrip", "pass": not failures, "failures": failures}


def test_utf8_generation_guard():
    """Check strict UTF-8 state transitions independently of BPE segmentation."""
    from src.tokenizer import BPETokenizer

    state = 0
    for byte in "行".encode("utf-8"):
        state = BPETokenizer.advance_utf8_state(state, bytes([byte]))
        if state is None:
            return {"test": "utf8_generation_guard", "pass": False,
                    "detail": "a valid Chinese character was rejected"}
    invalid = BPETokenizer.advance_utf8_state(0, bytes([0x80]))
    return {
        "test": "utf8_generation_guard",
        "pass": state == 0 and invalid is None,
        "complete_character_state": state,
        "stray_continuation_rejected": invalid is None,
    }


def test_kv_cache_equivalence():
    from src.model import load_for_eval

    ckpt = ROOT / "ckpt" / "best.pt"
    if not ckpt.exists():
        return {"test": "kv_cache_equivalence", "pass": None,
                "skip": "ckpt/best.pt does not exist"}
    model, tok = load_for_eval(str(ckpt))
    model.eval()
    ids = torch.tensor([tok.encode("从前有座山")], dtype=torch.long)
    with torch.no_grad():
        logits_full = model(ids)
        cache = None
        logits_inc = []
        for index in range(ids.size(1)):
            output, cache = model(ids[:, index:index + 1], kv_cache=cache, return_cache=True)
            logits_inc.append(output)
        logits_inc = torch.cat(logits_inc, dim=1)
    diff = (logits_full - logits_inc).abs().max().item()
    return {"test": "kv_cache_equivalence", "pass": diff < 1e-4,
            "max_abs_diff": diff}


def test_perplexity_on_dev():
    """Report token PPL and UTF-8-byte-normalized NLL for each BPE experiment."""
    from src.model import load_for_eval

    ckpt = ROOT / "ckpt" / "best.pt"
    dev_path = ROOT / "data" / "dev.txt"
    info = load_dataset_info()
    threshold = float(info.get("ppl_threshold", 200))
    if not ckpt.exists():
        return {"test": "perplexity_on_dev", "pass": None,
                "skip": "ckpt/best.pt does not exist"}
    if not dev_path.exists():
        return {"test": "perplexity_on_dev", "pass": None,
                "skip": "data/dev.txt does not exist; build the dataset first"}
    model, tok = load_for_eval(str(ckpt))
    model.eval()

    block = getattr(model, "block_size", None) or getattr(model, "max_seq_len", 256)
    ids = tok.encode(dev_path.read_text(encoding="utf-8"))[:4096]
    nll, n_tokens, n_bytes = 0.0, 0, 0
    with torch.no_grad():
        for start in range(0, max(1, len(ids) - 1), block):
            window = ids[start:start + block + 1]
            if len(window) < 2:
                break
            chunk = torch.tensor([window], dtype=torch.long)
            logits = model(chunk)
            nll += F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                chunk[:, 1:].reshape(-1), reduction="sum").item()
            n_tokens += chunk.size(1) - 1
            n_bytes += len(tok.ids_to_bytes(window[1:]))
    if n_tokens == 0 or n_bytes == 0:
        return {"test": "perplexity_on_dev", "pass": None,
                "skip": "the development split is too short"}
    token_ppl = math.exp(nll / n_tokens)
    byte_nll = nll / n_bytes
    return {
        "test": "perplexity_on_dev",
        "pass": token_ppl < threshold,
        "token_perplexity": round(token_ppl, 2),
        "threshold": threshold,
        "utf8_byte_nll": round(byte_nll, 4),
        "utf8_byte_perplexity": round(math.exp(byte_nll), 4),
        "n_tokens": n_tokens,
        "n_utf8_bytes": n_bytes,
        "dataset": info.get("dataset", "unknown"),
        "comparison_note": "Only byte-normalized NLL/PPL is comparable across BPE vocabularies.",
    }


def test_m5_generation_quality():
    """Require fixed held-out samples plus an explicit human readability review."""
    generation_path = ROOT / "eval" / "generation_quality.json"
    review_path = ROOT / "eval" / "generation_review.json"
    ckpt = ROOT / "ckpt" / "best.pt"
    tokenizer_path = ROOT / "ckpt" / "tokenizer.json"
    if not generation_path.exists() or not review_path.exists():
        return {
            "test": "m5_generation_quality",
            "pass": False,
            "detail": "Run generate.py and complete eval/generation_review.json before accepting M5.",
        }
    try:
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        review = json.loads(review_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return {"test": "m5_generation_quality", "pass": False, "detail": str(error)}

    samples = generation.get("samples", [])
    strategies = {"greedy", "temperature", "top_k", "top_p"}
    sample_keys = [(str(item.get("prompt_id")), item.get("strategy")) for item in samples]
    reasons: list[str] = []
    if generation.get("prompt_count") != 10:
        reasons.append("generation file must use exactly 10 fixed held-out prompts")
    if len(samples) != 40 or set(strategy for _, strategy in sample_keys) != strategies:
        reasons.append("need 10 prompts x greedy/temperature/top_k/top_p")
    if len(sample_keys) != len(set(sample_keys)):
        reasons.append("duplicate prompt/strategy samples found")
    if ckpt.exists() and generation.get("checkpoint_sha256") != sha256(ckpt):
        reasons.append("generation samples do not match the current checkpoint")
    if tokenizer_path.exists() and generation.get("tokenizer_sha256") != sha256(tokenizer_path):
        reasons.append("generation samples do not match the current tokenizer")

    for item in samples:
        text = item.get("text", "")
        parameters = item.get("parameters", {})
        try:
            raw_text = bytes.fromhex(item.get("raw_utf8_bytes_hex", "")).decode("utf-8", errors="strict")
        except (TypeError, ValueError, UnicodeDecodeError):
            reasons.append("a saved raw sample is not valid UTF-8")
            break
        if raw_text != text or not item.get("complete_utf8_boundary") or item.get("trimmed_trailing_token_count") != 0:
            reasons.append("a saved sample changed bytes or ended mid-character")
            break
        if "\ufffd" in text or len(text) <= len(str(item.get("prompt", ""))):
            reasons.append("a saved sample contains a replacement character or no continuation")
            break
        if parameters.get("repetition_penalty") != 1.0 or parameters.get("no_repeat_ngram_size") != 0:
            reasons.append("M5 baseline must not use byte-token repetition penalties")
            break
        if not parameters.get("enforce_utf8") or not parameters.get("complete_utf8_at_end"):
            reasons.append("M5 sample lacks strict UTF-8 generation metadata")
            break

    reviews = review.get("reviews", [])
    review_keys = [(str(item.get("prompt_id")), item.get("strategy")) for item in reviews]
    required_flags = ("readable", "theme_continuity", "no_mechanical_repetition")
    if set(review_keys) != set(sample_keys):
        reasons.append("manual review does not cover every saved sample")
        passed_keys: set[tuple[str, str]] = set()
    else:
        passed_keys = {
            (str(item.get("prompt_id")), item.get("strategy"))
            for item in reviews
            if all(item.get(flag) is True for flag in required_flags)
        }
        overall_ratio = len(passed_keys) / len(sample_keys) if sample_keys else 0.0
        per_strategy = {
            strategy: sum(key in passed_keys for key in sample_keys if key[1] == strategy)
            for strategy in strategies
        }
        if overall_ratio < 0.8:
            reasons.append("fewer than 80% of samples passed all manual-review criteria")
        if any(count < 7 for count in per_strategy.values()):
            reasons.append("each decoding strategy must pass at least 7 of 10 prompts")

    passed_count = len(passed_keys)
    per_strategy_passed = {
        strategy: sum(key in passed_keys for key in sample_keys if key[1] == strategy)
        for strategy in strategies
    }

    return {
        "test": "m5_generation_quality",
        "pass": not reasons,
        "sample_count": len(samples),
        "review_count": len(reviews),
        "manual_pass_count": passed_count,
        "manual_pass_ratio": round(passed_count / len(sample_keys), 3) if sample_keys else 0.0,
        "manual_pass_by_strategy": per_strategy_passed,
        "strategy_counts": dict(Counter(item.get("strategy") for item in samples)),
        "reasons": reasons,
    }


if __name__ == "__main__":
    run_tests([
        test_tokenizer_roundtrip,
        test_utf8_generation_guard,
        test_kv_cache_equivalence,
        test_perplexity_on_dev,
        test_m5_generation_quality,
    ], ROOT)

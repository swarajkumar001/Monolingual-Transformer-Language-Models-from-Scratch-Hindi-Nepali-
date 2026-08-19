#!/usr/bin/env python3
"""
Step 13 -- evaluate the selected tokenizer and backfill real token counts.

    python hindi/scripts/s13_tokenizer_eval.py

Produces everything the brief's tokenizer section asks for: vocabulary size,
token-frequency statistics, average characters per token, fertility and UNK
rate on held-out (test) text, and worked tokenization examples. This is also
where ``n_tokens`` -- deliberately absent on every document since collection,
because a real count needs a trained tokenizer (see ``corpuslib/schema.py``)
-- finally gets written: all three final splits are re-tokenized and
rewritten in place with the field populated, which is what lets S14 report
real token totals instead of the word-count estimate everywhere after this.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpuslib import runlog, tokeval  # noqa: E402
from corpuslib.config import load_language_config  # noqa: E402
from corpuslib.jsonlio import atomic_write_json, read_jsonl, write_jsonl  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-examples", type=int, default=0)
    parser.add_argument("--top-tokens", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_language_config()
    config.ensure_dirs()
    log = runlog.StepLogger(config.lang, config.events_path)
    started = time.time()

    import yaml
    tok_cfg = yaml.safe_load((config.configs_dir / "tokenizer.yaml").read_text("utf-8"))
    model_prefix_name = tok_cfg.get("model_prefix", f"{config.lang}_bpe")
    report_cfg = tok_cfg.get("report", {}) or {}
    n_examples = args.n_examples or int(report_cfg.get("n_examples", 10))
    top_tokens = args.top_tokens or int(report_cfg.get("top_tokens", 50))

    model_path = config.tokenizer_dir / f"{model_prefix_name}.model"
    if not model_path.exists():
        print(f"BLOCKED: {model_path} does not exist. Run s12_train_tokenizer.py first.")
        log("S13", f"Tokenizer evaluation ({config.display_name})", runlog.BLOCKED,
            "No trained tokenizer found.")
        return 1

    splits = {}
    for name in ("train", "val", "test"):
        path = config.final_dir / f"{name}.jsonl"
        if not path.exists():
            print(f"BLOCKED: {path} missing. Run s11_finalize_splits.py first.")
            log("S13", f"Tokenizer evaluation ({config.display_name})", runlog.BLOCKED,
                f"{name}.jsonl missing.")
            return 1
        splits[name] = path

    import sentencepiece as spm
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    print(
        f"S13 tokenizer eval ({config.display_name}) -- model={model_path.name} "
        f"vocab_size={processor.vocab_size():,}",
        flush=True,
    )

    # -- retokenise every split, backfilling n_tokens ------------------------
    split_totals = {}
    for name, path in splits.items():
        records = list(read_jsonl(path))
        total_tokens = 0
        for record in records:
            pieces = processor.encode(record["text"], out_type=str)
            record["n_tokens"] = len(pieces)
            total_tokens += len(pieces)
        write_jsonl(path, records)
        split_totals[name] = {"docs": len(records), "tokens": total_tokens}
        print(f"  {name}: {len(records):,} docs, {total_tokens:,} real tokens", flush=True)

    # -- fertility / UNK / char-per-token on TEST, and token-frequency on TRAIN
    test_texts = (r["text"] for r in read_jsonl(splits["test"]))
    test_stats = tokeval.measure(processor, test_texts)

    train_texts = (r["text"] for r in read_jsonl(splits["train"]))
    train_stats = tokeval.measure(processor, train_texts)
    most_common = train_stats["piece_counts"].most_common(top_tokens)

    # -- worked examples ------------------------------------------------------
    examples = []
    for record in read_jsonl(splits["test"]):
        text = record["text"].strip()
        if not (20 <= len(text) <= 200):
            continue
        pieces = processor.encode(text, out_type=str)
        examples.append({"text": text, "pieces": pieces, "n_tokens": len(pieces)})
        if len(examples) >= n_examples:
            break

    elapsed = time.time() - started
    payload = {
        "model_path": str(model_path),
        "vocab_size": processor.vocab_size(),
        "split_totals": split_totals,
        "test_fertility": round(test_stats["fertility"], 4),
        "test_unk_rate": round(test_stats["unk_rate"], 5),
        "test_chars_per_token": round(test_stats["chars_per_token"], 3),
        "train_chars_per_token": round(train_stats["chars_per_token"], 3),
        "top_tokens": [{"piece": p, "count": c} for p, c in most_common],
        "examples": examples,
        "seconds": round(elapsed, 1),
    }
    atomic_write_json(config.tokenizer_dir / "evaluation_report.json", payload)

    print(f"  test fertility={test_stats['fertility']:.4f}  "
          f"unk_rate={test_stats['unk_rate']:.4%}  "
          f"chars/token={test_stats['chars_per_token']:.3f}")
    print(f"  wrote {config.tokenizer_dir / 'evaluation_report.json'}")

    log(
        "S13",
        f"Tokenizer evaluation ({config.display_name})",
        runlog.SUCCESS,
        f"vocab={processor.vocab_size():,}, test fertility "
        f"{test_stats['fertility']:.4f}, test UNK rate {test_stats['unk_rate']:.4%}. "
        f"Backfilled real n_tokens on all three splits "
        f"({sum(s['tokens'] for s in split_totals.values()):,} tokens total).",
        metrics={k: v for k, v in payload.items() if k not in ("top_tokens", "examples")},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

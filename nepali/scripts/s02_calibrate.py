#!/usr/bin/env python3
"""
Step 02 -- measure the real words -> tokens factor for this language.

    python hindi/scripts/s02_calibrate.py
    python hindi/scripts/s02_calibrate.py --sample-chars 50_000_000   # faster, less stable

Every word-based budget in ``configs/language.yaml`` (collection targets, the
20%/24% manual ratios) is provisional until a tokenizer exists, because the
words->tokens ratio for Devanagari BPE is not 1:1 and guessing it is exactly
what produced the previous plan's unusable budgets. This step trains a
throwaway SentencePiece model on a sample of the *raw* collected text (before
any cleaning removes material) and measures the ratio directly, so every
step after this one can convert between words and tokens with a number this
project actually measured rather than assumed.

The throwaway model here is not one of the candidate tokenizers -- S12 trains
those, on cleaned text, sweeping the real vocabulary sizes. This model exists
only to produce a stable fertility estimate quickly.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpuslib import runlog, schema, stageio  # noqa: E402
from corpuslib.config import load_language_config  # noqa: E402
from corpuslib.jsonlio import atomic_write_json  # noqa: E402

CALIBRATION_VOCAB_SIZE = 8000
MIN_PLAUSIBLE_FACTOR = 1.3
MAX_PLAUSIBLE_FACTOR = 2.6


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-chars",
        type=int,
        default=200_000_000,
        help="stop sampling raw text once this many characters are collected",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_language_config()
    config.ensure_dirs()
    log = runlog.StepLogger(config.lang, config.events_path)
    started = time.time()

    print(f"S02 calibration ({config.display_name}) -- sampling raw text...", flush=True)

    sample_chars = 0
    sample_words = 0
    docs_seen = 0
    lines: list[str] = []
    for record in stageio.iter_raw_documents(config.raw_dir):
        text = record.get("text", "")
        if not text:
            continue
        docs_seen += 1
        sample_chars += len(text)
        sample_words += record.get("n_words") or schema.count_words(text)
        # SentencePiece trains line-by-line; a document's paragraphs make
        # reasonable "sentences" for this purpose without needing a real
        # sentence splitter, which S03/S12 do not have either.
        lines.extend(p for p in text.split("\n") if p.strip())
        if sample_chars >= args.sample_chars:
            break

    if docs_seen == 0 or sample_words == 0:
        log(
            "S02",
            f"Token calibration ({config.display_name})",
            runlog.BLOCKED,
            "No raw documents found -- run s01_collect.py first.",
        )
        print("BLOCKED: no raw documents under data/raw/. Run s01_collect.py first.")
        return 1

    print(
        f"  sampled {docs_seen:,} documents, {sample_chars:,} chars, "
        f"{sample_words:,} words -- training a throwaway "
        f"{CALIBRATION_VOCAB_SIZE}-token model",
        flush=True,
    )

    import sentencepiece as spm

    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / "calibration_sample.txt"
        corpus_path.write_text("\n".join(lines), encoding="utf-8")
        model_prefix = Path(tmp) / "calibration"
        # SentencePiece's C++ trainer logs one line per merge via glog -- at a
        # real vocab size that is thousands of lines. `logstream` is the only
        # hook the Python binding exposes to redirect it instead of stderr.
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            spm.SentencePieceTrainer.train(
                input=str(corpus_path),
                model_prefix=str(model_prefix),
                vocab_size=CALIBRATION_VOCAB_SIZE,
                model_type="bpe",
                character_coverage=0.9995,
                byte_fallback=False,
                input_sentence_size=min(len(lines), 2_000_000),
                shuffle_input_sentence=True,
                logstream=devnull,
            )
        processor = spm.SentencePieceProcessor(model_file=str(model_prefix) + ".model")

        total_tokens = 0
        # Re-encode in batches rather than the whole sample at once -- SentencePiece's
        # Python binding has no streaming encode, and the sample can be large.
        batch: list[str] = []
        for line in lines:
            batch.append(line)
            if len(batch) >= 5000:
                total_tokens += sum(len(p) for p in processor.encode(batch))
                batch = []
        if batch:
            total_tokens += sum(len(p) for p in processor.encode(batch))

    factor = total_tokens / sample_words if sample_words else 0.0
    elapsed = time.time() - started

    plausible = MIN_PLAUSIBLE_FACTOR <= factor <= MAX_PLAUSIBLE_FACTOR
    payload = {
        "lang": config.lang,
        "sample_docs": docs_seen,
        "sample_chars": sample_chars,
        "sample_words": sample_words,
        "sample_tokens": total_tokens,
        "tokens_per_word": round(factor, 4),
        "calibration_vocab_size": CALIBRATION_VOCAB_SIZE,
        "plausible_range": [MIN_PLAUSIBLE_FACTOR, MAX_PLAUSIBLE_FACTOR],
        "seconds": round(elapsed, 1),
    }
    out_path = config.configs_dir / "token_calibration.json"
    atomic_write_json(out_path, payload)

    print(f"  tokens_per_word = {factor:.4f}  (plausible range {MIN_PLAUSIBLE_FACTOR}-{MAX_PLAUSIBLE_FACTOR})")
    print(f"  wrote {out_path}")

    status = runlog.SUCCESS if plausible else runlog.BLOCKED
    log(
        "S02",
        f"Token calibration ({config.display_name})",
        status,
        (
            f"Measured {factor:.4f} tokens/word from {docs_seen:,} documents "
            f"({sample_words:,} words -> {total_tokens:,} tokens with an "
            f"{CALIBRATION_VOCAB_SIZE}-token throwaway model)."
            if plausible
            else (
                f"Measured factor {factor:.4f} is outside the plausible range "
                f"[{MIN_PLAUSIBLE_FACTOR}, {MAX_PLAUSIBLE_FACTOR}] -- inspect the sample "
                f"before trusting downstream token budgets."
            )
        ),
        metrics=payload,
    )
    if not plausible:
        print(
            f"BLOCKED: factor {factor:.4f} is outside [{MIN_PLAUSIBLE_FACTOR}, "
            f"{MAX_PLAUSIBLE_FACTOR}] -- something is wrong with the sample or the "
            f"training text; inspect before proceeding."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

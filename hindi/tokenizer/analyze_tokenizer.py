"""Measure a trained Hindi SentencePiece tokenizer on held-out text.

Mirrors bhojpuri/tokenizer/analyze_tokenizer.py exactly (same logic, Hindi
paths/model-name prefix). Evaluated on data/splits/validation/ (never train,
never test - validation is the correct split for a model-selection decision
like "which vocab size", keeping test genuinely unseen for Phase 2/3
evaluation).

Reports the numbers needed to justify a vocab-size choice:
  - chars/token (the ratio the rest of Phase 1/2 planning depends on)
  - fertility (tokens per whitespace-delimited word)
  - UNK rate (structurally ~0.0 with byte_fallback on - not a useful signal
    on its own, reported only for completeness)
  - byte-fallback rate (fraction of emitted tokens that are raw-byte
    fallback pieces, i.e. text the vocab genuinely could not represent as a
    normal subword - this is the metric that actually stands in for "UNK
    rate" here)
  - vocab utilization (distinct piece ids used at least once / vocab_size)
  - top tokens by frequency
  - 3 sample tokenizations with piece strings and ids
"""

import argparse
import collections
import re
import sys
from pathlib import Path

import sentencepiece as spm

# stdout can default to a non-UTF-8 codepage on Windows (cp1252) when not
# attached to a real console (e.g. redirected to a log file via `> log.txt`)
# - printing Devanagari token pieces/samples then crashes with
# UnicodeEncodeError partway through the report. Force UTF-8 explicitly so
# this script is safe to run either interactively or redirected.
sys.stdout.reconfigure(encoding="utf-8")

LANG_DIR = Path(__file__).resolve().parents[1]
VAL_TEXT_DIR = LANG_DIR / "data" / "splits" / "validation" / "text"
TOKENIZER_DIR = Path(__file__).resolve().parent
_BYTE_FALLBACK_PIECE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def load_validation_text(max_docs: int) -> list[str]:
    """WHY: analysis must run on text the tokenizer never trained on
    (validation, not train) so fertility/utilization/byte-fallback numbers
    reflect genuine held-out generalization, not memorized training text.
    WHAT: reads up to max_docs .txt files from data/splits/validation/,
    skipping any that are empty after stripping.
    INPUT: max_docs cap on how many files to read (sampling, not the full
    validation split, since 2,000 docs is enough for stable rate estimates
    and reading all ~14k would be needlessly slow for a sanity-check pass).
    OUTPUT: list of document text strings."""
    files = sorted(VAL_TEXT_DIR.glob("*.txt"))
    if not files:
        raise SystemExit(f"no validation text files at {VAL_TEXT_DIR} - run split_hindi.py first")
    texts = []
    for fp in files[:max_docs]:
        text = fp.read_text(encoding="utf-8").strip()
        if text:
            texts.append(text)
    return texts


def analyze(model_path: Path, texts: list[str], n_samples: int = 3) -> dict:
    """WHY: bundles every metric this project's vocab-size decision is
    based on (see the module docstring's list) into one pass over the
    validation sample, so a single function call produces everything
    tokenizer_analysis.md quotes for one vocab size.
    WHAT: encodes each text with the SentencePiece model (both as ids and
    as piece strings, since UNK/byte-fallback detection needs the string
    form but chars/token needs only counts), accumulates running totals,
    and separately re-encodes the first n_samples texts to keep example
    (text, pieces, ids) triples for the report.
    INPUT: model_path (a trained .model file); texts (from
    load_validation_text); n_samples (how many worked examples to keep).
    OUTPUT: dict of every metric plus 'samples' - see the return statement
    for the exact keys; this is what print_report formats."""
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    vocab_size = sp.vocab_size()
    unk_id = sp.unk_id()

    total_chars = 0
    total_tokens = 0
    total_words = 0
    unk_count = 0
    byte_fallback_count = 0
    piece_freq = collections.Counter()
    used_ids = set()

    for text in texts:
        ids = sp.encode(text, out_type=int)
        pieces = sp.encode(text, out_type=str)
        total_chars += len(text)
        total_tokens += len(ids)
        total_words += len(text.split())
        for tid, piece in zip(ids, pieces):
            used_ids.add(tid)
            piece_freq[piece] += 1
            if tid == unk_id:
                unk_count += 1
            if _BYTE_FALLBACK_PIECE.match(piece):
                byte_fallback_count += 1

    samples = []
    for text in texts[:n_samples]:
        ids = sp.encode(text, out_type=int)
        pieces = sp.encode(text, out_type=str)
        samples.append({
            "text": text[:80] + ("..." if len(text) > 80 else ""),
            "pieces": pieces[:30],
            "ids": ids[:30],
        })

    return {
        "vocab_size": vocab_size,
        "n_docs_evaluated": len(texts),
        "chars_per_token": total_chars / total_tokens if total_tokens else float("nan"),
        "fertility_tokens_per_word": total_tokens / total_words if total_words else float("nan"),
        "unk_rate": unk_count / total_tokens if total_tokens else float("nan"),
        "byte_fallback_rate": byte_fallback_count / total_tokens if total_tokens else float("nan"),
        "vocab_utilization": len(used_ids) / vocab_size,
        "top_tokens": piece_freq.most_common(15),
        "samples": samples,
    }


def print_report(result: dict):
    """WHY: keeps formatting separate from computation (analyze() returns
    plain data, this only renders it) so the dict can also be consumed
    programmatically elsewhere without re-parsing printed text.
    WHAT: prints every metric from analyze()'s return dict in the fixed
    layout tokenizer_analysis.md's numbers were transcribed from.
    INPUT: result dict from analyze().
    OUTPUT: none (prints to stdout only)."""
    print(f"\n{'=' * 70}")
    print(f"vocab_size={result['vocab_size']}  (evaluated on {result['n_docs_evaluated']} validation docs)")
    print(f"{'=' * 70}")
    print(f"  chars/token:        {result['chars_per_token']:.4f}")
    print(f"  fertility (tok/wd): {result['fertility_tokens_per_word']:.4f}")
    print(f"  UNK rate:           {result['unk_rate']:.6f}  (structurally ~0 with byte_fallback on)")
    print(f"  byte-fallback rate: {result['byte_fallback_rate']:.6f}")
    print(f"  vocab utilization:  {result['vocab_utilization']:.4f}")
    print(f"  top tokens: {result['top_tokens']}")
    print("  sample tokenizations:")
    for s in result["samples"]:
        print(f"    text:   {s['text']}")
        print(f"    pieces: {s['pieces']}")
        print(f"    ids:    {s['ids']}")


def main():
    """WHY: CLI entry point so the sweep driver (run_hindi.bat /
    run_hindi_tokenizer_sweep.sh) can analyze one vocab size at a time
    right after training it.
    WHAT: loads the requested vocab size's model, loads validation text,
    runs analyze(), prints the report.
    INPUT: CLI args --vocab-size (required), --max-docs (default 2000).
    OUTPUT: none (prints only); exits with an error if that vocab size
    hasn't been trained yet."""
    parser = argparse.ArgumentParser(description="Analyze a trained Hindi tokenizer on held-out text")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--max-docs", type=int, default=2000)
    args = parser.parse_args()

    model_path = TOKENIZER_DIR / f"hindi_unigram_{args.vocab_size}.model"
    if not model_path.exists():
        raise SystemExit(f"{model_path} not found - run train_tokenizer.py --vocab-size {args.vocab_size} first")

    texts = load_validation_text(args.max_docs)
    result = analyze(model_path, texts)
    print_report(result)


if __name__ == "__main__":
    main()

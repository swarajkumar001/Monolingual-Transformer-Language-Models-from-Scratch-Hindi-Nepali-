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

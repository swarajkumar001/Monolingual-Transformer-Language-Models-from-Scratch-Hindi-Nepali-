"""Compute the real total token count for a language's corpus, using an
actual trained SentencePiece model over every document - not a sample-based
chars/token extrapolation.

Reports per split (train/validation/test) and overall: doc count, char
count, token count, and the manual-vs-downloaded token split (needed to
confirm the 20% manual floor holds at the token level, not just the char
level the pipeline already validates during cleaning).

This is the number that goes in the report as the actual token count against
the ~500M target - not min(placeholder ratio * chars).
"""

import argparse
import collections
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sentencepiece as spm

sys.stdout.reconfigure(encoding="utf-8")  # see analyze_tokenizer.py for why

REPO_ROOT = Path(__file__).resolve().parents[1]
READ_WORKERS = 32


def _read_or_none(path: Path):
    """WHY: factored to a plain function (not a lambda) so it's mappable by
    ThreadPoolExecutor.map in count_split; tolerates a missing file instead
    of raising, since a manifest entry can in principle outlive its text
    file (e.g. mid-pipeline changes) and that shouldn't crash the count.
    INPUT: path to a split text file.
    OUTPUT: its contents as a string, or None if the file doesn't exist."""
    return path.read_text(encoding="utf-8") if path.exists() else None


def count_split(sp: spm.SentencePieceProcessor, splits_dir: Path, split: str) -> dict:
    """Encodes every document in this split with the real trained tokenizer
    and sums exact token counts - no sampling. File reads are thread-pooled
    for the same reason as train_tokenizer.py's build_merged_corpus(): with
    ~500k+ individual small files, serial open/read is the dominant cost,
    not the (fast, C++) SentencePiece encoding itself."""
    manifest_path = splits_dir / f"{split}_manifest.json"
    if not manifest_path.exists():
        return {"docs": 0, "chars": 0, "tokens": 0, "manual_tokens": 0, "downloaded_tokens": 0}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    text_dir = splits_dir / split / "text"
    paths = [text_dir / f"{meta['document_id']}.txt" for meta in manifest]

    docs = 0
    chars = 0
    tokens = 0
    tokens_by_source = collections.Counter()

    with ThreadPoolExecutor(READ_WORKERS) as pool:
        texts = pool.map(_read_or_none, paths, chunksize=64)
        for meta, text in zip(manifest, texts):
            if text is None:
                continue
            n_tokens = len(sp.encode(text, out_type=int))
            docs += 1
            chars += len(text)
            tokens += n_tokens
            tokens_by_source[meta.get("source_type", "unknown")] += n_tokens

    return {
        "docs": docs,
        "chars": chars,
        "tokens": tokens,
        "manual_tokens": tokens_by_source.get("manual", 0),
        "downloaded_tokens": tokens_by_source.get("downloaded", 0),
    }


def main():
    """WHY: CLI entry point, run once per language on the chosen (winning)
    vocab size only - not per sweep candidate, since this full-corpus pass
    is the expensive step (see module docstring); analyze_tokenizer.py's
    validation-sample metrics are what the sweep itself compares.
    WHAT: loads the given tokenizer model, counts every split via
    count_split, prints per-split and total docs/chars/tokens plus the
    manual-vs-downloaded token breakdown and exact whole-corpus chars/token.
    INPUT: CLI args --lang, --tokenizer-model (path to a trained .model).
    OUTPUT: none (prints only)."""
    parser = argparse.ArgumentParser(description="Count real total tokens for a language's full corpus")
    parser.add_argument("--lang", required=True, choices=["hindi", "bhojpuri"])
    parser.add_argument("--tokenizer-model", required=True, help="path to a trained .model file")
    args = parser.parse_args()

    lang_dir = REPO_ROOT / args.lang
    splits_dir = lang_dir / "data" / "splits"
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_model)

    totals = {"docs": 0, "chars": 0, "tokens": 0, "manual_tokens": 0, "downloaded_tokens": 0}
    print(f"=== {args.lang} full-corpus token count ({Path(args.tokenizer_model).name}) ===")
    for split in ["train", "validation", "test"]:
        result = count_split(sp, splits_dir, split)
        print(f"  {split:12s} docs={result['docs']:>8,}  chars={result['chars']:>13,}  tokens={result['tokens']:>13,}")
        for k in totals:
            totals[k] += result[k]

    print(f"  {'TOTAL':12s} docs={totals['docs']:>8,}  chars={totals['chars']:>13,}  tokens={totals['tokens']:>13,}")
    manual_pct = 100 * totals["manual_tokens"] / totals["tokens"] if totals["tokens"] else 0.0
    print(f"\n  manual tokens:     {totals['manual_tokens']:,}")
    print(f"  downloaded tokens: {totals['downloaded_tokens']:,}")
    print(f"  manual % (by token, not char): {manual_pct:.2f}%")
    print(f"  chars/token (whole corpus, exact): {totals['chars'] / totals['tokens']:.4f}" if totals["tokens"] else "")


if __name__ == "__main__":
    main()

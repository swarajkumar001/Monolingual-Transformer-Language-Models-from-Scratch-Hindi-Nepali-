"""Exact per-source-name token counts for one language's chosen tokenizer.

WHY: scripts/count_tokens.py aggregates only by source_type (manual vs
downloaded) - enough for the 20% floor check, but not enough to answer
"how many tokens did each individual source (e.g. hi_wikipedia_dump vs
patrika_hindi) actually contribute". This script re-encodes the full
corpus once more, grouped by source_name instead, for that finer-grained
question. Deliberately a separate script rather than a flag on
count_tokens.py, so that script's simpler two-bucket output stays the one
validate_phase1.py-adjacent tooling depends on.

WHAT: for every document in train+validation+test, reads its text, encodes
it with the given SentencePiece model, and sums token/char/doc counts per
(source_type, source_name) pair - exact, no sampling, same method as
count_tokens.py.

INPUT: --lang, --tokenizer-model (a trained .model file).
OUTPUT: prints a table to stdout; source-level counts are what the caller
writes into a manifest JSON (this script does not write JSON itself, to
keep the exact-counting logic separate from how any particular manifest
chooses to present it).
"""

import argparse
import collections
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sentencepiece as spm

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]
READ_WORKERS = 32


def _read_or_none(path: Path):
    return path.read_text(encoding="utf-8") if path.exists() else None


def count_by_source(sp: spm.SentencePieceProcessor, splits_dir: Path, split: str, agg: dict) -> None:
    """Encodes every document in this split, accumulating into agg keyed by
    (source_type, source_name). Mutates agg in place so train/validation/test
    contribute to the same running totals."""
    manifest_path = splits_dir / f"{split}_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    text_dir = splits_dir / split / "text"
    paths = [text_dir / f"{meta['document_id']}.txt" for meta in manifest]

    with ThreadPoolExecutor(READ_WORKERS) as pool:
        texts = pool.map(_read_or_none, paths, chunksize=64)
        for meta, text in zip(manifest, texts):
            if text is None:
                continue
            key = (meta.get("source_type", "unknown"), meta.get("source_name", "unknown"))
            n_tokens = len(sp.encode(text, out_type=int))
            bucket = agg[key]
            bucket["docs"] += 1
            bucket["chars"] += len(text)
            bucket["tokens"] += n_tokens


def main():
    parser = argparse.ArgumentParser(description="Exact per-source-name token counts")
    parser.add_argument("--lang", required=True, choices=["hindi", "bhojpuri"])
    parser.add_argument("--tokenizer-model", required=True)
    args = parser.parse_args()

    lang_dir = REPO_ROOT / args.lang
    splits_dir = lang_dir / "data" / "splits"
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_model)

    agg = collections.defaultdict(lambda: {"docs": 0, "chars": 0, "tokens": 0})
    for split in ["train", "validation", "test"]:
        count_by_source(sp, splits_dir, split, agg)

    total_tokens = sum(v["tokens"] for v in agg.values())
    print(f"=== {args.lang} per-source token counts ({Path(args.tokenizer_model).name}) ===")
    for (source_type, source_name), v in sorted(agg.items(), key=lambda kv: -kv[1]["tokens"]):
        pct = 100 * v["tokens"] / total_tokens if total_tokens else 0.0
        print(f"  [{source_type:10s}] {source_name:45s} docs={v['docs']:>8,}  "
              f"chars={v['chars']:>13,}  tokens={v['tokens']:>13,}  ({pct:.2f}%)")
    print(f"\n  TOTAL tokens: {total_tokens:,}")


if __name__ == "__main__":
    main()

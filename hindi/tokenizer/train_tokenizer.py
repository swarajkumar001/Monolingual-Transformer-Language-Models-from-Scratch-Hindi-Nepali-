"""Train a Hindi SentencePiece tokenizer at a given vocab size.

Mirrors bhojpuri/tokenizer/train_tokenizer.py exactly (same logic, Hindi
paths/model-name prefix). Trains only on data/splits/train/ - never
validation or test, so fertility/coverage/UNK-rate measurements on held-out
text (see analyze_tokenizer.py) are a genuine held-out evaluation, not
train-set leakage.

Settings are fixed across the vocab-size sweep (8k/12k/16k/24k/32k) so the
sweep only varies vocab_size:
  - model_type=unigram
  - character_coverage=0.9995
  - byte_fallback=True (structurally makes UNK rate 0.0 - see
    analyze_tokenizer.py's byte_fallback_rate for the metric that actually
    matters instead)
  - split_digits=True
  - normalization_rule_name=identity (no case-folding/width-normalization
    that could alter Devanagari text)
  - input_sentence_size=2_000_000, shuffled sampling
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sentencepiece as spm

sys.stdout.reconfigure(encoding="utf-8")  # see analyze_tokenizer.py for why

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from clean_lib import acquire_single_instance_lock, release_single_instance_lock  # noqa: E402

LANG_DIR = Path(__file__).resolve().parents[1]
TRAIN_TEXT_DIR = LANG_DIR / "data" / "splits" / "train" / "text"
TOKENIZER_DIR = Path(__file__).resolve().parent
MERGED_CORPUS = TOKENIZER_DIR / "_train_corpus.txt"
LOCK_PATH = TOKENIZER_DIR / ".train.lock"
READ_WORKERS = 32


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_merged_corpus() -> Path:
    """Flatten every train-split document into one newline-per-line corpus
    file, which is what SentencePieceTrainer expects as `input`. Cached
    across the vocab-size sweep so this (the slow part, given ~500k+ small
    files - Hindi's cleaned pool is larger than Bhojpuri's) only runs once.

    Reads are thread-pooled: on Windows/NTFS, per-file open/read of ~500k
    small files serially was measured at ~24.5KB/s wall-clock (a ~2.5h ETA
    for this step alone, on the Bhojpuri run) - each read is I/O-wait
    dominated, not CPU-bound, so it releases the GIL and threading gives a
    large real speedup here. Writing stays single-threaded (via
    executor.map, which preserves input order) so the merged corpus is
    still written in a deterministic order."""
    if MERGED_CORPUS.exists():
        return MERGED_CORPUS
    files = sorted(TRAIN_TEXT_DIR.glob("*.txt"))
    if not files:
        raise SystemExit(f"no train text files found at {TRAIN_TEXT_DIR} - run split_hindi.py first")
    with MERGED_CORPUS.open("w", encoding="utf-8") as out, ThreadPoolExecutor(READ_WORKERS) as pool:
        for text in pool.map(_read, files, chunksize=64):
            for line in text.split("\n"):
                line = line.strip()
                if line:
                    out.write(line + "\n")
    print(f"built merged train corpus: {MERGED_CORPUS} ({len(files):,} docs)")
    return MERGED_CORPUS


def main():
    parser = argparse.ArgumentParser(description="Train a Hindi SentencePiece tokenizer")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    acquire_single_instance_lock(LOCK_PATH)
    try:
        corpus = build_merged_corpus()
        model_prefix = str(TOKENIZER_DIR / f"hindi_unigram_{args.vocab_size}")

        spm.SentencePieceTrainer.Train(
            input=str(corpus),
            model_prefix=model_prefix,
            vocab_size=args.vocab_size,
            model_type="unigram",
            character_coverage=0.9995,
            byte_fallback=True,
            split_digits=True,
            normalization_rule_name="identity",
            input_sentence_size=2_000_000,
            shuffle_input_sentence=True,
            num_threads=args.threads,
        )
        print(f"trained {model_prefix}.model / {model_prefix}.vocab")
    finally:
        release_single_instance_lock(LOCK_PATH)


if __name__ == "__main__":
    main()

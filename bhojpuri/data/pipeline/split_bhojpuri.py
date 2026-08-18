"""Split cleaned Bhojpuri documents into train/validation/test, per
report/phase1/phase1_strategy.md section 7.

- Splits at the document level (never sentence/chunk level) to avoid leakage.
- ~96/2/2 train/val/test by document count, seeded for reproducibility.
- Writes both the actual text files (data/splits/{split}/text/) and explicit
  manifests (data/splits/{split}_manifest.json) listing every document's
  metadata, as required by the spec.

Run after clean_bhojpuri.py. Always rebuilds data/splits/ from data/cleaned/
rather than appending, so it is safe to re-run whenever the cleaned corpus
changes (e.g. once manual data is added).
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from clean_lib import acquire_single_instance_lock, release_single_instance_lock  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1]
CLEANED_DIR = DATA_DIR / "cleaned"
SPLITS_DIR = DATA_DIR / "splits"
LOCK_PATH = DATA_DIR / "metadata" / ".split.lock"
LANG = "bhojpuri"

SPLIT_RATIOS = {"train": 0.96, "validation": 0.02, "test": 0.02}
DEFAULT_SEED = 42


def link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink src to dst (near-instant, byte-identical, same filesystem
    only) instead of shutil.copyfile - copying ~500k individual small files
    one syscall at a time is the dominant cost of this script on Windows/
    NTFS (measured at ~2,500 files/min, i.e. hours for a full split). Falls
    back to a real copy only if hardlinking fails (e.g. src/dst end up on
    different volumes)."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Split cleaned Bhojpuri docs into train/val/test")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    acquire_single_instance_lock(LOCK_PATH)
    try:
        meta_path = CLEANED_DIR / "metadata.jsonl"
        if not meta_path.exists():
            print(f"[{LANG}] No cleaned metadata at {meta_path}; run clean_bhojpuri.py first.")
            return
        docs = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not docs:
            print(f"[{LANG}] No cleaned documents found; run clean_bhojpuri.py first.")
            return

        rng = random.Random(args.seed)
        rng.shuffle(docs)

        n = len(docs)
        n_val = max(1, round(n * SPLIT_RATIOS["validation"]))
        n_test = max(1, round(n * SPLIT_RATIOS["test"]))
        n_train = n - n_val - n_test

        assignment = (
            [("train", d) for d in docs[:n_train]]
            + [("validation", d) for d in docs[n_train : n_train + n_val]]
            + [("test", d) for d in docs[n_train + n_val :]]
        )

        for split in SPLIT_RATIOS:
            text_dir = SPLITS_DIR / split / "text"
            text_dir.mkdir(parents=True, exist_ok=True)
            for old in text_dir.glob("*.txt"):
                old.unlink()

        manifests = {split: [] for split in SPLIT_RATIOS}
        stats = {split: {"docs": 0, "chars": 0} for split in SPLIT_RATIOS}

        for split, meta in assignment:
            doc_id = meta["document_id"]
            src = CLEANED_DIR / "text" / f"{doc_id}.txt"
            if not src.exists():
                continue
            dst = SPLITS_DIR / split / "text" / f"{doc_id}.txt"
            link_or_copy(src, dst)
            split_meta = dict(meta)
            split_meta["split"] = split
            manifests[split].append(split_meta)
            stats[split]["docs"] += 1
            stats[split]["chars"] += split_meta.get("characters", 0)

        for split in SPLIT_RATIOS:
            manifest_path = SPLITS_DIR / f"{split}_manifest.json"
            manifest_path.write_text(json.dumps(manifests[split], indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"[{LANG}] split {n} documents (seed={args.seed}):")
        for split in SPLIT_RATIOS:
            print(f"  {split}: {stats[split]['docs']} docs, {stats[split]['chars']} chars")
    finally:
        release_single_instance_lock(LOCK_PATH)


if __name__ == "__main__":
    main()

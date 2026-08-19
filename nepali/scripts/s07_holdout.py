#!/usr/bin/env python3
"""
Step 07 -- carve out val/test before any quality filtering touches them.

    python hindi/scripts/s07_holdout.py

Despite the file number, this reads S04's output (normalised + language-ID'd,
nothing more) and runs **before** S05/S06 in the actual data flow, not after.
The reason is in the plan: held-out documents must pass through normalisation
and LID only, because if the same heuristic/perplexity filters that select
training data also curate the evaluation set, evaluation partly measures the
filters' own preferences rather than genuine model quality. ``s05_heuristics``
and ``s06_perplexity`` read this step's ``train_pool`` output, never S04's
directly -- that is what keeps val/test unfiltered.

Splitting is by **document**, never by line (a line split leaks context
across train/eval), and is a deterministic hash of ``doc_id`` against the
configured seed -- not a shuffle held in memory -- so the assignment is
reproducible from the seed alone and the whole pool never needs to fit in
RAM at once.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpuslib import runlog, stageio  # noqa: E402
from corpuslib.config import load_language_config  # noqa: E402

STAGE_IN = "s04_langid"
STAGE_TRAIN_POOL = "s07_train_pool"
STAGE_VAL = "s07_val"
STAGE_TEST = "s07_test"


def split_bucket(doc_id: str, seed: int) -> float:
    """Deterministic pseudo-uniform value in [0, 1) for one document."""
    digest = hashlib.blake2b(f"{seed}:{doc_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="process at most N documents")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_language_config()
    config.ensure_dirs()
    log = runlog.StepLogger(config.lang, config.events_path)
    started = time.time()

    stage_in_dir = config.interim_dir / STAGE_IN
    if not stageio.stage_manifest(stage_in_dir):
        print(f"BLOCKED: {stage_in_dir} has no stage manifest. Run s04_langid.py first.")
        log("S07", f"Hold out val/test ({config.display_name})", runlog.BLOCKED,
            "s04_langid has not been run yet.")
        return 1

    splits_cfg = config.section("splits")
    seed = int(splits_cfg.get("seed", 20260817))
    train_ratio = float(splits_cfg.get("train", 0.90))
    val_ratio = float(splits_cfg.get("val", 0.05))
    test_ratio = float(splits_cfg.get("test", 0.05))
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"BLOCKED: split ratios sum to {total_ratio}, not 1.0 -- fix configs/language.yaml")
        log("S07", f"Hold out val/test ({config.display_name})", runlog.BLOCKED,
            f"train+val+test = {total_ratio}, expected 1.0")
        return 1

    print(
        f"S07 holdout ({config.display_name}) -- seed={seed} "
        f"train={train_ratio:.0%} val={val_ratio:.0%} test={test_ratio:.0%}",
        flush=True,
    )

    counts = {"train": 0, "val": 0, "test": 0}
    seen_doc_ids: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    docs_in = 0

    def bucketed(target: str):
        nonlocal docs_in
        for record in stageio.read_stage(stage_in_dir):
            if args.limit and docs_in >= args.limit:
                break
            docs_in_local = docs_in
            docs_in += 1
            bucket = split_bucket(record["doc_id"], seed)
            if bucket < train_ratio:
                name = "train"
            elif bucket < train_ratio + val_ratio:
                name = "val"
            else:
                name = "test"
            counts[name] += 1
            seen_doc_ids[name].add(record["doc_id"])
            if name == target:
                yield record

    # Three independent passes over S04's output rather than three parallel
    # generators sharing one iterator -- simpler, and S04's output is already
    # on disk so re-reading it costs I/O, not re-computation.
    n_train = stageio.write_stage(config.interim_dir / STAGE_TRAIN_POOL, bucketed("train"))
    docs_in = 0
    n_val = stageio.write_stage(config.interim_dir / STAGE_VAL, bucketed("val"))
    docs_in = 0
    n_test = stageio.write_stage(config.interim_dir / STAGE_TEST, bucketed("test"))

    overlap_train_val = seen_doc_ids["train"] & seen_doc_ids["val"]
    overlap_train_test = seen_doc_ids["train"] & seen_doc_ids["test"]
    overlap_val_test = seen_doc_ids["val"] & seen_doc_ids["test"]
    zero_overlap = not (overlap_train_val or overlap_train_test or overlap_val_test)

    total = n_train + n_val + n_test
    elapsed = time.time() - started
    payload = {
        "docs_in": total,
        "train_docs": n_train,
        "val_docs": n_val,
        "test_docs": n_test,
        "train_fraction": round(n_train / total, 4) if total else 0.0,
        "val_fraction": round(n_val / total, 4) if total else 0.0,
        "test_fraction": round(n_test / total, 4) if total else 0.0,
        "target_train": train_ratio,
        "target_val": val_ratio,
        "target_test": test_ratio,
        "zero_doc_id_overlap": zero_overlap,
        "seed": seed,
        "seconds": round(elapsed, 1),
    }
    stageio.write_stage_manifest(config.interim_dir / STAGE_TRAIN_POOL, payload)

    print(f"  train={n_train:,} ({payload['train_fraction']:.2%})  "
          f"val={n_val:,} ({payload['val_fraction']:.2%})  "
          f"test={n_test:,} ({payload['test_fraction']:.2%})")
    print(f"  zero doc_id overlap across splits: {zero_overlap}")

    ok = zero_overlap and total > 0
    log(
        "S07",
        f"Hold out val/test ({config.display_name})",
        runlog.SUCCESS if ok else runlog.BLOCKED,
        (
            f"Split {total:,} documents: train {n_train:,} "
            f"({payload['train_fraction']:.1%}), val {n_val:,} "
            f"({payload['val_fraction']:.1%}), test {n_test:,} "
            f"({payload['test_fraction']:.1%}). Zero doc_id overlap."
            if ok
            else "Split produced overlapping doc_ids or zero documents -- a bug, not a filter."
        ),
        metrics=payload,
    )
    if not ok:
        print("BLOCKED: doc_id overlap detected between splits, or no documents produced.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

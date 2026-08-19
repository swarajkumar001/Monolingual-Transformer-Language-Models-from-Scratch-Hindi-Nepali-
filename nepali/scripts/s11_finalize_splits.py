#!/usr/bin/env python3
"""
Step 11 -- write the final train/val/test files.

    python hindi/scripts/s11_finalize_splits.py

Train comes from S10 (balanced, deduplicated, decontaminated). Val and test
come straight from S07 -- normalised and language-ID'd only, never touched by
quality filtering, dedup, or lane balancing, so they stay a fair, unbiased
sample of what was collected.

**Verify:** manual ratio is reported *per split*, not only overall. A split
whose manual share deviates by more than 10 points from the overall figure is
flagged -- lane balancing only targets the train pool, so val/test could in
principle drift if collection itself were lane-skewed in a way that happened
to concentrate in one split, and that would silently bias evaluation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpuslib import runlog, schema, stageio  # noqa: E402
from corpuslib.config import load_language_config  # noqa: E402
from corpuslib.jsonlio import atomic_write_json, write_jsonl  # noqa: E402

STAGE_TRAIN_IN = "s10_balanced"
STAGE_VAL_IN = "s07_val"
STAGE_TEST_IN = "s07_test"
DEVIATION_FLAG_POINTS = 0.10


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args(argv)


def _split_stats(records: list[dict]) -> dict:
    manual_words = sum(r["n_words"] for r in records if r["lane"] == schema.LANE_MANUAL)
    downloaded_words = sum(r["n_words"] for r in records if r["lane"] == schema.LANE_DOWNLOADED)
    total_words = manual_words + downloaded_words
    return {
        "docs": len(records),
        "manual_words": manual_words,
        "downloaded_words": downloaded_words,
        "total_words": total_words,
        "manual_ratio": round(manual_words / total_words, 4) if total_words else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    config = load_language_config()
    config.ensure_dirs()
    log = runlog.StepLogger(config.lang, config.events_path)
    started = time.time()

    sources = {"train": config.interim_dir / STAGE_TRAIN_IN,
               "val": config.interim_dir / STAGE_VAL_IN,
               "test": config.interim_dir / STAGE_TEST_IN}
    missing = [name for name, d in sources.items()
               if not stageio.stage_manifest(d) and not list(d.glob("part-*.jsonl"))]
    if missing:
        print(f"BLOCKED: missing stage output for {missing}. Run the preceding steps first.")
        log("S11", f"Finalise splits ({config.display_name})", runlog.BLOCKED,
            f"Missing stage output: {missing}")
        return 1

    print(f"S11 finalise splits ({config.display_name})", flush=True)

    all_doc_ids: dict[str, set[str]] = {}
    stats: dict[str, dict] = {}
    for name, stage_dir in sources.items():
        records = list(stageio.read_stage(stage_dir))
        all_doc_ids[name] = {r["doc_id"] for r in records}
        stats[name] = _split_stats(records)
        out_path = config.final_dir / f"{name}.jsonl"
        write_jsonl(out_path, records)
        print(f"  {name}: {stats[name]['docs']:,} docs, "
              f"manual ratio {stats[name]['manual_ratio']:.2%} -> {out_path}", flush=True)

    overlap = {
        "train_val": sorted(all_doc_ids["train"] & all_doc_ids["val"])[:5],
        "train_test": sorted(all_doc_ids["train"] & all_doc_ids["test"])[:5],
        "val_test": sorted(all_doc_ids["val"] & all_doc_ids["test"])[:5],
    }
    zero_overlap = not any(overlap.values())

    overall_words = sum(s["total_words"] for s in stats.values())
    overall_manual = sum(s["manual_words"] for s in stats.values())
    overall_ratio = overall_manual / overall_words if overall_words else 0.0

    flagged = [
        name for name, s in stats.items()
        if s["total_words"] and abs(s["manual_ratio"] - overall_ratio) > DEVIATION_FLAG_POINTS
    ]

    elapsed = time.time() - started
    payload = {
        "splits": stats,
        "overall_manual_ratio": round(overall_ratio, 4),
        "zero_doc_id_overlap": zero_overlap,
        "overlap_samples": overlap,
        "flagged_splits": flagged,
        "deviation_flag_points": DEVIATION_FLAG_POINTS,
        "seconds": round(elapsed, 1),
    }
    atomic_write_json(config.final_dir / "split_stats.json", payload)

    print(f"  overall manual ratio: {overall_ratio:.2%}")
    print(f"  zero doc_id overlap across splits: {zero_overlap}")
    if flagged:
        print(f"  FLAGGED (>{DEVIATION_FLAG_POINTS:.0%} deviation from overall): {flagged}")

    ok = zero_overlap and all(s["docs"] > 0 for s in stats.values())
    log(
        "S11",
        f"Finalise splits ({config.display_name})",
        runlog.SUCCESS if ok else runlog.BLOCKED,
        (
            f"Wrote train/val/test ({stats['train']['docs']:,}/{stats['val']['docs']:,}/"
            f"{stats['test']['docs']:,} docs). Overall manual ratio {overall_ratio:.2%}."
            + (f" Flagged for deviation: {flagged}." if flagged else "")
            if ok
            else "Non-empty split or doc_id overlap violated -- a bug, not a filter."
        ),
        metrics=payload,
    )
    if not ok:
        print("BLOCKED: empty split, or doc_id overlap detected between splits.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

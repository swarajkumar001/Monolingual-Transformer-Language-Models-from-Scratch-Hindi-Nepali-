#!/usr/bin/env python3
"""
Step 08 -- deduplicate the train pool, manual-priority on every collision.

    python hindi/scripts/s08_dedup.py
    python hindi/scripts/s08_dedup.py --limit 50000     # smoke test

Exact ``doc_id`` matches first, then MinHash + LSH (100 permutations, b=20
bands of r=5 rows, threshold ~0.55 -- see ``corpuslib.dedup``). Runs on S06's
output: the train pool only, after quality filtering. val/test (S07) are
never touched here -- decontamination (S09) is the mechanism that keeps train
from leaking into them, not this step.

**The rule that matters:** cross-lane collisions resolve in favour of the
manual copy. Sangraha and IndicCorp are web-crawl derived and do contain
Wikipedia and news text our own manual sources also collect -- that overlap
is expected and was measured with MinHash before this pipeline existed. A
lane-blind dedup pass would delete manual documents at random and erode the
20% requirement, which is exactly how the previous run's manual share
evaporated. ``cross_lane_duplicate_pairs`` and
``manual_docs_preserved_by_priority`` in the report below are the direct,
quantified answer to that.

Both dedup passes need the full candidate set in memory at once -- the LSH
bucket index has to see every document before it can tell which are
duplicates of which, so there is no streaming form of this step. That is a
property of ``corpuslib.dedup`` (already covered by test_corpuslib.py), not
new here.

MinHash signature computation (the expensive part -- 100 permutations per
document) runs across CPU cores via ``--workers`` (default: cpu_count-1); the
LSH bucket insertion that depends on manual-first ordering stays
single-threaded, since that is what makes the priority rule deterministic.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpuslib import dedup, runlog, stageio  # noqa: E402
from corpuslib.config import load_language_config, load_yaml  # noqa: E402

STAGE_IN = "s06_perplexity"
STAGE_OUT = "s08_dedup"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="process at most N documents")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="CPU workers for MinHash signature computation (default: cpu_count-1)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_language_config()
    config.ensure_dirs()
    log = runlog.StepLogger(config.lang, config.events_path)
    started = time.time()

    stage_in_dir = config.interim_dir / STAGE_IN
    if not stageio.stage_manifest(stage_in_dir):
        print(f"BLOCKED: {stage_in_dir} has no stage manifest. Run s06_perplexity.py first.")
        log("S08", f"Deduplicate ({config.display_name})", runlog.BLOCKED,
            "s06_perplexity has not been run yet.")
        return 1

    cleaning_cfg = load_yaml(config.configs_dir / "cleaning.yaml")
    dcfg = cleaning_cfg.get("dedup", {}) or {}
    num_perm = int(dcfg.get("num_perm", dedup.DEFAULT_NUM_PERM))
    bands = int(dcfg.get("bands", dedup.DEFAULT_BANDS))
    rows = int(dcfg.get("rows", dedup.DEFAULT_ROWS))
    shingle_size = int(dcfg.get("shingle_size", dedup.DEFAULT_SHINGLE))
    threshold = float(dcfg.get("threshold", 0.55))

    print(
        f"S08 dedup ({config.display_name}) -- num_perm={num_perm} "
        f"bands={bands} rows={rows} threshold={threshold}",
        flush=True,
    )

    records = list(stageio.read_stage(stage_in_dir))
    if args.limit:
        records = records[: args.limit]
    print(f"  loaded {len(records):,} documents", flush=True)

    stats = dedup.DedupStats()
    after_exact = dedup.remove_exact(records, stats)
    print(f"  exact dedup: {stats.exact_removed:,} removed, {len(after_exact):,} remain", flush=True)

    def progress(position: int, kept: int) -> None:
        print(f"  ...LSH pass at {position:,} documents, {kept:,} kept so far", flush=True)

    after_near = dedup.remove_near_duplicates(
        after_exact,
        num_perm=num_perm,
        bands=bands,
        rows=rows,
        shingle_size=shingle_size,
        threshold=threshold,
        stats=stats,
        on_progress=progress,
        workers=args.workers,
    )
    print(f"  near dedup: {stats.near_removed:,} removed, {len(after_near):,} remain", flush=True)

    written = stageio.write_stage(config.interim_dir / STAGE_OUT, after_near)
    elapsed = time.time() - started

    report = stats.to_dict()
    report["seconds"] = round(elapsed, 1)
    report["config"] = {
        "num_perm": num_perm, "bands": bands, "rows": rows,
        "shingle_size": shingle_size, "threshold": threshold,
    }
    stageio.write_stage_manifest(config.interim_dir / STAGE_OUT, report)

    print(
        f"  input={stats.input_docs:,}  kept={written:,}  retention={report['retention']:.2%}\n"
        f"  cross_lane_duplicate_pairs={stats.cross_lane_pairs:,}  "
        f"manual_docs_preserved_by_priority={stats.manual_saved_by_priority:,}\n"
        f"  removed_by_lane={dict(stats.removed_by_lane)}"
    )

    log(
        "S08",
        f"Deduplicate ({config.display_name})",
        runlog.SUCCESS,
        (
            f"Removed {report['total_removed']:,} of {stats.input_docs:,} documents "
            f"({stats.exact_removed:,} exact, {stats.near_removed:,} near). "
            f"{stats.cross_lane_pairs:,} cross-lane duplicate pairs found; "
            f"{stats.manual_saved_by_priority:,} resolved to the manual copy."
        ),
        metrics=report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Step 09 -- remove train documents that leak into val or test.

    python hindi/scripts/s09_decontaminate.py
    python hindi/scripts/s09_decontaminate.py --limit 50000

Same MinHash machinery as S08, run between the deduplicated train pool and
the union of val+test (both from S07, untouched since). A paraphrased test
document leaks nearly as much signal as an exact copy, so exact ``doc_id``
comparison alone is not enough -- this is the near-duplicate check, applied
in the other direction from S08: here it only ever removes from **train**,
never from val/test, because the held-out sets are the fixed reference an
exact-copy training document would be leaking from.

**Verify:** a final rescan reports zero above-threshold train/holdout
matches. If it does not, that is this step failing to do its job, not a
retention number to shrug at.

Holdout signatures are computed once and its LSH index built once, then
queried twice (removal pass, then rescan) -- the original version rebuilt
both the holdout index and every train signature from scratch for the
rescan, which was pure repeated work: the rescan only needs to know whether
the *already-computed* kept-train signatures still collide with the *same*
holdout index. Train-signature hashing itself is parallelised across CPU cores via
``--workers`` (default: cpu_count-1). Together, eliminating the duplicate
hashing pass and parallelising what remains should cut this step well below
the ~2h it measured at on the 1.56M-document Nepali run -- not re-benchmarked
end to end, since that run itself takes hours.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpuslib import dedup, runlog, stageio  # noqa: E402
from corpuslib.config import load_language_config, load_yaml  # noqa: E402

STAGE_TRAIN_IN = "s08_dedup"
STAGE_VAL = "s07_val"
STAGE_TEST = "s07_test"
STAGE_OUT = "s09_decontaminated"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="process at most N train documents")
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

    train_dir = config.interim_dir / STAGE_TRAIN_IN
    val_dir = config.interim_dir / STAGE_VAL
    test_dir = config.interim_dir / STAGE_TEST
    if not stageio.stage_manifest(train_dir):
        print(f"BLOCKED: {train_dir} has no stage manifest. Run s08_dedup.py first.")
        log("S09", f"Decontaminate ({config.display_name})", runlog.BLOCKED,
            "s08_dedup has not been run yet.")
        return 1

    cleaning_cfg = load_yaml(config.configs_dir / "cleaning.yaml")
    dccfg = cleaning_cfg.get("decontamination", {}) or {}
    if not dccfg.get("enabled", True):
        written = stageio.write_stage(
            config.interim_dir / STAGE_OUT, stageio.read_stage(train_dir)
        )
        stageio.write_stage_manifest(
            config.interim_dir / STAGE_OUT, {"applied": False, "docs_out": written}
        )
        log("S09", f"Decontaminate ({config.display_name})", runlog.FALLBACK,
            "Disabled in config; train pool passed through unchanged.",
            metrics={"docs_out": written}, fallback="disabled_in_config")
        return 0

    threshold = float(dccfg.get("threshold", 0.55))
    dedup_cfg = cleaning_cfg.get("dedup", {}) or {}
    num_perm = int(dedup_cfg.get("num_perm", dedup.DEFAULT_NUM_PERM))
    bands = int(dedup_cfg.get("bands", dedup.DEFAULT_BANDS))
    rows = int(dedup_cfg.get("rows", dedup.DEFAULT_ROWS))
    shingle_size = int(dedup_cfg.get("shingle_size", dedup.DEFAULT_SHINGLE))

    print(f"S09 decontaminate ({config.display_name}) -- threshold={threshold}", flush=True)

    train = list(stageio.read_stage(train_dir))
    if args.limit:
        train = train[: args.limit]
    holdout = list(stageio.read_stage(val_dir)) + list(stageio.read_stage(test_dir))
    print(f"  train={len(train):,}  holdout(val+test)={len(holdout):,}", flush=True)

    # Holdout side: hash once, index once -- reused for both the removal pass
    # and the rescan below.
    holdout_signatures = dedup.compute_signatures(
        [r["text"] for r in holdout],
        num_perm=num_perm, shingle_size=shingle_size, workers=args.workers,
    )
    holdout_buckets = dedup.build_lsh_buckets(holdout_signatures, bands=bands, rows=rows)

    train_signatures = dedup.compute_signatures(
        [r["text"] for r in train],
        num_perm=num_perm, shingle_size=shingle_size, workers=args.workers,
    )
    hits = dedup.query_matches(
        train_signatures, holdout_signatures, holdout_buckets,
        bands=bands, rows=rows, threshold=threshold,
    )
    contaminated_ids = {train[i]["doc_id"] for i, hit in enumerate(hits) if hit is not None}
    kept = [r for r, hit in zip(train, hits) if hit is None]
    kept_signatures = [s for s, hit in zip(train_signatures, hits) if hit is None]
    written = stageio.write_stage(config.interim_dir / STAGE_OUT, kept)

    # Final rescan: zero above-threshold matches should remain between what
    # was kept and the holdout sets. Reuses kept_signatures/holdout_buckets --
    # nothing here needs re-hashing, since neither side's text changed.
    rescan_hits = dedup.query_matches(
        kept_signatures, holdout_signatures, holdout_buckets,
        bands=bands, rows=rows, threshold=threshold,
    )
    rescan_ids = {kept[i]["doc_id"] for i, hit in enumerate(rescan_hits) if hit is not None}

    elapsed = time.time() - started
    payload = {
        "applied": True,
        "train_docs_in": len(train),
        "holdout_docs": len(holdout),
        "contaminated_removed": len(contaminated_ids),
        "train_docs_out": written,
        "rescan_remaining_matches": len(rescan_ids),
        "threshold": threshold,
        "seconds": round(elapsed, 1),
    }
    stageio.write_stage_manifest(config.interim_dir / STAGE_OUT, payload)

    print(f"  removed {len(contaminated_ids):,} contaminated train documents")
    print(f"  final rescan above-threshold matches: {len(rescan_ids)}")

    ok = len(rescan_ids) == 0
    log(
        "S09",
        f"Decontaminate ({config.display_name})",
        runlog.SUCCESS if ok else runlog.BLOCKED,
        (
            f"Removed {len(contaminated_ids):,} train documents matching val/test "
            f"above {threshold}. Final rescan: 0 matches remain."
            if ok
            else (
                f"Final rescan still finds {len(rescan_ids)} above-threshold matches "
                f"after removal -- decontamination did not fully succeed."
            )
        ),
        metrics=payload,
    )
    if not ok:
        print(f"BLOCKED: {len(rescan_ids)} contaminating matches remain after removal.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Step 05 -- heuristic quality filters.

    python hindi/scripts/s05_heuristics.py
    python hindi/scripts/s05_heuristics.py --smoke

Length, word count, symbol ratio, Devanagari ratio, repeated-line ratio,
mean word length, sentence terminators, URL density and lexical diversity --
every threshold read from ``configs/cleaning.yaml: heuristics`` and applied
as a fixed gate (see ``corpuslib.heuristics``). Hindi runs an aggressive
posture because it has surplus data; Nepali runs a conservative one because
every dropped document there is unrecoverable -- the two config files differ,
this script does not.

Writes 20 kept and 20 dropped documents per drop reason to
``data/interim/filter_samples/`` so the choice is reviewable against real
text rather than trusted on the retention number alone.

Reads S07's ``train_pool`` output, not S04's directly -- despite the file
numbers, S07 must fork val/test out of the pool *before* this step and
s06_perplexity ever touch it (see s07_holdout.py's docstring), so those
splits never carry the quality filters' fingerprint.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpuslib import heuristics, runlog, stageio  # noqa: E402
from corpuslib.config import load_language_config, load_yaml  # noqa: E402

STAGE_IN = "s07_train_pool"
STAGE_OUT = "s05_heuristics"
SAMPLES_DIR_NAME = "filter_samples"
SAMPLE_CAP_PER_REASON = 20


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="process 5,000 documents")
    parser.add_argument("--limit", type=int, default=0, help="process at most N documents")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_language_config()
    config.ensure_dirs()
    log = runlog.StepLogger(config.lang, config.events_path)
    started = time.time()
    limit = args.limit or (5000 if args.smoke else 0)

    stage_in_dir = config.interim_dir / STAGE_IN
    if not stageio.stage_manifest(stage_in_dir):
        print(f"BLOCKED: {stage_in_dir} has no stage manifest. Run s07_holdout.py first.")
        log("S05", f"Heuristic filters ({config.display_name})", runlog.BLOCKED,
            "s07_holdout (train_pool) has not been run yet.")
        return 1

    cleaning_cfg = load_yaml(config.configs_dir / "cleaning.yaml")
    heur_cfg = cleaning_cfg.get("heuristics", {}) or {}
    print(f"S05 heuristics ({config.display_name}) -- thresholds: {heur_cfg}", flush=True)

    docs_in = 0
    kept_count = 0
    drop_reasons: Counter[str] = Counter()
    samples_kept: dict[str, list[dict]] = {}
    samples_dropped: dict[str, list[dict]] = {}

    def records():
        nonlocal docs_in, kept_count
        for record in stageio.read_stage(stage_in_dir):
            if limit and docs_in >= limit:
                break
            docs_in += 1
            verdict = heuristics.evaluate(record.get("text", ""), heur_cfg)

            if verdict.passed:
                kept_count += 1
                record["meta"] = dict(record.get("meta") or {})
                record["meta"]["heuristic_stats"] = verdict.stats
                bucket = samples_kept.setdefault("accepted", [])
                if len(bucket) < SAMPLE_CAP_PER_REASON:
                    bucket.append(record)
                if docs_in % 200_000 == 0:
                    print(f"  ...{docs_in:,} evaluated, {kept_count:,} kept", flush=True)
                yield record
            else:
                drop_reasons[verdict.reason] += 1
                bucket = samples_dropped.setdefault(verdict.reason, [])
                if len(bucket) < SAMPLE_CAP_PER_REASON:
                    bucket.append(record)
                if docs_in % 200_000 == 0:
                    print(f"  ...{docs_in:,} evaluated, {kept_count:,} kept", flush=True)

    written = stageio.write_stage(config.interim_dir / STAGE_OUT, records())

    samples_root = config.interim_dir / SAMPLES_DIR_NAME
    for reason in set(samples_dropped) | {"accepted"}:
        stageio.write_samples(
            samples_root, reason,
            samples_kept.get("accepted", []), samples_dropped.get(reason, []),
            n=SAMPLE_CAP_PER_REASON,
        )

    elapsed = time.time() - started
    retention = written / docs_in if docs_in else 0.0
    ret_min = float(heur_cfg.get("expected_retention_min", 0.0))
    ret_max = float(heur_cfg.get("expected_retention_max", 1.0))
    in_band = ret_min <= retention <= ret_max

    payload = {
        "docs_in": docs_in,
        "docs_kept": written,
        "retention": round(retention, 4),
        "expected_retention_min": ret_min,
        "expected_retention_max": ret_max,
        "drop_reasons": dict(drop_reasons),
        "seconds": round(elapsed, 1),
        "smoke": bool(args.smoke),
    }
    stageio.write_stage_manifest(config.interim_dir / STAGE_OUT, payload)

    print(f"  docs_in={docs_in:,}  kept={written:,}  retention={retention:.2%}")
    print(f"  drop reasons: {dict(drop_reasons)}")
    print(f"  samples written to {samples_root}")

    should_abort = (not in_band) and not args.smoke
    status = runlog.SUCCESS if in_band else (runlog.WARNING if args.smoke else runlog.BLOCKED)
    log(
        "S05",
        f"Heuristic filters ({config.display_name})",
        status,
        (
            f"Retention {retention:.2%} of {docs_in:,} documents, inside the "
            f"expected [{ret_min:.0%}, {ret_max:.0%}] band."
            if in_band
            else (
                f"Retention {retention:.2%} is outside the expected "
                f"[{ret_min:.0%}, {ret_max:.0%}] band -- read {samples_root} "
                f"before trusting this run."
            )
        ),
        metrics=payload,
    )
    if should_abort:
        print(
            f"BLOCKED: retention {retention:.2%} outside expected "
            f"[{ret_min:.0%}, {ret_max:.0%}]. Read {samples_root} before proceeding."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Step 04 -- language identification over every document, including manual sources.

    python hindi/scripts/s04_langid.py
    python hindi/scripts/s04_langid.py --smoke     # 5,000 docs, inspect samples first

fastText ``lid.176`` scores every document -- manual sources are not exempt.
Trusting a source by its origin (the previous run's fallback when LID broke)
is what let a NumPy incompatibility delete 94% of the corpus in silence for
four hours: everything was "accepted" by construction because the classifier
never ran.

The decision uses the *margin* between the target label and the best
competing Devanagari label (hi/ne/mai/bho/mr/sa/dty/...), not raw confidence
-- see ``corpuslib.langid`` for why raw confidence alone does not separate
Hindi/Nepali/Maithili reliably. Documents shorter than ``min_chars`` are
never scored.

This step **aborts** (nonzero exit, BLOCKED in the log) if retention falls
outside ``langid.expected_retention_{min,max}`` from ``configs/language.yaml``,
rather than logging SUCCESS on a run that quietly deleted most of the corpus.
``--smoke`` skips the abort (a 5,000-document sample is not a reliable
retention estimate) so samples can be inspected before committing to a full
run.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpuslib import runlog, stageio  # noqa: E402
from corpuslib.config import load_language_config  # noqa: E402
from corpuslib.langid import LanguageIdentifier, LidErrorRate, LidUnavailable  # noqa: E402

STAGE_IN = "s03_normalized"
STAGE_OUT = "s04_langid"
SAMPLES_DIR_NAME = "langid_samples"
SAMPLE_CAP_PER_REASON = 20


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="process 5,000 documents and skip the retention-band abort",
    )
    parser.add_argument("--limit", type=int, default=0, help="process at most N documents")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_language_config()
    config.ensure_dirs()
    log = runlog.StepLogger(config.lang, config.events_path)
    started = time.time()

    limit = args.limit or (5000 if args.smoke else 0)
    lid_cfg = config.langid

    stage_in_dir = config.interim_dir / STAGE_IN
    if not stageio.stage_manifest(stage_in_dir):
        print(f"BLOCKED: {stage_in_dir} has no stage manifest. Run s03_normalize.py first.")
        log("S04", f"Language ID ({config.display_name})", runlog.BLOCKED,
            "s03_normalized has not been run yet.")
        return 1

    print(f"S04 langid ({config.display_name}) -- loading fastText lid.176...", flush=True)
    try:
        identifier = LanguageIdentifier(
            target_label=lid_cfg.get("target_label", config.lang_code),
            min_confidence=float(lid_cfg.get("min_confidence", 0.65)),
            min_margin=float(lid_cfg.get("min_margin", 0.15)),
            min_chars=int(lid_cfg.get("min_chars", 120)),
            min_devanagari_ratio=float(lid_cfg.get("min_devanagari_ratio", 0.55)),
            max_error_rate=float(lid_cfg.get("max_error_rate", 0.01)),
        )
    except LidUnavailable as exc:
        print(f"BLOCKED: LID model unavailable: {exc}")
        log("S04", f"Language ID ({config.display_name})", runlog.BLOCKED, str(exc))
        return 1
    print("  self-test passed.", flush=True)

    docs_in = 0
    kept_count = 0
    reject_reasons: Counter[str] = Counter()
    # A handful of kept/dropped examples per reason, for the reviewable samples.
    samples_kept: dict[str, list[dict]] = {}
    samples_dropped: dict[str, list[dict]] = {}

    def bucket_reason(reason: str) -> str:
        # Collapse parametrised reasons ("narrow_margin_vs_hi:0.12") to their
        # family so sample files don't fragment into hundreds of near-empty ones.
        return reason.split(":", 1)[0]

    def records():
        nonlocal docs_in, kept_count
        try:
            for record in stageio.read_stage(stage_in_dir):
                if limit and docs_in >= limit:
                    break
                docs_in += 1
                verdict = identifier.classify(record.get("text", ""))
                reason_key = bucket_reason(verdict.reason)

                if verdict.accepted:
                    kept_count += 1
                    record["meta"] = dict(record.get("meta") or {})
                    record["meta"].update(verdict.as_meta())
                    bucket = samples_kept.setdefault(reason_key, [])
                    if len(bucket) < SAMPLE_CAP_PER_REASON:
                        bucket.append(record)
                    if docs_in % 200_000 == 0:
                        print(f"  ...{docs_in:,} scored, {kept_count:,} kept", flush=True)
                    yield record
                else:
                    reject_reasons[reason_key] += 1
                    bucket = samples_dropped.setdefault(reason_key, [])
                    if len(bucket) < SAMPLE_CAP_PER_REASON:
                        bucket.append(record)
                    if docs_in % 200_000 == 0:
                        print(f"  ...{docs_in:,} scored, {kept_count:,} kept", flush=True)
        except LidErrorRate as exc:
            print(f"BLOCKED mid-run: {exc}", file=sys.stderr)
            raise

    try:
        written = stageio.write_stage(config.interim_dir / STAGE_OUT, records())
    except LidErrorRate as exc:
        log("S04", f"Language ID ({config.display_name})", runlog.BLOCKED, str(exc),
            metrics={"docs_scored": docs_in, "docs_kept": kept_count})
        return 1

    samples_root = config.interim_dir / SAMPLES_DIR_NAME
    for reason_key in set(samples_kept) | set(samples_dropped):
        stageio.write_samples(
            samples_root, reason_key,
            samples_kept.get(reason_key, []), samples_dropped.get(reason_key, []),
            n=SAMPLE_CAP_PER_REASON,
        )

    elapsed = time.time() - started
    retention = written / docs_in if docs_in else 0.0
    ret_min = float(lid_cfg.get("expected_retention_min", 0.0))
    ret_max = float(lid_cfg.get("expected_retention_max", 1.0))
    in_band = ret_min <= retention <= ret_max

    payload = {
        "docs_in": docs_in,
        "docs_kept": written,
        "retention": round(retention, 4),
        "expected_retention_min": ret_min,
        "expected_retention_max": ret_max,
        "reject_reasons": dict(reject_reasons),
        "lid_error_rate": round(identifier.error_rate, 5),
        "lid_docs_scored": identifier.n_scored,
        "lid_docs_errored": identifier.n_errors,
        "seconds": round(elapsed, 1),
        "smoke": bool(args.smoke),
    }
    stageio.write_stage_manifest(config.interim_dir / STAGE_OUT, payload)

    print(f"  docs_in={docs_in:,}  kept={written:,}  retention={retention:.2%}")
    print(f"  reject reasons: {dict(reject_reasons)}")
    print(f"  LID error rate: {identifier.error_rate:.3%} (halt threshold {identifier.max_error_rate:.1%})")
    print(f"  samples written to {samples_root}")

    should_abort = (not in_band) and not args.smoke
    status = runlog.SUCCESS if in_band else (runlog.WARNING if args.smoke else runlog.BLOCKED)
    log(
        "S04",
        f"Language ID ({config.display_name})",
        status,
        (
            f"Retention {retention:.2%} of {docs_in:,} documents, inside the "
            f"expected [{ret_min:.0%}, {ret_max:.0%}] band."
            if in_band
            else (
                f"Retention {retention:.2%} is outside the expected "
                f"[{ret_min:.0%}, {ret_max:.0%}] band -- inspect "
                f"{samples_root} before trusting this run."
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

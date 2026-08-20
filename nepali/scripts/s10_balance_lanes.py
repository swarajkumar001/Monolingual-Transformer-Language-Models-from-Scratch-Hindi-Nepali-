#!/usr/bin/env python3
"""
Step 10 -- trim the downloaded lane to its target token count; never touch manual.

    python hindi/scripts/s10_balance_lanes.py

Earlier version of this step shrank the downloaded lane until the manual
*ratio* hit 24% -- which meant a starved manual lane (Nepali: 5.82% before
balancing) forced away almost all of a large, perfectly usable downloaded
lane (209.8M words cut to 41.1M) just to hold a percentage. That treats
composition and volume as the same problem; they are not. This step, and the
whole pipeline, works off two independent absolute targets instead of a
ratio: ``targets.manual_tokens_target`` and ``targets.downloaded_tokens_target``.

This step:

1. Keeps 100% of the manual lane, unconditionally -- it is the scarce
   resource and is never subsampled here.
2. Trims the **downloaded** lane down to approximately
   ``targets.downloaded_tokens_target`` (default 400M tokens), converted to a
   word budget via the measured tokens/word factor from
   ``configs/token_calibration.json`` (S02), falling back to
   ``targets.assumed_tokens_per_word`` if S02 has not run.
3. Reports whether manual tokens reached ``targets.manual_tokens_target``. If
   not, that is a WARNING pointing at collection -- add or deepen manual
   sources in ``configs/sources.yaml`` and re-run ``s01_collect.py --lane
   manual --resume`` -- not a reason to discard more downloaded text than the
   target calls for. ``report/qa_gate.py`` already treats a documented
   shortfall as WARN rather than FAIL.

**Verify:** ``balance_report.json`` shows ``downloaded_units_after`` at or
near the configured target, and ``manual_units`` unchanged from the input.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpuslib import runlog, stageio  # noqa: E402
from corpuslib.config import load_language_config, tokens_per_word  # noqa: E402
from corpuslib.ledger import cap_downloaded_lane  # noqa: E402

STAGE_IN = "s09_decontaminated"
STAGE_OUT = "s10_balanced"


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
        print(f"BLOCKED: {stage_in_dir} has no stage manifest. Run s09_decontaminate.py first.")
        log("S10", f"Balance lanes ({config.display_name})", runlog.BLOCKED,
            "s09_decontaminated has not been run yet.")
        return 1

    downloaded_tokens_target = config.downloaded_tokens_target
    tpw = tokens_per_word(config)
    downloaded_word_cap = int(downloaded_tokens_target / tpw)

    records = list(stageio.read_stage(stage_in_dir))
    if args.limit:
        records = records[: args.limit]
    print(
        f"S10 trim downloaded lane ({config.display_name}) -- {len(records):,} documents, "
        f"downloaded target {downloaded_tokens_target:,} tokens "
        f"(~{downloaded_word_cap:,} words @ {tpw:.4f} tok/word)",
        flush=True,
    )

    kept, report = cap_downloaded_lane(
        records, downloaded_unit_cap=downloaded_word_cap, unit="n_words", seed=int(
            config.section("splits").get("seed", 20260817)
        ),
    )
    written = stageio.write_stage(config.interim_dir / STAGE_OUT, kept)

    elapsed = time.time() - started
    report["seconds"] = round(elapsed, 1)
    report["docs_in"] = len(records)
    report["docs_out"] = written
    report["tokens_per_word"] = tpw
    stageio.write_stage_manifest(config.interim_dir / STAGE_OUT, report)
    from corpuslib.jsonlio import atomic_write_json
    atomic_write_json(config.interim_dir / "balance_report.json", report)

    print(f"  action: {report['action']}")
    print(f"  downloaded words: {report['downloaded_units_before']:,} -> "
          f"{report['downloaded_units_after']:,}  (target {downloaded_word_cap:,})")

    manual_tokens_target = config.manual_tokens_target
    manual_words = report["manual_units"]
    manual_tokens_estimate = int(manual_words * tpw)
    meets_manual_target = manual_tokens_estimate >= manual_tokens_target
    status = runlog.SUCCESS if meets_manual_target else runlog.WARNING
    print(f"  manual tokens (estimate): {manual_tokens_estimate:,} "
          f"(target {manual_tokens_target:,})")
    log(
        "S10",
        f"Balance lanes ({config.display_name})",
        status,
        (
            f"{report['reason']} Manual tokens (estimate) {manual_tokens_estimate:,} "
            f"(target {manual_tokens_target:,})."
        ),
        metrics=report,
        decision=report["action"],
    )
    if not meets_manual_target:
        print(
            f"WARNING: manual tokens (estimate) {manual_tokens_estimate:,} is below the "
            f"{manual_tokens_target:,}-token target. The downloaded lane was NOT cut further to "
            f"compensate -- collect more manual text instead: add/deepen sources in "
            f"configs/sources.yaml, then `python {config.lang}/scripts/s01_collect.py --lane "
            f"manual --resume` and re-run S02 onward. This is a documented WARN, not a blocking "
            f"failure -- see report/qa_gate.py."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

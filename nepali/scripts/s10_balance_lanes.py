#!/usr/bin/env python3
"""
Step 10 -- cap the downloaded lane at a fixed token budget; never touch manual.

    python hindi/scripts/s10_balance_lanes.py

Earlier version of this step shrank the downloaded lane until the manual
*ratio* hit 24% -- which meant a starved manual lane (Nepali: 5.82% before
balancing) forced away almost all of a large, perfectly usable downloaded
lane (209.8M words cut to 41.1M) just to hold a percentage. That treats
composition and volume as the same problem; they are not.

This step instead:

1. Keeps 100% of the manual lane, unconditionally -- it is the scarce
   resource and is never subsampled here.
2. Trims the **downloaded** lane to ``targets.downloaded_tokens_cap``
   (default 400M tokens, i.e. ``final_tokens * (1 - manual_ratio_floor)``),
   converted to a word budget via the measured tokens/word factor from
   ``configs/token_calibration.json`` (S02), falling back to
   ``targets.assumed_tokens_per_word`` if S02 has not run.
3. Reports the resulting manual ratio. If it is still below the floor, that
   is a WARNING pointing at collection -- add or deepen manual sources in
   ``configs/sources.yaml`` and re-run ``s01_collect.py --lane manual
   --resume`` -- not a reason to discard more downloaded text than the
   budget calls for. ``report/qa_gate.py`` already treats a documented
   shortfall as WARN rather than FAIL.

**Verify:** ``balance_report.json`` shows ``downloaded_units_after`` at or
under the configured cap, and ``manual_units`` unchanged from the input.
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

    targets = config.targets
    final_tokens = int(targets.get("final_tokens", 500_000_000))
    downloaded_tokens_cap = int(
        targets.get("downloaded_tokens_cap", round(final_tokens * (1 - config.manual_floor)))
    )
    tpw = tokens_per_word(config)
    downloaded_word_cap = int(downloaded_tokens_cap / tpw)

    records = list(stageio.read_stage(stage_in_dir))
    if args.limit:
        records = records[: args.limit]
    print(
        f"S10 cap downloaded lane ({config.display_name}) -- {len(records):,} documents, "
        f"downloaded cap {downloaded_tokens_cap:,} tokens "
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
          f"{report['downloaded_units_after']:,}  (cap {downloaded_word_cap:,})")
    print(f"  manual ratio: {report['ratio_before']:.2%} -> {report['ratio_after']:.2%}")

    meets_floor = report["ratio_after"] >= config.manual_floor
    status = runlog.SUCCESS if meets_floor else runlog.WARNING
    log(
        "S10",
        f"Balance lanes ({config.display_name})",
        status,
        (
            f"{report['reason']} Manual ratio {report['ratio_before']:.2%} -> "
            f"{report['ratio_after']:.2%} (floor {config.manual_floor:.0%})."
        ),
        metrics=report,
        decision=report["action"],
    )
    if not meets_floor:
        print(
            f"WARNING: manual ratio {report['ratio_after']:.2%} is below the {config.manual_floor:.0%} "
            f"floor. The downloaded lane was NOT cut further to force the ratio -- collect more "
            f"manual text instead: add/deepen sources in configs/sources.yaml, then "
            f"`python {config.lang}/scripts/s01_collect.py --lane manual --resume` and re-run "
            f"S02 onward. This is a documented WARN, not a blocking failure -- see report/qa_gate.py."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

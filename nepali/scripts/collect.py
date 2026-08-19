#!/usr/bin/env python3
"""
Collect this language's sources -- always resumes.

    python hindi/scripts/collect.py --lane manual
    python hindi/scripts/collect.py --lane downloaded
    python hindi/scripts/collect.py --lane both
    python hindi/scripts/collect.py --lane manual --only wikipedia_api,wiktionary_api
    python hindi/scripts/collect.py --lane manual --smoke

There is no "fresh" mode here, on purpose. This always runs S01 with
``--resume``, so running it five times in a row costs one real collection
run, not five: a source whose prior run already reached its word/doc target
is skipped immediately, before any network call (see
``corpuslib.sources.base.Collector.run``'s ``already_met`` check), and a
source stopped early on ``time_limit`` continues from its checkpoint instead
of restarting. A genuine from-scratch re-collect is what
``run_pipeline.py clean`` (which wipes and trims deliberately, with a
confirmation prompt) is for -- not something this script should risk
whenever ``--resume`` is forgotten.

Runs S00 (doctor) first unless ``--skip-doctor`` is passed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lane", choices=["manual", "downloaded", "both"], required=True)
    parser.add_argument("--only", default="", help="comma-separated source_ids")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-docs", type=int, default=200)
    parser.add_argument("--workers", type=int, default=6, help="concurrent sources")
    parser.add_argument("--skip-doctor", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.skip_doctor:
        code = subprocess.run([PYTHON, str(SCRIPTS_DIR / "s00_doctor.py")]).returncode
        if code != 0:
            print("doctor reported blocking problems -- fix them, or pass --skip-doctor to proceed anyway")
            return code

    cmd = [
        PYTHON, str(SCRIPTS_DIR / "s01_collect.py"),
        "--lane", args.lane, "--workers", str(args.workers), "--resume",
    ]
    if args.only:
        cmd += ["--only", args.only]
    if args.smoke:
        cmd += ["--smoke", "--smoke-docs", str(args.smoke_docs)]

    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted -- progress is checkpointed; just run this again")
        raise SystemExit(130)

#!/usr/bin/env python3
"""
Single-language driver -- clean disk, or run the S02-S14 pipeline.

    python hindi/scripts/run_pipeline.py clean
    python hindi/scripts/run_pipeline.py pipeline
    python hindi/scripts/run_pipeline.py sync          # hindi only, see below

Collection (S01) is not here -- see ``collect.py`` in this same directory.
It is a separate, much longer-running, network-bound concern with its own
always-resume behaviour, and keeping it out of this driver means running
``pipeline`` again can never accidentally re-trigger a multi-hour collect.

Run the identical command from ``nepali/scripts/run_pipeline.py`` for the
other language. There is no shared driver between them by design -- running
both at once just means opening two terminals, one per language, each
writing to its own ``<lang>/data/`` tree. Nothing here assumes it is the
only thing running: the CPU-heavy stages (S08/S09) already default to a
conservative worker count for exactly that reason (see
``corpuslib.dedup.compute_signatures``); pass ``--workers`` explicitly for
more if you know only one language is running.

Subcommands
-----------
``clean [--yes]``
    Deletes this language's ``data/interim/``, ``data/final/`` and
    ``tokenizer/`` (all regenerable by re-running ``pipeline``), and trims
    the raw **downloaded** lane down to ``targets.downloaded_tokens_cap``
    (default 400M tokens). The raw **manual** lane is never touched -- see
    ``corpuslib.rawtrim`` for why. Prints the exact plan and asks for
    confirmation before deleting anything, unless ``--yes`` is passed.

``pipeline [--from STAGE] [--limit N] [--workers N] [--keep-interim]``
    Runs S02 through S14 in execution order (S07 before S05/S06, despite the
    file number -- see ``s07_holdout.py``), stopping at the first stage that
    exits non-zero. ``--from s08_dedup.py`` resumes from a specific stage.

    After each stage finishes, its manifest is copied into
    ``data/interim/stats/<stage>.json`` (survives everything below), and any
    *earlier* stage directory whose last remaining reader was the stage that
    just finished is deleted -- e.g. ``s03_normalized/`` is removed once S04
    has read it, but ``s07_val/``/``s07_test/`` stick around until S11, since
    S09 and S11 both still need them. This is what keeps a full run from
    piling up ~90GB of redundant full-corpus copies in ``data/interim/`` (one
    per stage) the way it used to. Pass ``--keep-interim`` to disable this and
    keep every stage's output on disk, e.g. while debugging one stage.

``sync``
    Pushes this ``scripts/`` directory to every sibling ``<lang>/scripts/``
    directory next to it. Only meaningful run from ``hindi/scripts/`` --
    hindi is this repo's source of truth for the shared ``corpuslib`` code
    (see ``report/qa_gate.py``'s hash-identity check); running it from
    ``nepali/scripts/`` refuses, since nepali is a copy, not the source.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from corpuslib import rawtrim, stageio  # noqa: E402
from corpuslib.config import load_language_config  # noqa: E402

PYTHON = sys.executable

# Execution order, not file-number order -- S07 before S05/S06.
STAGES = [
    "s02_calibrate.py",
    "s03_normalize.py",
    "s04_langid.py",
    "s07_holdout.py",
    "s05_heuristics.py",
    "s06_perplexity.py",
    "s08_dedup.py",
    "s09_decontaminate.py",
    "s10_balance_lanes.py",
    "s11_finalize_splits.py",
    "s12_train_tokenizer.py",
    "s13_tokenizer_eval.py",
    "s14_corpus_stats.py",
]
_LIMIT_STAGES = {
    "s03_normalize.py", "s04_langid.py", "s05_heuristics.py", "s06_perplexity.py",
    "s07_holdout.py", "s08_dedup.py", "s09_decontaminate.py", "s10_balance_lanes.py",
}
_CPU_HEAVY_STAGES = {"s08_dedup.py", "s09_decontaminate.py"}

# Every interim/<name>/ directory a stage produces.
_STAGE_OUTPUT_DIRS = {
    "s03_normalize.py": ["s03_normalized"],
    "s04_langid.py": ["s04_langid"],
    "s07_holdout.py": ["s07_train_pool", "s07_val", "s07_test"],
    "s05_heuristics.py": ["s05_heuristics"],
    "s06_perplexity.py": ["s06_perplexity"],
    "s08_dedup.py": ["s08_dedup"],
    "s09_decontaminate.py": ["s09_decontaminated"],
    "s10_balance_lanes.py": ["s10_balanced"],
}
# interim/<name>/ -> the LAST stage (by STAGES order) that still reads it.
# Most directories are read once, by the very next stage -- s07_val/s07_test
# are the exception, needed by both S09 (decontaminate) and S11 (assembles
# final val/test straight from them, since only the train pool goes through
# S05-S10), so they outlive several stages a naive "delete stage i-1" would
# have removed them after.
_LAST_CONSUMER = {
    "s03_normalized": "s04_langid.py",
    "s04_langid": "s07_holdout.py",
    "s07_train_pool": "s05_heuristics.py",
    "s07_val": "s11_finalize_splits.py",
    "s07_test": "s11_finalize_splits.py",
    "s05_heuristics": "s06_perplexity.py",
    "s06_perplexity": "s08_dedup.py",
    "s08_dedup": "s09_decontaminate.py",
    "s09_decontaminated": "s10_balance_lanes.py",
    "s10_balanced": "s11_finalize_splits.py",
}


def run(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


def cmd_clean(args: argparse.Namespace) -> int:
    config = load_language_config(root=SCRIPTS_DIR.parent)
    clean_plan = rawtrim.plan(config)
    print(clean_plan.render(config))
    print()
    if not args.yes:
        answer = input("Proceed with deletion? [y/N] ").strip().lower()
        if answer != "y":
            print("aborted -- nothing deleted")
            return 1
    rawtrim.apply(config, clean_plan)
    print("done. Next: `python collect.py --lane manual` to grow manual, "
          "then `run_pipeline.py pipeline` to rebuild interim/final/tokenizer.")
    return 0


def _cleanup_after_stage(interim_dir: Path, stage: str) -> None:
    """Persist this stage's own manifest(s), then delete any earlier stage
    directory whose last reader was this stage."""
    for output_name in _STAGE_OUTPUT_DIRS.get(stage, []):
        stageio.persist_stats(interim_dir, output_name)

    for dir_name, last_consumer in _LAST_CONSUMER.items():
        if last_consumer != stage:
            continue
        path = interim_dir / dir_name
        if path.exists():
            shutil.rmtree(path)
            print(f"  (deleted interim/{dir_name}/ -- fully consumed as of {stage})")


def cmd_pipeline(args: argparse.Namespace) -> int:
    stages = STAGES
    if args.from_stage:
        if args.from_stage not in STAGES:
            print(f"unknown stage {args.from_stage!r}; choose from {STAGES}")
            return 2
        stages = STAGES[STAGES.index(args.from_stage):]

    config = load_language_config(root=SCRIPTS_DIR.parent)

    for stage in stages:
        cmd = [PYTHON, str(SCRIPTS_DIR / stage)]
        if args.limit and stage in _LIMIT_STAGES:
            cmd += ["--limit", str(args.limit)]
        if args.workers and stage in _CPU_HEAVY_STAGES:
            cmd += ["--workers", str(args.workers)]
        code = run(cmd)
        if code != 0:
            print(f"STOPPED at {stage} (exit {code})")
            return code
        if not args.keep_interim:
            _cleanup_after_stage(config.interim_dir, stage)
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    lang_root = SCRIPTS_DIR.parent
    repo_root = lang_root.parent
    if lang_root.name != "hindi":
        print(f"refusing: this is {lang_root.name}/scripts/run_pipeline.py, a synced COPY. "
              f"Run `python hindi/scripts/run_pipeline.py sync` instead -- hindi is the source of truth.")
        return 1

    import hashlib

    def hash_tree(root: Path) -> dict[str, str]:
        return {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts
        }

    reference = hash_tree(SCRIPTS_DIR)
    targets = sorted(
        p / "scripts" for p in repo_root.iterdir()
        if p.is_dir() and p != lang_root and (p / "scripts").is_dir()
    )
    if not targets:
        print("no sibling <lang>/scripts/ directories found next to hindi/ -- nothing to sync")
        return 0
    for target in targets:
        if hash_tree(target) == reference:
            print(f"{target} already matches, skipped")
            continue
        shutil.rmtree(target)
        shutil.copytree(SCRIPTS_DIR, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        print(f"synced -> {target}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_clean = sub.add_parser("clean", help="delete regenerable data + trim downloaded raw to its cap")
    p_clean.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    p_pipeline = sub.add_parser("pipeline", help="S02-S14 for this language")
    p_pipeline.add_argument("--from", dest="from_stage", default=None, help="resume from this stage's filename")
    p_pipeline.add_argument("--limit", type=int, default=0, help="pass --limit N to every stage (smoke test)")
    p_pipeline.add_argument("--workers", type=int, default=0, help="override S08/S09's MinHash worker count")
    p_pipeline.add_argument("--keep-interim", action="store_true", help="disable the rolling per-stage cleanup")

    sub.add_parser("sync", help="hindi only: push scripts/ to sibling language folders")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    handlers = {"clean": cmd_clean, "pipeline": cmd_pipeline, "sync": cmd_sync}
    return handlers[args.command](args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        raise SystemExit(130)

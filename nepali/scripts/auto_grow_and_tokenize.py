#!/usr/bin/env python3
"""
Unattended driver, run from either language's scripts/ directory (path-agnostic).

    python nepali/scripts/auto_grow_and_tokenize.py [--wait-pid PID]

0. One-time only, before Round 1: trims the raw DOWNLOADED lane down to
   approximately ``targets.downloaded_tokens_target`` (corpuslib.rawtrim,
   same mechanism ``run_pipeline.py clean`` uses) and wipes data/interim,
   data/final, tokenizer/ so the rebuild starts from a right-sized corpus
   instead of reprocessing far more raw data than the target needs. This is
   sized off the words->tokens calibration factor alone, which does NOT
   account for S03-S09 cleaning attrition (~60% loss is typical) -- so
   downloaded routinely comes out short after the first round, by design;
   growth (below) is what corrects that, not a bigger initial trim. Raw
   MANUAL is never touched.
1. (optionally) waits for an already-running collect.py process to exit, so
   it never overlaps a collection currently in flight.
2. Runs the full S02-S14 pipeline (rebuilds data/final/{train,val,test},
   trains BPE ONLY at vocab_sizes [5000, 6000, 8000, 10000] per
   configs/tokenizer.yaml -- no other vocab size is trained -- and installs
   whichever of those four has the best fertility). Every round retrains all
   four sizes from scratch on the freshly rebuilt data.
3. Reads data/final/corpus_stats.json for the real, measured totals and
   checks two independent ABSOLUTE targets (no ratio, no overall-total gate):
       manual lane tokens      >= targets.manual_tokens_target  (100,000,000)
       downloaded lane tokens  >= targets.downloaded_tokens_target (400,000,000)
4. Whichever lane(s) are short get grown: raises target_words for that
   lane's still-growable sources in configs/sources.yaml, then re-runs
   `collect.py --lane <manual|downloaded|both>`. collect.py's own
   resume/already-collected-shard tracking is what guarantees the
   re-collection never duplicates a document already on disk -- this holds
   for downloaded too, even after the raw trim in step 0, because trimming
   only deletes shard files and updates the manifest count; each source's own
   resume cursor (_state.json) is untouched, so resuming continues collecting
   NEW material from where the untrimmed run left off rather than re-fetching
   anything that was trimmed away.
5. Repeats from step 2, up to MAX_ROUNDS total pipeline runs. A lane growing
   by less than PLATEAU_PCT between two consecutive rounds is treated as at
   its real ceiling and is not grown again -- filters are never loosened to
   manufacture volume (see README's "Never widen a filter to hit a retention
   number").
6. Writes <lang>/data/growth_log.json (one entry per round) and
   <lang>/data/growth_summary.json (final verdict + honest shortfall
   reasoning if a target was not reached).

Touches ONLY files under this one language directory (configs, data,
tokenizer, scripts' own outputs). Never touches the other language, report/,
or anything at the repo root, and never edits filter thresholds in
cleaning.yaml.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
LANG_DIR = SCRIPTS_DIR.parent
REPO_ROOT = LANG_DIR.parent
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python3")

sys.path.insert(0, str(SCRIPTS_DIR))

SOURCES_YAML = LANG_DIR / "configs" / "sources.yaml"
CORPUS_STATS = LANG_DIR / "data" / "final" / "corpus_stats.json"
GROWTH_LOG = LANG_DIR / "data" / "growth_log.json"
GROWTH_SUMMARY = LANG_DIR / "data" / "growth_summary.json"

MAX_ROUNDS = 4                 # total pipeline runs, including the first
PLATEAU_PCT = 0.05             # <5% growth vs previous round -> treat as ceiling
MIN_FREE_DISK_GB = 15
MAX_TOTAL_SECONDS = 10 * 3600  # hard wall-clock safety net

# Growth ceilings per source -- generous, but bounded, so a runaway loop
# can't fill the disk. If a source hits its own real "source_exhausted"
# before this, that shows up in collection_status.json, not here.
MANUAL_SOURCE_CAPS = {
    "news_onlinekhabar": 260_000_000,  # real pool ~243M words (462k posts)
}
DOWNLOADED_SOURCE_CAPS = {
    "sangraha_verified": 550_000_000,
    "sangraha_unverified": 280_000_000,
    "indiccorp_v2": 200_000_000,
}
GROWTH_FACTOR = 1.6


def log(msg: str) -> None:
    print(f"[auto_grow] {msg}", flush=True)


def run(cmd: list[str]) -> int:
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=str(LANG_DIR)).returncode


def wait_for_pid(pid: int) -> None:
    log(f"waiting for pid {pid} (currently-running collect.py) to exit...")
    while True:
        try:
            import os
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        except PermissionError:
            break
        time.sleep(15)
    log(f"pid {pid} has exited; proceeding.")


def free_disk_gb() -> float:
    usage = shutil.disk_usage(str(LANG_DIR))
    return usage.free / (1024 ** 3)


def trim_raw_downloaded_and_wipe_regenerable() -> None:
    """One-time, before Round 1: size raw downloaded to the target and clear
    interim/final/tokenizer so the rebuild starts from a known-clean state."""
    from corpuslib import rawtrim
    from corpuslib.config import load_language_config

    config = load_language_config(root=LANG_DIR)
    clean_plan = rawtrim.plan(config)
    log("raw downloaded trim + regenerable-dir wipe plan:")
    for line in clean_plan.render(config).splitlines():
        log("  " + line)
    rawtrim.apply(config, clean_plan)
    log("trim applied.")


# --- targeted YAML text patching (preserves every comment) -----------------

def _fmt(n: int) -> str:
    return f"{n:_}"


def patch_source_target(text: str, source_id: str, new_target: int) -> str:
    marker = f"- source_id: {source_id}"
    start = text.index(marker)
    # block runs until the next "  - source_id:" or end of file
    next_idx = text.find("\n  - source_id:", start + len(marker))
    end = next_idx if next_idx != -1 else len(text)
    block = text[start:end]
    new_block, n = re.subn(
        r"(target_words:\s*)[\d_]+",
        lambda m: m.group(1) + _fmt(new_target),
        block,
        count=1,
    )
    if n == 0:
        raise RuntimeError(f"could not find target_words: line for {source_id}")
    return text[:start] + new_block + text[end:]


def get_source_target(text: str, source_id: str) -> int:
    marker = f"- source_id: {source_id}"
    start = text.index(marker)
    next_idx = text.find("\n  - source_id:", start + len(marker))
    end = next_idx if next_idx != -1 else len(text)
    block = text[start:end]
    m = re.search(r"target_words:\s*([\d_]+)", block)
    if not m:
        raise RuntimeError(f"could not find target_words: line for {source_id}")
    return int(m.group(1).replace("_", ""))


def grow_lane(caps: dict[str, int]) -> bool:
    """Bump target_words for each source in ``caps`` by GROWTH_FACTOR, up to
    its ceiling. Returns True if anything was actually changed."""
    text = SOURCES_YAML.read_text("utf-8")
    changed = False
    for source_id, cap in caps.items():
        current = get_source_target(text, source_id)
        new_target = min(int(current * GROWTH_FACTOR), cap)
        if new_target <= current:
            log(f"  {source_id}: already at its growth ceiling ({current:,}), not bumped")
            continue
        text = patch_source_target(text, source_id, new_target)
        log(f"  {source_id}: target_words {current:,} -> {new_target:,}")
        changed = True
    if changed:
        SOURCES_YAML.write_text(text, "utf-8")
    else:
        log("  no growable source left below its ceiling")
    return changed


def read_totals() -> dict:
    d = json.loads(CORPUS_STATS.read_text("utf-8"))
    manual = int(d["overall_manual_tokens"])
    downloaded = int(d.get("overall_downloaded_tokens", d["overall_tokens"] - manual))
    return {"total_tokens": manual + downloaded, "manual_tokens": manual, "downloaded_tokens": downloaded}


def append_growth_log(entry: dict) -> None:
    existing = []
    if GROWTH_LOG.exists():
        try:
            existing = json.loads(GROWTH_LOG.read_text("utf-8"))
        except Exception:
            existing = []
    existing.append(entry)
    GROWTH_LOG.write_text(json.dumps(existing, indent=2, ensure_ascii=False), "utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, default=None)
    parser.add_argument("--skip-raw-trim", action="store_true",
                         help="skip the one-time raw-downloaded trim (already done)")
    parser.add_argument("--resume-collect", choices=["manual", "downloaded", "both"], default=None,
                         help="resume after a previous run's re-collect step failed: skip the raw "
                              "trim and the already-completed round's pipeline run, retry "
                              "`collect.py --lane <this>` (sources.yaml was already bumped by the "
                              "failed run), then continue the normal round loop from growth_log.json")
    args = parser.parse_args()

    started = time.time()
    if args.wait_pid:
        wait_for_pid(args.wait_pid)

    from corpuslib.config import load_language_config
    config = load_language_config(root=LANG_DIR)
    manual_target = config.manual_tokens_target
    downloaded_target = config.downloaded_tokens_target
    log(f"targets: manual >= {manual_target:,} tokens, downloaded >= {downloaded_target:,} tokens "
        f"-- whichever lane falls short gets grown and the pipeline re-run, up to {MAX_ROUNDS} rounds")

    prev_totals: dict | None = None
    plateaued = {"manual": False, "downloaded": False}
    final_reason = ""
    start_round = 1

    if args.resume_collect:
        log(f"===== RESUMING: retrying the failed re-collect step ({args.resume_collect} lane) =====")
        existing = json.loads(GROWTH_LOG.read_text("utf-8")) if GROWTH_LOG.exists() else []
        if not existing:
            log("  no growth_log.json entries found -- nothing to resume from, starting clean instead.")
            args.resume_collect = None
        else:
            last = existing[-1]
            prev_totals = {k: last[k] for k in ("total_tokens", "manual_tokens", "downloaded_tokens")}
            plateaued = {"manual": last["manual_plateaued"], "downloaded": last["downloaded_plateaued"]}
            start_round = last["round"] + 1
            log(f"  resuming from round {start_round} (last completed round {last['round']}: "
                f"manual={last['manual_tokens']:,} downloaded={last['downloaded_tokens']:,})")
            c_code = run([PYTHON, str(SCRIPTS_DIR / "collect.py"), "--lane", args.resume_collect])
            if c_code != 0:
                final_reason = f"collect.py failed again on resume (exit {c_code}); stopping."
                log(final_reason)
                GROWTH_SUMMARY.write_text(json.dumps({
                    "targets": {"manual_tokens_target": manual_target,
                                "downloaded_tokens_target": downloaded_target},
                    "final_totals": prev_totals, "reason": final_reason,
                    "elapsed_seconds": round(time.time() - started, 1),
                }, indent=2, ensure_ascii=False), "utf-8")
                return 1

    if not args.resume_collect:
        log("===== ONE-TIME: trim raw downloaded to target, wipe interim/final/tokenizer =====")
        if not args.skip_raw_trim:
            trim_raw_downloaded_and_wipe_regenerable()
        else:
            log("  --skip-raw-trim passed; not touching raw data.")

    for round_idx in range(start_round, MAX_ROUNDS + 1):
        round_started = time.time()
        log(f"===== ROUND {round_idx}/{MAX_ROUNDS}: running S02-S14 pipeline =====")
        code = run([PYTHON, str(SCRIPTS_DIR / "run_pipeline.py"), "pipeline"])
        if code != 0:
            final_reason = f"pipeline failed at round {round_idx} (exit {code}); stopping."
            log(final_reason)
            break

        totals = read_totals()
        met = {
            "manual": totals["manual_tokens"] >= manual_target,
            "downloaded": totals["downloaded_tokens"] >= downloaded_target,
        }
        log(f"  totals: manual={totals['manual_tokens']:,} (target {manual_target:,}, "
            f"met={met['manual']}) downloaded={totals['downloaded_tokens']:,} "
            f"(target {downloaded_target:,}, met={met['downloaded']})")

        if prev_totals is not None:
            for lane in ("manual", "downloaded"):
                prev_v, new_v = prev_totals[f"{lane}_tokens"], totals[f"{lane}_tokens"]
                growth = (new_v - prev_v) / prev_v if prev_v > 0 else float("inf")
                if growth < PLATEAU_PCT:
                    plateaued[lane] = True
                    log(f"  {lane} lane grew only {growth:.1%} this round -> treating as at its "
                        f"real ceiling, will not grow it further")

        append_growth_log({
            "round": round_idx,
            **totals,
            "met_manual": met["manual"],
            "met_downloaded": met["downloaded"],
            "manual_plateaued": plateaued["manual"],
            "downloaded_plateaued": plateaued["downloaded"],
            "seconds": round(time.time() - round_started, 1),
        })
        prev_totals = totals

        if met["manual"] and met["downloaded"]:
            final_reason = f"both targets met after round {round_idx}."
            log(final_reason)
            break

        if time.time() - started > MAX_TOTAL_SECONDS:
            final_reason = f"hit the {MAX_TOTAL_SECONDS/3600:.0f}h wall-clock safety net; stopping."
            log(final_reason)
            break

        if round_idx == MAX_ROUNDS:
            final_reason = f"reached MAX_ROUNDS={MAX_ROUNDS} without meeting every target."
            log(final_reason)
            break

        want_grow = {lane: (not met[lane]) and (not plateaued[lane]) for lane in ("manual", "downloaded")}
        if not any(want_grow.values()):
            final_reason = "every still-short lane has plateaued at its real ceiling; stopping."
            log(final_reason)
            break

        free_gb = free_disk_gb()
        if free_gb < MIN_FREE_DISK_GB:
            final_reason = f"only {free_gb:.1f}GB free (< {MIN_FREE_DISK_GB}GB floor); stopping growth."
            log(final_reason)
            break

        if want_grow["manual"]:
            log("  growing MANUAL lane for next round")
            grow_lane(MANUAL_SOURCE_CAPS)
        if want_grow["downloaded"]:
            log("  growing DOWNLOADED lane for next round")
            grow_lane(DOWNLOADED_SOURCE_CAPS)

        lane_arg = "both" if all(want_grow.values()) else ("manual" if want_grow["manual"] else "downloaded")
        log(f"===== ROUND {round_idx}: re-collecting ({lane_arg} lane; resume, no duplicates) =====")
        c_code = run([PYTHON, str(SCRIPTS_DIR / "collect.py"), "--lane", lane_arg])
        if c_code != 0:
            final_reason = f"collect.py failed at round {round_idx} (exit {c_code}); stopping."
            log(final_reason)
            break

    summary = {
        "targets": {"manual_tokens_target": manual_target, "downloaded_tokens_target": downloaded_target},
        "final_totals": prev_totals,
        "reason": final_reason,
        "elapsed_seconds": round(time.time() - started, 1),
        "note": ("If a target was not met, it is a measured, structural shortfall (real "
                 "content pool exhausted / diminishing returns), not something forced by "
                 "loosening cleaning.yaml filters -- see growth_log.json for the round-by-round "
                 "numbers."),
    }
    GROWTH_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), "utf-8")
    log("===== DONE =====")
    log(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

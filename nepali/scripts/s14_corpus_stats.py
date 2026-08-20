#!/usr/bin/env python3
"""
Step 14 -- final corpus statistics: tokens, sources, retention funnel, dedup.

    python hindi/scripts/s14_corpus_stats.py

Reads the final splits (with real ``n_tokens`` from S13), every stage
manifest written along the way, and the structured event log, and produces
one JSON file the report generator (S15) and QA gate (S16) both read rather
than recomputing any of these numbers themselves. Nothing here is measured
twice by two different definitions.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpuslib import runlog, schema, stageio  # noqa: E402
from corpuslib.config import load_language_config  # noqa: E402
from corpuslib.jsonlio import atomic_write_json, read_jsonl  # noqa: E402

STAGE_ORDER = [
    ("s03_normalized", "S03 normalise"),
    ("s04_langid", "S04 language ID"),
    ("s07_train_pool", "S07 train pool (post-split)"),
    ("s05_heuristics", "S05 heuristic filters"),
    ("s06_perplexity", "S06 perplexity filter"),
    ("s08_dedup", "S08 deduplicate"),
    ("s09_decontaminated", "S09 decontaminate"),
    ("s10_balanced", "S10 balance lanes"),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args(argv)


def _split_breakdown(records: list[dict]) -> dict:
    by_source: dict[str, dict] = defaultdict(lambda: {"docs": 0, "words": 0, "tokens": 0, "lane": ""})
    by_lane: dict[str, dict] = {
        schema.LANE_MANUAL: {"docs": 0, "words": 0, "tokens": 0},
        schema.LANE_DOWNLOADED: {"docs": 0, "words": 0, "tokens": 0},
    }
    total_tokens = total_words = 0
    for r in records:
        src = by_source[r["source_id"]]
        src["docs"] += 1
        src["words"] += r.get("n_words", 0)
        src["tokens"] += r.get("n_tokens", 0)
        src["lane"] = r["lane"]
        lane = by_lane[r["lane"]]
        lane["docs"] += 1
        lane["words"] += r.get("n_words", 0)
        lane["tokens"] += r.get("n_tokens", 0)
        total_tokens += r.get("n_tokens", 0)
        total_words += r.get("n_words", 0)
    return {
        "docs": len(records),
        "total_words": total_words,
        "total_tokens": total_tokens,
        "by_lane": by_lane,
        "by_source": dict(by_source),
    }


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    config = load_language_config()
    config.ensure_dirs()
    log = runlog.StepLogger(config.lang, config.events_path)
    started = time.time()

    splits = {}
    for name in ("train", "val", "test"):
        path = config.final_dir / f"{name}.jsonl"
        if not path.exists():
            print(f"BLOCKED: {path} missing. Run s13_tokenizer_eval.py first.")
            log("S14", f"Corpus statistics ({config.display_name})", runlog.BLOCKED,
                f"{name}.jsonl missing.")
            return 1
        splits[name] = path

    print(f"S14 corpus statistics ({config.display_name})", flush=True)

    split_stats = {name: _split_breakdown(list(read_jsonl(path))) for name, path in splits.items()}
    overall_tokens = sum(s["total_tokens"] for s in split_stats.values())
    overall_manual_tokens = sum(
        s["by_lane"][schema.LANE_MANUAL]["tokens"] for s in split_stats.values()
    )
    overall_downloaded_tokens = overall_tokens - overall_manual_tokens

    # -- retention funnel, from stage manifests ------------------------------
    funnel = []
    for stage_name, label in STAGE_ORDER:
        manifest = stageio.stage_manifest(config.interim_dir / stage_name)
        if not manifest:
            continue
        docs_in = manifest.get("docs_in") or manifest.get("input_docs")
        docs_out = (
            manifest.get("docs_out") or manifest.get("docs_kept")
            or manifest.get("train_docs") or manifest.get("kept")
        )
        funnel.append({
            "stage": stage_name, "label": label,
            "docs_in": docs_in, "docs_out": docs_out,
            "retention": round(docs_out / docs_in, 4) if docs_in else None,
        })

    dedup_manifest = stageio.stage_manifest(config.interim_dir / "s08_dedup")
    decon_manifest = stageio.stage_manifest(config.interim_dir / "s09_decontaminated")
    balance_manifest = stageio.stage_manifest(config.interim_dir / "s10_balanced")

    meets_manual_target = overall_manual_tokens >= config.manual_tokens_target
    meets_downloaded_target = overall_downloaded_tokens >= config.downloaded_tokens_target
    payload = {
        "lang": config.lang,
        "model_role": config.model_role,
        "splits": split_stats,
        "overall_tokens": overall_tokens,
        "overall_manual_tokens": overall_manual_tokens,
        "overall_downloaded_tokens": overall_downloaded_tokens,
        "manual_tokens_target": config.manual_tokens_target,
        "downloaded_tokens_target": config.downloaded_tokens_target,
        "meets_manual_target": meets_manual_target,
        "meets_downloaded_target": meets_downloaded_target,
        "retention_funnel": funnel,
        "dedup": {
            "exact_removed": dedup_manifest.get("exact_duplicates_removed"),
            "near_removed": dedup_manifest.get("near_duplicates_removed"),
            "cross_lane_duplicate_pairs": dedup_manifest.get("cross_lane_duplicate_pairs"),
            "manual_docs_preserved_by_priority": dedup_manifest.get("manual_docs_preserved_by_priority"),
        },
        "decontamination": {
            "contaminated_removed": decon_manifest.get("contaminated_removed"),
            "rescan_remaining_matches": decon_manifest.get("rescan_remaining_matches"),
        },
        "lane_balancing": {
            "action": balance_manifest.get("action"),
            "ratio_before": balance_manifest.get("ratio_before"),
            "ratio_after": balance_manifest.get("ratio_after"),
            "downloaded_docs_dropped": balance_manifest.get("downloaded_docs_dropped"),
        },
        "seconds": round(time.time() - started, 1),
    }
    atomic_write_json(config.final_dir / "corpus_stats.json", payload)

    print(f"  overall: {overall_tokens:,} tokens -- manual {overall_manual_tokens:,} "
          f"(target {config.manual_tokens_target:,}, "
          f"{'met' if meets_manual_target else 'NOT MET'}), downloaded "
          f"{overall_downloaded_tokens:,} (target {config.downloaded_tokens_target:,}, "
          f"{'met' if meets_downloaded_target else 'NOT MET'})")
    for stage in funnel:
        ret = f"{stage['retention']:.2%}" if stage["retention"] is not None else "n/a"
        print(f"    {stage['stage']:<22} {stage['docs_in']!s:>10} -> {stage['docs_out']!s:>10}  ({ret})")

    log(
        "S14",
        f"Corpus statistics ({config.display_name})",
        runlog.SUCCESS if (meets_manual_target and meets_downloaded_target) else runlog.WARNING,
        f"{overall_tokens:,} final tokens. Manual {overall_manual_tokens:,} "
        f"(target {config.manual_tokens_target:,}, "
        f"{'met' if meets_manual_target else 'NOT MET'}). Downloaded {overall_downloaded_tokens:,} "
        f"(target {config.downloaded_tokens_target:,}, "
        f"{'met' if meets_downloaded_target else 'NOT MET'}).",
        metrics={
            "overall_tokens": overall_tokens,
            "overall_manual_tokens": overall_manual_tokens,
            "overall_downloaded_tokens": overall_downloaded_tokens,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

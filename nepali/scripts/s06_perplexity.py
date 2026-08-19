#!/usr/bin/env python3
"""
Step 06 -- drop the least fluent tail using a reference perplexity model.

    python hindi/scripts/s06_perplexity.py
    python hindi/scripts/s06_perplexity.py --smoke

An order-5 word n-gram model (stupid backoff -- see ``corpuslib.ngramlm``) is
trained on a held-out sample of edited prose: ``cleaning.yaml:
perplexity.reference_sources`` (Wikisource, OCR books by default) up to
``reference_sample_docs`` documents. Documents belonging to those reference
sources are excluded from scoring entirely -- they define what "reads like
edited prose" means for this run, so scoring them against themselves would be
circular. Every other document is scored and the worst
``drop_worst_fraction`` by cross-entropy is dropped (10% for Hindi's surplus
posture, 3% for Nepali).

**Self-check:** if the dropped tail's mean character count falls below
``sanity_min_mean_chars``, that means the "worst" documents are mostly
near-empty fragments the length filter should already have caught, which is
the signature of a broken or miscalibrated LM, not a real fluency signal. The
step skips itself in that case (passthrough, logged as FALLBACK_USED) rather
than dropping a fraction of the corpus for a reason that does not hold up.

Two passes over S05's output keep memory flat regardless of corpus size:
the first computes every document's score to find the cutoff; the second
re-scores and applies it. Only per-document scores (floats), never full text,
are held across the whole corpus at once.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpuslib import runlog, stageio  # noqa: E402
from corpuslib.config import load_language_config, load_yaml  # noqa: E402
from corpuslib.ngramlm import NGramLM, tokenize  # noqa: E402

STAGE_IN = "s05_heuristics"
STAGE_OUT = "s06_perplexity"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="process 5,000 documents")
    parser.add_argument("--limit", type=int, default=0, help="process at most N documents")
    return parser.parse_args(argv)


def _passthrough(config, stage_in_dir, log, reason: str, extra: dict) -> int:
    """Copy S05's output through unchanged and log why filtering was skipped."""
    written = stageio.write_stage(config.interim_dir / STAGE_OUT, stageio.read_stage(stage_in_dir))
    payload = {"applied": False, "reason": reason, "docs_out": written, **extra}
    stageio.write_stage_manifest(config.interim_dir / STAGE_OUT, payload)
    print(f"  SKIPPED: {reason}")
    log(
        "S06",
        f"Perplexity filter ({config.display_name})",
        runlog.FALLBACK,
        f"Filter skipped ({reason}); {written:,} documents passed through unchanged.",
        metrics=payload,
        fallback=reason,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_language_config()
    config.ensure_dirs()
    log = runlog.StepLogger(config.lang, config.events_path)
    started = time.time()
    limit = args.limit or (5000 if args.smoke else 0)

    stage_in_dir = config.interim_dir / STAGE_IN
    if not stageio.stage_manifest(stage_in_dir):
        print(f"BLOCKED: {stage_in_dir} has no stage manifest. Run s05_heuristics.py first.")
        log("S06", f"Perplexity filter ({config.display_name})", runlog.BLOCKED,
            "s05_heuristics has not been run yet.")
        return 1

    cleaning_cfg = load_yaml(config.configs_dir / "cleaning.yaml")
    pcfg = cleaning_cfg.get("perplexity", {}) or {}

    if not pcfg.get("enabled", True):
        return _passthrough(config, stage_in_dir, log, "disabled_in_config", {})

    reference_sources = set(pcfg.get("reference_sources", []) or [])
    reference_sample_docs = int(pcfg.get("reference_sample_docs", 8000))
    drop_fraction = float(pcfg.get("drop_worst_fraction", 0.10))
    order = int(pcfg.get("order", 5))
    sanity_min_mean_chars = float(pcfg.get("sanity_min_mean_chars", 0))

    print(
        f"S06 perplexity ({config.display_name}) -- order={order} "
        f"reference={sorted(reference_sources)} drop_worst={drop_fraction:.0%}",
        flush=True,
    )

    # -- train the reference model -----------------------------------------
    lm = NGramLM(order=order)
    ref_docs = 0
    ref_token_lists = []
    for record in stageio.read_stage(stage_in_dir):
        if record.get("source_id") in reference_sources:
            ref_token_lists.append(tokenize(record.get("text", "")))
            ref_docs += 1
            if ref_docs >= reference_sample_docs:
                break
    if ref_docs == 0:
        return _passthrough(
            config, stage_in_dir, log, "no_reference_documents_available",
            {"reference_sources": sorted(reference_sources)},
        )
    lm.fit(ref_token_lists)
    print(f"  reference model: {ref_docs:,} documents, vocab={lm.vocab_size:,}", flush=True)
    del ref_token_lists

    # -- pass 1: score every non-reference document to find the cutoff ------
    scores: list[float] = []
    docs_in = 0
    docs_scoreable = 0
    for record in stageio.read_stage(stage_in_dir):
        if limit and docs_in >= limit:
            break
        docs_in += 1
        if record.get("source_id") in reference_sources:
            continue
        docs_scoreable += 1
        scores.append(lm.score(tokenize(record.get("text", ""))))
        if docs_in % 200_000 == 0:
            print(f"  ...scored {docs_in:,}", flush=True)

    if not scores:
        return _passthrough(
            config, stage_in_dir, log, "no_scoreable_documents",
            {"docs_in": docs_in, "reference_sources": sorted(reference_sources)},
        )

    scores.sort()
    keep_count = max(0, round(len(scores) * (1.0 - drop_fraction)))
    cutoff = scores[keep_count - 1] if keep_count > 0 else scores[0]

    # -- sanity check: is the dropped tail near-empty, not just disfluent? --
    dropped_chars: list[int] = []
    for record in stageio.read_stage(stage_in_dir):
        if limit and dropped_chars.__len__() + 1 > limit:
            break
        if record.get("source_id") in reference_sources:
            continue
        score = lm.score(tokenize(record.get("text", "")))
        if score > cutoff:
            dropped_chars.append(record.get("n_chars") or len(record.get("text", "")))
    mean_dropped_chars = sum(dropped_chars) / len(dropped_chars) if dropped_chars else 0.0

    if sanity_min_mean_chars and mean_dropped_chars < sanity_min_mean_chars:
        return _passthrough(
            config, stage_in_dir, log, "dropped_tail_looks_near_empty_lm_may_be_broken",
            {
                "mean_dropped_chars": round(mean_dropped_chars, 1),
                "sanity_min_mean_chars": sanity_min_mean_chars,
                "reference_docs": ref_docs,
                "docs_scored": docs_scoreable,
            },
        )

    # -- pass 2: apply the cutoff -------------------------------------------
    docs_out = 0

    def records():
        nonlocal docs_out
        seen = 0
        for record in stageio.read_stage(stage_in_dir):
            if limit and seen >= limit:
                break
            seen += 1
            if record.get("source_id") in reference_sources:
                docs_out += 1
                yield record
                continue
            score = lm.score(tokenize(record.get("text", "")))
            if score <= cutoff:
                docs_out += 1
                record["meta"] = dict(record.get("meta") or {})
                record["meta"]["perplexity_score_bits"] = round(score, 4)
                yield record

    written = stageio.write_stage(config.interim_dir / STAGE_OUT, records())
    elapsed = time.time() - started
    retention = written / docs_in if docs_in else 0.0

    payload = {
        "applied": True,
        "docs_in": docs_in,
        "docs_kept": written,
        "retention": round(retention, 4),
        "reference_docs": ref_docs,
        "reference_sources": sorted(reference_sources),
        "docs_scored": docs_scoreable,
        "drop_worst_fraction": drop_fraction,
        "cutoff_bits_per_token": round(cutoff, 4),
        "mean_dropped_chars": round(mean_dropped_chars, 1),
        "seconds": round(elapsed, 1),
        "smoke": bool(args.smoke),
    }
    stageio.write_stage_manifest(config.interim_dir / STAGE_OUT, payload)

    print(f"  docs_in={docs_in:,}  kept={written:,}  retention={retention:.2%}")
    print(f"  cutoff={cutoff:.3f} bits/token, mean_dropped_chars={mean_dropped_chars:.0f}")

    log(
        "S06",
        f"Perplexity filter ({config.display_name})",
        runlog.SUCCESS,
        (
            f"Trained an order-{order} model on {ref_docs:,} reference documents; "
            f"dropped the worst {drop_fraction:.0%} of {docs_scoreable:,} scored "
            f"documents (cutoff {cutoff:.3f} bits/token)."
        ),
        metrics=payload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

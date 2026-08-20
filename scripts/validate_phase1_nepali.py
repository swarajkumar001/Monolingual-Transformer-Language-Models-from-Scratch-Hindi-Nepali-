"""Validate the Nepali (Model L) Phase 1 deliverable against the course spec.

Nepali uses a different pipeline (nepali/scripts/, stage-based S00-S14) and a
different data layout than validate_phase1.py's hindi/bhojpuri assumptions
(data/raw/*/metadata.jsonl, data/cleaned/metadata.jsonl, data/splits/), so it
cannot simply be added to that script's LANGUAGES list. This is a parallel
validator, same checks in spirit, reading Nepali's own output artifacts:

  - nepali/data/raw/{manual,downloaded}/*/_manifest.json (raw sources present)
  - nepali/data/final/corpus_stats.json (manual token-share floor,
    both absolute token targets)
  - nepali/data/final/split_stats.json (train/val/test splits exist,
    no document_id overlap - already computed by s11_finalize_splits.py,
    read here rather than re-scanned)
  - nepali/tokenizer/*.model (tokenizer trained)

Deliberately NOT checked here: cross-language duplication against
hindi/bhojpuri. Nepali's final splits are ~9.2GB of JSONL with no lightweight
per-document manifest (unlike hindi/bhojpuri's separate id-only manifests), so
a real check would mean streaming that full volume - too heavy to run as a
side effect of validation without a separate, explicit pass. Flagged as
SKIPPED below, not silently omitted.

Prints PASS/FAIL/SKIPPED per check and exits 1 if any check fails
(SKIPPED does not fail the run).
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NEPALI_ROOT = REPO_ROOT / "nepali"
MANUAL_FLOOR_PCT = 20.0


def check_raw_sources_present():
    """WHY: the spec requires both a manual and a downloaded lane; an empty
    lane should fail loudly rather than surface later as a wrong manual %.
    OUTPUT: (passed, detail) - passed iff both raw/manual/ and
    raw/downloaded/ have at least one source with docs > 0, read from each
    source's own small _manifest.json (not the multi-GB shard files)."""
    counts = {}
    for lane in ("manual", "downloaded"):
        lane_dir = NEPALI_ROOT / "data" / "raw" / lane
        docs = chars = 0
        if lane_dir.is_dir():
            for manifest_path in lane_dir.glob("*/_manifest.json"):
                try:
                    m = json.loads(manifest_path.read_text("utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                docs += int(m.get("docs", 0))
                chars += int(m.get("chars", 0))
        counts[lane] = (docs, chars)
    passed = counts["manual"][0] > 0 and counts["downloaded"][0] > 0
    detail = (
        f"manual={counts['manual'][0]:,} docs/{counts['manual'][1]:,} chars, "
        f"downloaded={counts['downloaded'][0]:,} docs/{counts['downloaded'][1]:,} chars"
    )
    return passed, detail


def _load_corpus_stats():
    path = NEPALI_ROOT / "data" / "final" / "corpus_stats.json"
    if not path.exists():
        return None
    return json.loads(path.read_text("utf-8"))


def check_manual_floor():
    """WHY: same 20% manual floor the hindi/bhojpuri validator gates on.
    WHAT: Nepali's corpus_stats.json tracks lane share by TOKENS, not
    characters (the pipeline never keeps a per-lane character tally past
    S03) - so this is token-weighted, not char-weighted like the
    hindi/bhojpuri check. Noted explicitly rather than silently differing."""
    stats = _load_corpus_stats()
    if not stats:
        return False, "no data/final/corpus_stats.json found - run s14_corpus_stats.py first"
    manual = int(stats["overall_manual_tokens"])
    total = int(stats["overall_tokens"])
    if total == 0:
        return False, "corpus_stats.json present but zero total tokens"
    pct = 100 * manual / total
    passed = pct >= MANUAL_FLOOR_PCT
    return passed, (
        f"{pct:.2f}% manual ({manual:,} / {total:,} tokens, TOKEN-weighted not "
        f"char-weighted), floor is {MANUAL_FLOOR_PCT}%"
    )


def check_absolute_token_targets():
    """WHY: Nepali's own pipeline (unlike hindi/bhojpuri) gates on two
    absolute token floors rather than a single overall total - this check
    mirrors what nepali/scripts/auto_grow_and_tokenize.py itself checks."""
    stats = _load_corpus_stats()
    if not stats:
        return False, "no data/final/corpus_stats.json found"
    manual = int(stats["overall_manual_tokens"])
    downloaded = int(stats["overall_downloaded_tokens"])
    manual_target = int(stats.get("manual_tokens_target", 100_000_000))
    downloaded_target = int(stats.get("downloaded_tokens_target", 400_000_000))
    passed = manual >= manual_target and downloaded >= downloaded_target
    return passed, (
        f"manual={manual:,} (target {manual_target:,}), "
        f"downloaded={downloaded:,} (target {downloaded_target:,}), "
        f"total={manual + downloaded:,}"
    )


def check_splits_no_leakage():
    """WHY: document-level leakage between train/val/test would let a model
    memorize during training and appear to generalize on eval.
    WHAT: unlike the hindi/bhojpuri check (which re-derives overlap from
    per-split id manifests here in this script), Nepali's S11 already
    computed and recorded this in split_stats.json - read directly rather
    than re-scanning the ~9.2GB of final JSONL to rebuild id sets."""
    path = NEPALI_ROOT / "data" / "final" / "split_stats.json"
    if not path.exists():
        return False, "no data/final/split_stats.json found - run s11_finalize_splits.py first"
    stats = json.loads(path.read_text("utf-8"))
    splits = stats.get("splits", {})
    if not all(k in splits for k in ("train", "val", "test")):
        return False, f"split_stats.json missing a split: has {list(splits)}"
    zero_overlap = bool(stats.get("zero_doc_id_overlap"))
    flagged = stats.get("flagged_splits", [])
    passed = zero_overlap and not flagged
    counts = ", ".join(f"{s}={splits[s]['docs']:,}" for s in ("train", "val", "test"))
    return passed, f"{counts}; zero_doc_id_overlap={zero_overlap}, flagged_splits={flagged}"


def check_tokenizer_trained():
    """WHY: a Phase 1 deliverable is a trained tokenizer; this just checks
    *some* .model file exists, matching the hindi/bhojpuri check's scope."""
    tok_dir = NEPALI_ROOT / "tokenizer"
    models = list(tok_dir.glob("*.model")) if tok_dir.exists() else []
    if not models:
        return False, "NOT YET MEASURED - no .model file in nepali/tokenizer/ yet"
    names = ", ".join(sorted(m.name for m in models))
    return True, f"{len(models)} tokenizer model(s): {names}"


def main():
    all_passed = True
    print("=" * 70)
    print("PHASE 1 VALIDATION -- Nepali (Model L)")
    print("=" * 70)

    checks = [
        ("raw sources present (manual + downloaded)", check_raw_sources_present()),
        ("manual share >= 20% (final corpus, TOKEN-weighted)", check_manual_floor()),
        ("absolute token targets met (manual >=100M, downloaded >=400M)", check_absolute_token_targets()),
        ("train/val/test splits exist, no doc_id leakage", check_splits_no_leakage()),
        ("tokenizer trained", check_tokenizer_trained()),
    ]
    for name, (passed, detail) in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}\n         {detail}")
        all_passed = all_passed and passed

    print(f"\n--- cross-language (vs. hindi/bhojpuri) ---")
    print(
        "  [SKIPPED] no shared documents between nepali and hindi/bhojpuri\n"
        "         Nepali's final splits (~9.2GB JSONL) have no lightweight "
        "id-only manifest to check against; a real check means streaming "
        "that full volume and should be run as its own explicit pass, not "
        "silently inside this validator."
    )

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED (cross-language check skipped, see above)" if all_passed else "SOME CHECKS FAILED")
    print("=" * 70)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

"""Validate Phase 1 deliverables for Hindi against the course spec.

Every check reads its numbers from the pipeline's own JSONL provenance files
(data/raw/*/metadata.jsonl, data/cleaned/metadata.jsonl, data/splits/*_manifest.json)
rather than from any hardcoded count, so this script stays correct as the
corpus keeps changing. Run after clean_hindi.py and split_hindi.py.

Checks:
  - both raw/manual and raw/downloaded sources are non-empty
  - manual token share (char-weighted, from source_type) is >= 20%
  - train/validation/test splits exist and do not overlap (no shared document_id)
  - a SentencePiece tokenizer model has been trained

Originally also validated Bhojpuri and the cross-language overlap between
Hindi and Bhojpuri, back when Bhojpuri was Model L. Bhojpuri has since been
superseded by Nepali and its tracked files removed from the repo, so this
script now covers Hindi only. Nepali has its own parallel validator
(scripts/validate_phase1_nepali.py, since its pipeline/data layout differs)
and its own cross-language check against Hindi
(scripts/check_nepali_hindi_overlap.py) - run those separately.

Prints PASS/FAIL per check and exits 1 if any check fails.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANUAL_FLOOR_PCT = 20.0
LANGUAGES = ["hindi"]


def load_jsonl(path: Path):
    """WHY: every check below reads a *.jsonl provenance file; centralizing
    the read means a missing file or one malformed line (e.g. from a past
    concurrent-write race, since fixed) never crashes a check - it's
    treated as "nothing there yet" or "skip this line", matching how the
    pipeline scripts themselves tolerate malformed lines (see
    clean_lib.read_metadata).
    INPUT: path to a .jsonl file (may not exist).
    OUTPUT: generator yielding each valid line's parsed dict; yields
    nothing if the file is missing."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_manifest(path: Path):
    """WHY: split manifests are a single JSON array (not JSONL), written
    once by split_<lang>.py - a distinct format from the streaming
    provenance logs load_jsonl reads, so it needs its own loader.
    INPUT: path to a *_manifest.json file (may not exist).
    OUTPUT: the parsed list of document metadata dicts, or [] if missing
    or malformed."""
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def check_raw_sources_present(lang: str):
    """WHY: the spec requires both a manual and a downloaded lane per
    language; an empty lane (e.g. a crawl that silently produced 0 docs)
    should fail loudly here rather than surface later as a wrong manual %.
    INPUT: lang ("hindi" or "bhojpuri").
    OUTPUT: (passed: bool, detail: str) - passed iff both raw/manual/ and
    raw/downloaded/ have at least one document."""
    counts = {}
    for kind in ["manual", "downloaded"]:
        meta_path = REPO_ROOT / lang / "data" / "raw" / kind / "metadata.jsonl"
        docs = list(load_jsonl(meta_path))
        counts[kind] = (len(docs), sum(d.get("characters", 0) for d in docs))
    passed = counts["manual"][0] > 0 and counts["downloaded"][0] > 0
    detail = (
        f"manual={counts['manual'][0]:,} docs/{counts['manual'][1]:,} chars, "
        f"downloaded={counts['downloaded'][0]:,} docs/{counts['downloaded'][1]:,} chars"
    )
    return passed, detail


def check_manual_floor(lang: str):
    """Manual share is computed from the cleaned corpus (what actually ends up
    in the model), char-weighted by the source_type field each document
    carries end to end from raw/ through cleaned/ - never from a separate
    counter that could drift from the provenance data itself."""
    meta_path = REPO_ROOT / lang / "data" / "cleaned" / "metadata.jsonl"
    docs = list(load_jsonl(meta_path))
    if not docs:
        return False, "no cleaned metadata found - run clean_*.py first"
    manual_chars = sum(d.get("characters", 0) for d in docs if d.get("source_type") == "manual")
    total_chars = sum(d.get("characters", 0) for d in docs)
    if total_chars == 0:
        return False, "cleaned metadata present but zero characters"
    pct = 100 * manual_chars / total_chars
    passed = pct >= MANUAL_FLOOR_PCT
    return passed, f"{pct:.2f}% manual ({manual_chars:,} / {total_chars:,} cleaned chars), floor is {MANUAL_FLOOR_PCT}%"


def check_splits_no_leakage(lang: str):
    """WHY: document-level leakage between splits (the same document_id
    appearing in more than one of train/validation/test) would let a model
    memorize during training and appear to generalize on eval - a silent,
    serious eval-integrity bug, not just a data-hygiene nit.
    INPUT: lang.
    OUTPUT: (passed, detail) - passed iff all three manifests exist,
    each is internally duplicate-free, and no document_id appears in more
    than one split."""
    splits_dir = REPO_ROOT / lang / "data" / "splits"
    manifests = {}
    for split in ["train", "validation", "test"]:
        manifest = load_manifest(splits_dir / f"{split}_manifest.json")
        if not manifest:
            return False, f"{split}_manifest.json missing or empty"
        manifests[split] = manifest

    id_sets = {split: {d["document_id"] for d in docs} for split, docs in manifests.items()}
    for split, docs in manifests.items():
        if len(id_sets[split]) != len(docs):
            return False, f"{split} manifest has duplicate document_ids within itself"

    overlaps = []
    splits = list(id_sets)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            shared = id_sets[splits[i]] & id_sets[splits[j]]
            if shared:
                overlaps.append(f"{splits[i]}∩{splits[j]}={len(shared)}")
    if overlaps:
        return False, f"document_id overlap between splits: {', '.join(overlaps)}"

    counts = ", ".join(f"{s}={len(id_sets[s]):,}" for s in splits)
    return True, f"no overlap; {counts}"


def check_tokenizer_trained(lang: str):
    """WHY: a Phase 1 deliverable is a trained tokenizer per language; this
    just checks *some* .model file exists, not which vocab size was
    chosen (that decision and its justification live in
    tokenizer_analysis.md, not something this script can verify).
    INPUT: lang.
    OUTPUT: (passed, detail) - passed iff at least one *.model file exists
    under <lang>/tokenizer/."""
    tok_dir = REPO_ROOT / lang / "tokenizer"
    models = list(tok_dir.glob("*.model")) if tok_dir.exists() else []
    if not models:
        return False, "NOT YET MEASURED - no .model file in tokenizer/ yet"
    names = ", ".join(m.name for m in models)
    return True, f"{len(models)} tokenizer model(s): {names}"


def main():
    """WHY: single command to answer "is Phase 1 actually done" without
    manually re-deriving every number from the raw data each time.
    WHAT: runs all 4 per-language checks for each language in LANGUAGES,
    plus the cross-language check once, printing PASS/FAIL with the
    measured detail for each.
    INPUT/OUTPUT: none in; prints the full report; exits 1 if any check
    failed, 0 if all passed (so this is usable as a CI-style gate, not
    just a human-readable report)."""
    all_passed = True
    print("=" * 70)
    print("PHASE 1 VALIDATION")
    print("=" * 70)

    for lang in LANGUAGES:
        print(f"\n--- {lang} ---")
        checks = [
            ("raw sources present (manual + downloaded)", check_raw_sources_present(lang)),
            ("manual share >= 20% (cleaned corpus, char-weighted)", check_manual_floor(lang)),
            ("train/validation/test splits exist, no leakage", check_splits_no_leakage(lang)),
            ("tokenizer trained", check_tokenizer_trained(lang)),
        ]
        for name, (passed, detail) in checks:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}\n         {detail}")
            all_passed = all_passed and passed

    print(f"\n--- cross-language (Nepali vs. Hindi) ---")
    print("  see scripts/check_nepali_hindi_overlap.py - not run here, since")
    print("  this script only reads Hindi's own cleaned metadata")

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 70)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

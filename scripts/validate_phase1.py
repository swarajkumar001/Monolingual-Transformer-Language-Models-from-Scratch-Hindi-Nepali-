"""Validate Phase 1 deliverables for both languages against the course spec.

Every check reads its numbers from the pipeline's own JSONL provenance files
(data/raw/*/metadata.jsonl, data/cleaned/metadata.jsonl, data/splits/*_manifest.json)
rather than from any hardcoded count, so this script stays correct as the
corpus keeps changing. Run after clean_<lang>.py and split_<lang>.py.

Checks per language:
  - both raw/manual and raw/downloaded sources are non-empty
  - manual token share (char-weighted, from source_type) is >= 20%
  - train/validation/test splits exist and do not overlap (no shared document_id)
  - a SentencePiece tokenizer model has been trained

Cross-language check:
  - no document_id or content (dedup_key) is shared between the two cleaned corpora

Prints PASS/FAIL per check and exits 1 if any check fails.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANUAL_FLOOR_PCT = 20.0
LANGUAGES = ["hindi", "bhojpuri"]


def load_jsonl(path: Path):
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
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def check_raw_sources_present(lang: str):
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
    tok_dir = REPO_ROOT / lang / "tokenizer"
    models = list(tok_dir.glob("*.model")) if tok_dir.exists() else []
    if not models:
        return False, "NOT YET MEASURED - no .model file in tokenizer/ yet"
    names = ", ".join(m.name for m in models)
    return True, f"{len(models)} tokenizer model(s): {names}"


def check_no_cross_language_sharing():
    cleaned = {}
    for lang in LANGUAGES:
        meta_path = REPO_ROOT / lang / "data" / "cleaned" / "metadata.jsonl"
        cleaned[lang] = list(load_jsonl(meta_path))

    if not all(cleaned.values()):
        missing = [lang for lang, docs in cleaned.items() if not docs]
        return False, f"cannot check yet - no cleaned metadata for: {', '.join(missing)}"

    ids = {lang: {d["document_id"] for d in docs} for lang, docs in cleaned.items()}
    keys = {
        lang: {d["dedup_key"] for d in docs if d.get("dedup_key")}
        for lang, docs in cleaned.items()
    }

    id_overlap = ids[LANGUAGES[0]] & ids[LANGUAGES[1]]
    key_overlap = keys[LANGUAGES[0]] & keys[LANGUAGES[1]]

    if id_overlap or key_overlap:
        return False, f"shared document_ids={len(id_overlap)}, shared dedup_keys(content)={len(key_overlap)}"
    return True, f"0 shared document_ids, 0 shared dedup_keys across {LANGUAGES[0]}/{LANGUAGES[1]}"


def main():
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

    print(f"\n--- cross-language ---")
    passed, detail = check_no_cross_language_sharing()
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] no shared documents between {LANGUAGES[0]} and {LANGUAGES[1]}\n         {detail}")
    all_passed = all_passed and passed

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 70)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

"""Remove documents whose content (dedup_key) is shared between the Hindi
and Bhojpuri cleaned corpora, keeping the copy in whichever language the
text actually is and deleting it from the other.

WHY: validate_phase1.py's cross-language check found 2 documents (out of
704,103 Hindi + 507,994 Bhojpuri) with an identical dedup_key on both sides.
Inspection showed both are genuinely Bhojpuri news text (Bhojpuri markers
"कहलन"/"जइहें" etc.) that ai4bharat/sangraha mislabeled as Hindi (`hin`)
while HuggingFaceFW/fineweb-2 correctly labeled the same text `bho_Deva`.
clean_hindi.py's lexical-marker contamination filter (see clean_lib.py
BHOJPURI_MARKERS) missed both because their specific Bhojpuri markers
aren't in that fixed word list - exactly the "coarse heuristic, will not
catch every case" limitation already documented in clean_stats.json's
bhojpuri_reject_caveat. This script is the remediation: at this scale (2
docs total) a full pipeline rerun with an expanded marker list isn't
warranted, so the two mislabeled copies are removed directly from the
Hindi side (where they don't belong), leaving the correctly-labeled
Bhojpuri copies untouched.

WHAT: for each language, drops the given document_ids from
data/cleaned/metadata.jsonl and deletes their text files. Does not touch
data/splits/ - callers must rerun split_<lang>.py afterward to regenerate
splits from the corrected cleaned corpus.

INPUT: hindi/data/cleaned/{metadata.jsonl,text/}, a hardcoded list of
document_ids to remove (found via validate_phase1.py + manual inspection,
not automated - this is a one-time, audited correction, not a recurring
pipeline stage).
OUTPUT: rewrites hindi/data/cleaned/metadata.jsonl with the 2 docs removed
and deletes their .txt files. Prints before/after doc and char counts.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Confirmed via: shared dedup_key between hindi/bhojpuri cleaned corpora,
# then manual text inspection (see report/phase1/dataset_statistics.md
# "Cross-language contamination" section for the full writeup).
REMOVE_FROM_HINDI = [
    "hi_downloaded_sangraha_e6bb7e97f6",
    "hi_downloaded_sangraha_unverified_7ca84f1c71",
]


def remove_docs(lang: str, doc_ids: list[str]) -> None:
    cleaned_dir = REPO_ROOT / lang / "data" / "cleaned"
    meta_path = cleaned_dir / "metadata.jsonl"
    text_dir = cleaned_dir / "text"

    docs = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    before_docs, before_chars = len(docs), sum(d.get("characters", 0) for d in docs)

    remove_set = set(doc_ids)
    kept = [d for d in docs if d["document_id"] not in remove_set]
    removed = [d for d in docs if d["document_id"] in remove_set]

    meta_path.write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in kept) + "\n",
        encoding="utf-8",
    )
    for d in removed:
        (text_dir / f"{d['document_id']}.txt").unlink(missing_ok=True)

    after_docs, after_chars = len(kept), sum(d.get("characters", 0) for d in kept)
    print(f"[{lang}] removed {len(removed)}/{len(doc_ids)} requested docs")
    print(f"[{lang}] docs: {before_docs:,} -> {after_docs:,}   chars: {before_chars:,} -> {after_chars:,}")
    if len(removed) != len(doc_ids):
        missing = remove_set - {d["document_id"] for d in removed}
        print(f"[{lang}] WARNING: not found in cleaned metadata: {missing}")


if __name__ == "__main__":
    remove_docs("hindi", REMOVE_FROM_HINDI)

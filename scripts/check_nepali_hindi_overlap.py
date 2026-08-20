"""Cross-language duplicate-content check: Nepali vs. Hindi.

The assignment spec requires "do not share documents across corpora." The
hindi/bhojpuri validator (validate_phase1.py) already checks this pairwise
between those two using each cleaned corpus's dedup_key field. Nepali was
never checked against Hindi. This script does that check using the exact
same dedup_key algorithm (scripts/clean_lib.py:dedup_key - sha1 of
whitespace-collapsed, punctuation-stripped, lowercased text), applied fresh
to Nepali's text (which has no dedup_key field of its own, unlike the
hindi/bhojpuri pipeline).

Hindi's side is read from data/cleaned/metadata.jsonl only (dedup_key is
already computed and stored there) - this deliberately avoids opening
Hindi's 704,101 individual per-document text files under data/cleaned/text/,
which would be far slower for no benefit here.

Nepali's side streams data/final/{train,val,test}.jsonl (~9.2GB total) and
computes dedup_key(text) fresh for every document.

Prints a report; exits 1 if any shared dedup_key is found between the two
corpora (matching validate_phase1.py's convention), 0 otherwise.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_WHITESPACE_RUN = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)


def dedup_key(text: str) -> str:
    """Identical to scripts/clean_lib.py:dedup_key - must stay byte-for-byte
    the same algorithm for the two languages' hashes to be comparable."""
    key = _WHITESPACE_RUN.sub(" ", text).strip()
    key = _NON_WORD.sub("", key).lower()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def load_hindi_dedup_keys() -> dict[str, str]:
    """OUTPUT: {dedup_key: document_id}, from hindi/data/cleaned/metadata.jsonl
    only - text files are never opened."""
    path = REPO_ROOT / "hindi" / "data" / "cleaned" / "metadata.jsonl"
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            key = d.get("dedup_key")
            if key:
                out[key] = d["document_id"]
    return out


def check_nepali_split(split_path: Path, hindi_keys: dict[str, str]) -> list[tuple[str, str, str]]:
    """OUTPUT: list of (nepali_doc_id, hindi_doc_id, dedup_key) for every
    Nepali document whose freshly-computed dedup_key matches a Hindi one."""
    matches = []
    n = 0
    with split_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            n += 1
            key = dedup_key(d.get("text", ""))
            hindi_id = hindi_keys.get(key)
            if hindi_id:
                matches.append((d.get("doc_id", "?"), hindi_id, key))
            if n % 200_000 == 0:
                print(f"  ...{split_path.name}: {n:,} docs checked, {len(matches)} matches so far", flush=True)
    print(f"  {split_path.name}: {n:,} docs checked total")
    return matches


def main() -> int:
    started = time.time()
    print("Loading Hindi dedup_key set from data/cleaned/metadata.jsonl...")
    hindi_keys = load_hindi_dedup_keys()
    print(f"  {len(hindi_keys):,} Hindi dedup_keys loaded ({time.time()-started:.1f}s)")

    all_matches: list[tuple[str, str, str]] = []
    final_dir = REPO_ROOT / "nepali" / "data" / "final"
    for split in ("train", "val", "test"):
        path = final_dir / f"{split}.jsonl"
        print(f"\nChecking nepali/{path.relative_to(REPO_ROOT)} against Hindi...")
        matches = check_nepali_split(path, hindi_keys)
        all_matches.extend(matches)

    elapsed = time.time() - started
    print("\n" + "=" * 70)
    print(f"CROSS-LANGUAGE CHECK: Nepali vs. Hindi ({elapsed:.1f}s total)")
    print("=" * 70)
    if all_matches:
        print(f"[FAIL] {len(all_matches)} shared dedup_key(s) found between Nepali and Hindi:")
        for nep_id, hin_id, key in all_matches[:20]:
            print(f"    nepali={nep_id}  hindi={hin_id}  dedup_key={key}")
        if len(all_matches) > 20:
            print(f"    ... and {len(all_matches) - 20} more")
        return 1
    else:
        print("[PASS] 0 shared dedup_keys between Nepali and Hindi's cleaned corpora")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

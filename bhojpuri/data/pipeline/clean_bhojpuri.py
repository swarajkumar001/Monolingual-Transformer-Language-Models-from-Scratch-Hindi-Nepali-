"""Clean Bhojpuri raw text (downloaded + manual) into data/interim/ and data/cleaned/.

Pipeline stage per report/phase1/phase1_strategy.md section 6 (plus the
non-language-word removal stage added per course ruling - digits stay):
  raw text -> NFC normalize + strip zero-width/control chars -> strip pure-
  Latin-script tokens -> malformed-Unicode and short-doc removal ->
  [data/interim/ snapshot here] -> language-ID filter -> exact + near-duplicate
  removal -> data/cleaned/

Language-ID caveat (documented, not silently ignored): langdetect's supported-
language list does NOT include Bhojpuri ("bho"). Bhojpuri is also written in
Devanagari and shares heavy vocabulary overlap with Hindi, so even if langdetect
guessed a nearby label (usually "hi" or "ne") that verdict would not reliably
distinguish real Bhojpuri text from Hindi/Nepali text mislabeled as Bhojpuri by
a source. Given that, this script relies on a Devanagari-script-ratio filter as
the practical language check, on the assumption that the upstream sources
(Sangraha-adjacent BhojpuriCorpus partitions, bh-Wikipedia) already label
Bhojpuri correctly at the document level. This is a real limitation of the
Bhojpuri pipeline, not a solved problem - it should be called out in the Phase 1
report, not hidden.

Run after download_bhojpuri.py (and ingest_manual.py, once manual data exists).
Always rebuilds data/interim/ and data/cleaned/ from data/raw/ rather than
appending, so it is safe to re-run whenever raw data changes.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from clean_lib import (  # noqa: E402
    dedup_key,
    devanagari_ratio,
    is_too_short,
    malformed_ratio,
    normalize_text,
    read_metadata,
    reset_dir,
    strip_latin_tokens,
    write_doc,
)

LANG = "bhojpuri"
LANG_ISO = "bho"  # not a langdetect-supported code; see module docstring
DATA_DIR = Path(__file__).resolve().parents[1]
RAW_DIRS = [DATA_DIR / "raw" / "downloaded", DATA_DIR / "raw" / "manual"]
INTERIM_DIR = DATA_DIR / "interim"
CLEANED_DIR = DATA_DIR / "cleaned"
STATS_PATH = DATA_DIR / "metadata" / "clean_stats.json"
DROPPED_LOG_PATH = DATA_DIR / "metadata" / "dropped_documents.jsonl"

# Measured from data/raw/downloaded/metadata.jsonl before picking this: Bhojpuri
# source docs are sentence-level (median 63 chars, mostly short but genuine
# sentences from BhojpuriCorpus partitions like eng_bho and Wikipedia stubs, not
# boilerplate - see report/phase1/phase1_strategy.md). A Hindi-style 200-char
# floor would discard ~83% of documents (~21% of characters) of an already-
# scarce corpus. 10 chars only removes genuinely degenerate fragments
# (single words / punctuation), which measurement showed is ~1,993 documents.
MIN_CHARS = 10
MIN_DEVANAGARI_RATIO = 0.5
MAX_MALFORMED_RATIO = 0.01


def language_ok(text: str) -> tuple[bool, str]:
    if devanagari_ratio(text) < MIN_DEVANAGARI_RATIO:
        return False, "low_devanagari_ratio"
    return True, ""


def main():
    parser = argparse.ArgumentParser(description="Clean Bhojpuri raw text (Model L)")
    parser.add_argument("--min-chars", type=int, default=MIN_CHARS)
    args = parser.parse_args()

    reset_dir(INTERIM_DIR)
    reset_dir(CLEANED_DIR)
    DROPPED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DROPPED_LOG_PATH.exists():
        DROPPED_LOG_PATH.unlink()

    seen_hashes = set()
    stats = {
        "docs_in": 0,
        "docs_cleaned": 0,
        "dropped_malformed": 0,
        "dropped_short": 0,
        "dropped_language": {"low_devanagari_ratio": 0},
        "dropped_duplicate": 0,
        "chars_in": 0,
        "chars_cleaned": 0,
        "latin_tokens_removed": 0,
        "language_id_caveat": (
            "langdetect has no 'bho' label; Devanagari-script-ratio was used as a "
            "practical proxy and cannot distinguish Bhojpuri from closely-related "
            "Hindi/Nepali text mislabeled at the source."
        ),
    }

    for raw_dir in RAW_DIRS:
        meta_path = raw_dir / "metadata.jsonl"
        for meta in read_metadata(meta_path):
            stats["docs_in"] += 1
            doc_id = meta["document_id"]
            text_path = raw_dir / "text" / f"{doc_id}.txt"
            if not text_path.exists():
                continue
            raw_text = text_path.read_text(encoding="utf-8", errors="ignore")
            stats["chars_in"] += len(raw_text)

            text = normalize_text(raw_text)
            text, n_removed = strip_latin_tokens(text)
            text = normalize_text(text)  # re-collapse whitespace left by removed tokens
            stats["latin_tokens_removed"] += n_removed

            if malformed_ratio(text) > MAX_MALFORMED_RATIO:
                stats["dropped_malformed"] += 1
                _log_dropped(doc_id, "malformed_unicode")
                continue
            if is_too_short(text, args.min_chars):
                stats["dropped_short"] += 1
                _log_dropped(doc_id, "too_short")
                continue

            interim_meta = dict(meta)
            interim_meta["characters"] = len(text)
            interim_meta["cleaning_status"] = "interim"
            write_doc(INTERIM_DIR, doc_id, text, interim_meta)

            ok, reason = language_ok(text)
            if not ok:
                stats["dropped_language"][reason] = stats["dropped_language"].get(reason, 0) + 1
                _log_dropped(doc_id, reason)
                continue

            key = dedup_key(text)
            if key in seen_hashes:
                stats["dropped_duplicate"] += 1
                _log_dropped(doc_id, "duplicate")
                continue
            seen_hashes.add(key)

            cleaned_meta = dict(meta)
            cleaned_meta["characters"] = len(text)
            cleaned_meta["cleaning_status"] = "cleaned"
            write_doc(CLEANED_DIR, doc_id, text, cleaned_meta)
            stats["docs_cleaned"] += 1
            stats["chars_cleaned"] += len(text)

    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[{LANG}] {stats['docs_cleaned']}/{stats['docs_in']} docs kept "
          f"({stats['chars_cleaned']} / {stats['chars_in']} chars)")
    print(json.dumps(stats, indent=2))


def _log_dropped(doc_id: str, reason: str) -> None:
    with DROPPED_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"document_id": doc_id, "reason": reason}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

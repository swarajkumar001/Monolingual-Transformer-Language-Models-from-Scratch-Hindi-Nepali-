"""Clean Hindi raw text (downloaded + manual) into data/interim/ and data/cleaned/.

Pipeline stage per report/phase1/phase1_strategy.md section 6 (plus the
non-language-word removal stage added per course ruling - digits stay):
  raw text -> NFC normalize + strip zero-width/control chars -> strip pure-
  Latin-script tokens -> malformed-Unicode and short-doc removal ->
  [data/interim/ snapshot here] -> language-ID filter -> exact + near-duplicate
  removal -> data/cleaned/

Run after download_hindi.py (and ingest_manual.py, once manual data exists).
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
    langdetect_verdict,
    malformed_ratio,
    normalize_text,
    read_metadata,
    reset_dir,
    strip_latin_tokens,
    write_doc,
)

LANG = "hindi"
LANG_ISO = "hi"
DATA_DIR = Path(__file__).resolve().parents[1]
RAW_DIRS = [DATA_DIR / "raw" / "downloaded", DATA_DIR / "raw" / "manual"]
INTERIM_DIR = DATA_DIR / "interim"
CLEANED_DIR = DATA_DIR / "cleaned"
STATS_PATH = DATA_DIR / "metadata" / "clean_stats.json"
DROPPED_LOG_PATH = DATA_DIR / "metadata" / "dropped_documents.jsonl"

MIN_CHARS = 200
MIN_DEVANAGARI_RATIO = 0.5
MAX_MALFORMED_RATIO = 0.01
LANGDETECT_MIN_PROB = 0.85


def language_ok(text: str) -> tuple[bool, str]:
    """Hindi has full langdetect support, so trust its verdict as the primary
    check. Devanagari-ratio is a cheap, script-level pre-filter."""
    if devanagari_ratio(text) < MIN_DEVANAGARI_RATIO:
        return False, "low_devanagari_ratio"
    verdict = langdetect_verdict(text, LANG_ISO, min_prob=LANGDETECT_MIN_PROB)
    if verdict is False:
        return False, "langdetect_mismatch"
    # verdict is None -> langdetect couldn't decide (too short/ambiguous text);
    # fall back to the Devanagari-ratio pass above instead of dropping it.
    return True, ""


def main():
    parser = argparse.ArgumentParser(description="Clean Hindi raw text (Model H)")
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
        "dropped_language": {"low_devanagari_ratio": 0, "langdetect_mismatch": 0},
        "dropped_duplicate": 0,
        "chars_in": 0,
        "chars_cleaned": 0,
        "latin_tokens_removed": 0,
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

            # data/interim/ snapshot: normalized text that passed the cheap
            # structural checks, before language-ID and dedup are applied.
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

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
    acquire_single_instance_lock,
    dedup_key,
    devanagari_ratio,
    is_too_short,
    langdetect_verdict,
    load_resume_state,
    malformed_ratio,
    marker_counts,
    normalize_text,
    read_metadata,
    recompute_disk_truth_stats,
    release_single_instance_lock,
    reset_dir,
    strip_latin_tokens,
    write_doc,
)

CHECKPOINT_EVERY = 2000
LOCK_PATH = Path(__file__).resolve().parents[1] / "metadata" / ".clean.lock"

LANG = "hindi"
LANG_ISO = "hi"
DATA_DIR = Path(__file__).resolve().parents[1]
# Manual processed BEFORE downloaded: dedup below keeps first-seen and drops
# later duplicates, so this ordering means a manual copy always wins a
# cross-source duplicate tie against a downloaded copy of the same text
# (e.g. a scraped article that also appears in a bulk corpus), never the
# reverse. Never share this ordering assumption without this comment - it's
# load-bearing for the dedup tie-break rule, not incidental.
RAW_DIRS = [DATA_DIR / "raw" / "manual", DATA_DIR / "raw" / "downloaded"]
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
    check. Devanagari-ratio is a cheap, script-level pre-filter.

    langdetect has no 'bho' label, so Bhojpuri text - also Devanagari, with
    heavy vocabulary overlap with Hindi - gets misclassified as 'hi' by
    langdetect and would otherwise pass straight into the Hindi corpus. That
    violates the course's no-shared-documents-across-languages rule, not just
    a style/quality issue, so it must be rejected here rather than merely
    logged. The lexical marker check runs before langdetect for that reason:
    a document that reads as Bhojpuri should never reach the Hindi corpus
    regardless of what langdetect would have said about it."""
    if devanagari_ratio(text) < MIN_DEVANAGARI_RATIO:
        return False, "low_devanagari_ratio"
    bho_hits, hin_hits = marker_counts(text)
    if bho_hits >= 2 and bho_hits > hin_hits:
        return False, "likely_bhojpuri"
    verdict = langdetect_verdict(text, LANG_ISO, min_prob=LANGDETECT_MIN_PROB)
    if verdict is False:
        return False, "langdetect_mismatch"
    # verdict is None -> langdetect couldn't decide (too short/ambiguous text);
    # fall back to the Devanagari-ratio pass above instead of dropping it.
    return True, ""


def main():
    parser = argparse.ArgumentParser(description="Clean Hindi raw text (Model H)")
    parser.add_argument("--min-chars", type=int, default=MIN_CHARS)
    parser.add_argument("--resume", action="store_true",
                         help="Continue an interrupted run instead of rebuilding from "
                              "scratch: skip docs already in data/interim/ or the dropped "
                              "log, reload dedup hashes from data/cleaned/, and resume the "
                              "running stats counters.")
    args = parser.parse_args()

    acquire_single_instance_lock(LOCK_PATH)

    processed_ids = set()
    seen_hashes = set()
    resumed_stats = None
    if args.resume:
        processed_ids, seen_hashes, resumed_stats = load_resume_state(
            INTERIM_DIR, CLEANED_DIR, DROPPED_LOG_PATH, STATS_PATH)
        print(f"[{LANG}] resuming: {len(processed_ids)} docs already processed, "
              f"{len(seen_hashes)} dedup hashes reloaded")
    else:
        reset_dir(INTERIM_DIR)
        reset_dir(CLEANED_DIR)
        DROPPED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if DROPPED_LOG_PATH.exists():
            DROPPED_LOG_PATH.unlink()

    stats = {
        "docs_in": 0,
        "docs_cleaned": 0,
        "dropped_malformed": 0,
        "dropped_short": 0,
        "dropped_language": {
            "low_devanagari_ratio": 0,
            "likely_bhojpuri": 0,
            "langdetect_mismatch": 0,
        },
        "dropped_duplicate": 0,
        "chars_in": 0,
        "chars_cleaned": 0,
        "latin_tokens_removed": 0,
        "bhojpuri_reject_caveat": (
            "langdetect has no 'bho' label, so it cannot itself catch Bhojpuri text "
            "misclassified as Hindi (they share vocabulary and script). "
            "dropped_language.likely_bhojpuri is an active reject filter (see "
            "clean_lib.marker_counts / language_ok below), not just a count: any "
            "document where Bhojpuri lexical markers (बा/बाटे/बाड़ें/एगो/गइल/भइल/"
            "कइल/etc., >=2 hits) outnumber Hindi markers (है/हैं/एक/गया/हुआ/किया/etc.) "
            "is dropped before reaching data/cleaned/. This is a coarse lexical "
            "heuristic, not a language model, so it will not catch every case - "
            "documented as a known limitation, not a solved problem."
        ),
    }
    if resumed_stats is not None:
        stats.update(resumed_stats)

    for raw_dir in RAW_DIRS:
        meta_path = raw_dir / "metadata.jsonl"
        for meta in read_metadata(meta_path):
            doc_id = meta["document_id"]
            if doc_id in processed_ids:
                continue
            stats["docs_in"] += 1
            if stats["docs_in"] % CHECKPOINT_EVERY == 0:
                STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
                STATS_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
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
            cleaned_meta["dedup_key"] = key
            write_doc(CLEANED_DIR, doc_id, text, cleaned_meta)
            stats["docs_cleaned"] += 1
            stats["chars_cleaned"] += len(text)

    stats = recompute_disk_truth_stats(stats, INTERIM_DIR, CLEANED_DIR, DROPPED_LOG_PATH)
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[{LANG}] {stats['docs_cleaned']}/{stats['docs_in']} docs kept "
          f"({stats['chars_cleaned']} / {stats['chars_in']} chars)")
    print(json.dumps(stats, indent=2))
    release_single_instance_lock(LOCK_PATH)


def _log_dropped(doc_id: str, reason: str) -> None:
    with DROPPED_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"document_id": doc_id, "reason": reason}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

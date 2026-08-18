"""Shared cleaning/normalization/dedup logic for the Hindi and Bhojpuri pipelines.

Shared code only - each language's raw/interim/cleaned/splits data stays in its own
directory tree and is never mixed here. Implements the pipeline described in
report/phase1/phase1_strategy.md section 6:

  raw text -> NFC normalize + strip zero-width/control chars -> language-ID filter
  -> boilerplate/short-doc/malformed-Unicode removal -> exact + near-duplicate removal
"""

import hashlib
import json
import os
import re
import time
import unicodedata
from pathlib import Path

try:
    from langdetect import DetectorFactory, LangDetectException, detect_langs

    DetectorFactory.seed = 0
except ImportError:
    detect_langs = None
    LangDetectException = Exception

# Zero-width and control characters that are NOT meaningful in Devanagari text:
# LRM/RLM (‎/‏, directional marks, unused in Devanagari), BOM
# (﻿), and C0/DEL control codes. Deliberately excludes ZWNJ (‌) and
# ZWJ (‍) - those control conjunct ligature formation in real Devanagari
# orthography (e.g. half-form vs. full-conjunct rendering) and stripping them
# can change a word's correct written form, not just its whitespace.
_ZERO_WIDTH_AND_CONTROL = re.compile(
    "[‎‏﻿\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)
_INLINE_WHITESPACE_RUN = re.compile(r"[ \t]+")
_BLANK_LINE_RUN = re.compile(r"\n{3,}")
_NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE_RUN = re.compile(r"\s+")
_REPLACEMENT_CHAR = "�"
# A token counts as "non-language" only if it is PURE Latin script (letters
# plus internal hyphen/apostrophe and trailing punctuation) with no digits and
# no Devanagari - e.g. "IRCTC", "Delhi-NCR", "Musk's". Deliberately narrow:
# digits stay per the course requirement, and mixed tokens (e.g. "5G",
# "iPhone14") are left alone since they're not pure foreign-script noise.
_PURE_LATIN_TOKEN = re.compile(r"^[A-Za-z]+(?:[.\-'][A-Za-z]+)*[.,!?;:\"'()\[\]]*$")

DEVANAGARI_BLOCK = (0x0900, 0x097F)

# Lexical markers for the Hindi/Bhojpuri cross-contamination check. Both
# languages are Devanagari and lexically close, so script-ratio alone (see
# devanagari_ratio) cannot tell them apart - a Hindi document isn't "wrong
# script," it's wrong language. These are common function words/verb endings
# that differ between the two (Bhojpuri: बा/बाटे/बाड़ें/बानी copula forms,
# एगो/गइल/भइल/कइल etc.; Hindi: है/हैं copula, एक, गया, हुआ, किया). This is a
# coarse signal for REPORTING contamination in both directions, not a hard
# pass/fail filter - a short, genuine document may simply not contain any of
# these specific words, and that must not be read as evidence of the wrong
# language.
BHOJPURI_MARKERS = [
    "बा", "बाटे", "बाड़ें", "बानी", "एगो", "गइल", "भइल", "कइल",
    "रउआ", "रउरा", "तोहार", "हमनी", "जवन", "खातिर", "करे ला",
]
HINDI_MARKERS = ["है", "हैं", "एक", "गया", "हुआ", "किया", "आप", "तुम्हारा"]


def marker_counts(text: str) -> tuple[int, int]:
    """Returns (bhojpuri_marker_hits, hindi_marker_hits) - raw occurrence
    counts of the word lists above, word-boundary matched."""
    bho = sum(len(re.findall(rf"(?<!\w){re.escape(m)}(?!\w)", text)) for m in BHOJPURI_MARKERS)
    hin = sum(len(re.findall(rf"(?<!\w){re.escape(m)}(?!\w)", text)) for m in HINDI_MARKERS)
    return bho, hin


def normalize_text(text: str) -> str:
    """NFC-normalize and strip zero-width/control characters only (preserves
    Devanagari and all other real text, per phase1_strategy.md section 6)."""
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH_AND_CONTROL.sub("", text)
    text = _INLINE_WHITESPACE_RUN.sub(" ", text)
    text = _BLANK_LINE_RUN.sub("\n\n", text)
    return text.strip()


def devanagari_ratio(text: str) -> float:
    """Fraction of non-whitespace characters in the Devanagari Unicode block."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    dev = sum(1 for c in chars if DEVANAGARI_BLOCK[0] <= ord(c) <= DEVANAGARI_BLOCK[1])
    return dev / len(chars)


def malformed_ratio(text: str) -> float:
    """Fraction of characters that are the Unicode replacement char (decoding
    failures / mojibake)."""
    if not text:
        return 1.0
    return text.count(_REPLACEMENT_CHAR) / len(text)


def is_too_short(text: str, min_chars: int) -> bool:
    return len(text) < min_chars


def langdetect_verdict(text: str, expected_iso: str, sample_chars: int = 3000, min_prob: float = 0.85):
    """Returns True/False if langdetect can render a confident verdict, or None
    if it can't (too little/ambiguous text, or - notably - if langdetect has no
    label for the expected language at all, e.g. Bhojpuri ("bho") isn't in
    langdetect's supported-language list; callers must not trust a None as
    either pass or fail)."""
    if detect_langs is None:
        return None
    sample = text[:sample_chars]
    try:
        results = detect_langs(sample)
    except LangDetectException:
        return None
    if not results:
        return None
    top = results[0]
    return top.lang == expected_iso and top.prob >= min_prob


def strip_latin_tokens(text: str) -> tuple[str, int]:
    """Remove whitespace-delimited tokens that are purely Latin-script (course
    requirement: strip non-language words, digits stay). Devanagari tokens,
    digit tokens, and mixed-script/alphanumeric tokens are all left untouched.
    Line structure is preserved; a line that was entirely a removed token
    becomes an empty line (harmless - normalize_text's blank-line collapsing
    handles runs of these). Returns (cleaned_text, tokens_removed)."""
    removed = 0
    out_lines = []
    for line in text.split("\n"):
        kept = []
        for tok in line.split(" "):
            if tok and _PURE_LATIN_TOKEN.match(tok):
                removed += 1
                continue
            kept.append(tok)
        out_lines.append(" ".join(kept))
    return "\n".join(out_lines), removed


def dedup_key(text: str) -> str:
    """Hash of whitespace-collapsed, punctuation-stripped text. This is what
    phase1_strategy.md section 6 means by 'near-duplicate removal (normalized-
    text hash)' - it catches exact dupes and formatting/whitespace variants of
    the same document, not fuzzy/paraphrase similarity."""
    key = _WHITESPACE_RUN.sub(" ", text).strip()
    key = _NON_WORD.sub("", key).lower()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def read_metadata(meta_path: Path):
    if not meta_path.exists():
        return
    with meta_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Two crawler processes appending to the same metadata.jsonl
                # concurrently can interleave writes mid-line (no file lock on
                # the shared append). Skip the casualty instead of crashing
                # the whole cleaning run over one malformed record.
                print(f"  [warn] {meta_path}:{lineno} malformed JSON line, skipping")


STALE_LOCK_SECONDS = 15 * 60  # a lock this old means its owner crashed/was killed, not that it's slow


def acquire_single_instance_lock(lock_path: Path, retries: int = 5, wait: float = 0.2) -> None:
    """Refuse to run if another process already holds this lock. Background
    launches in this environment have been observed to spawn two OS processes
    per logical command - without this guard, two clean_*.py instances for
    the same language would both write to data/interim/ and data/cleaned/
    concurrently and corrupt them the same way concurrent crawlers corrupted
    metadata.jsonl (see scripts/crawl_sites.py site_lock).

    Uses os.O_CREAT | os.O_EXCL, which is an atomic create-if-absent at the
    OS level, not a check-then-write (an earlier version of this function did
    `if lock_path.exists(): ... else: write_text(...)`, which is two separate
    syscalls - the two duplicate-launch processes above can both pass the
    `exists()` check before either has written the file, so both proceed and
    both reset+rewrite data/interim/data/cleaned concurrently; this is exactly
    what happened during Phase 1 clean_hindi.py development. O_EXCL closes
    that window: only one of the two racing opens can succeed).

    PID-tagged with a staleness timeout so a genuinely crashed/killed owner
    doesn't permanently block reruns. Plain acquire/release (not a context
    manager) so callers don't need to re-indent an existing function body
    into a `with` block; release_single_instance_lock() is best-effort
    cleanup - if the process dies before calling it, the staleness timeout
    reclaims the lock instead."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(retries):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            os.close(fd)
            return
        except (FileExistsError, PermissionError):
            # Windows can raise PermissionError instead of FileExistsError
            # under create/delete contention - both mean "lock already exists."
            if not lock_path.exists():
                time.sleep(wait)
                continue
            age = time.time() - lock_path.stat().st_mtime
            if age < STALE_LOCK_SECONDS:
                raise SystemExit(
                    f"another process already holds {lock_path} (age {age:.0f}s) - "
                    f"refusing to run concurrently. If that process is dead, delete "
                    f"the lock file and rerun."
                )
            print(f"stale lock (age {age:.0f}s) from a crashed run, reclaiming {lock_path}")
            lock_path.unlink(missing_ok=True)
    raise SystemExit(f"could not acquire {lock_path} after {retries} attempts")


def release_single_instance_lock(lock_path: Path) -> None:
    lock_path.unlink(missing_ok=True)


def recompute_disk_truth_stats(stats: dict, interim_dir: Path, cleaned_dir: Path, dropped_log_path: Path) -> dict:
    """Overwrite the doc/char counters in `stats` with values recomputed
    directly from data/interim/, data/cleaned/, and the dropped-document log
    on disk, rather than the incrementally-tracked in-memory counters.
    Across several --resume cycles (this project has had a few, from
    background jobs getting killed by session teardowns), the carried-forward
    in-memory counters drifted from reality - docs_in/docs_cleaned printed
    after one such run were ~1.5-2x the true unique document count, even
    though the actual output files had zero duplicate document_ids. The
    output files and the (synchronously, durably appended) dropped-log are
    the correct sources of truth; only the summary counters need correction.
    chars_in is left as tracked (not recoverable exactly from disk, since raw
    pre-normalization character counts aren't persisted anywhere) - approximate."""
    interim_n = interim_c = 0
    for meta in read_metadata(interim_dir / "metadata.jsonl"):
        interim_n += 1
        interim_c += meta.get("characters", 0)
    cleaned_n = cleaned_c = 0
    for meta in read_metadata(cleaned_dir / "metadata.jsonl"):
        cleaned_n += 1
        cleaned_c += meta.get("characters", 0)

    dropped_reason_by_id = {}
    if dropped_log_path.exists():
        with dropped_log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    dropped_reason_by_id[rec["document_id"]] = rec["reason"]
                except (json.JSONDecodeError, KeyError):
                    continue
    dropped_counts = {}
    for reason in dropped_reason_by_id.values():
        dropped_counts[reason] = dropped_counts.get(reason, 0) + 1

    stats["docs_in"] = interim_n + dropped_counts.get("malformed_unicode", 0) + dropped_counts.get("too_short", 0)
    stats["docs_cleaned"] = cleaned_n
    stats["chars_cleaned"] = cleaned_c
    stats["docs_interim"] = interim_n
    stats["chars_interim"] = interim_c
    stats["dropped_malformed"] = dropped_counts.get("malformed_unicode", 0)
    stats["dropped_short"] = dropped_counts.get("too_short", 0)
    stats["dropped_duplicate"] = dropped_counts.get("duplicate", 0)
    for reason in list(stats.get("dropped_language", {})):
        stats["dropped_language"][reason] = dropped_counts.get(reason, 0)
    return stats


def reset_dir(out_dir: Path) -> None:
    """Clear a stage output directory (text/ + metadata.jsonl) so pipeline
    stages can be re-run from scratch rather than silently appending to stale
    output."""
    text_dir = out_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    for old in text_dir.glob("*.txt"):
        old.unlink()
    meta_path = out_dir / "metadata.jsonl"
    if meta_path.exists():
        meta_path.unlink()


def load_resume_state(interim_dir: Path, cleaned_dir: Path, dropped_log_path: Path, stats_path: Path):
    """Reconstruct in-memory state from a previous, interrupted clean_*.py run
    so --resume can continue from where it stopped instead of reprocessing the
    full raw corpus from doc 0. Cost is proportional to the partial run's
    output (interim + cleaned + dropped-log), not the full raw input, so this
    is cheap even when the interruption happened most of the way through -
    this environment has been observed to kill background jobs on session
    teardown well before a multi-hour clean run finishes.

    Returns (processed_ids, seen_hashes, stats_or_none).
    """
    processed_ids = set()
    for meta in read_metadata(interim_dir / "metadata.jsonl"):
        processed_ids.add(meta["document_id"])
    if dropped_log_path.exists():
        with dropped_log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    processed_ids.add(json.loads(line)["document_id"])
                except (json.JSONDecodeError, KeyError):
                    continue

    # seen_hashes must be recomputed from the actual cleaned text (not stored
    # anywhere) because dedup_key is a pure function of post-clean text -
    # rereading data/cleaned/ reproduces exactly the hashes the interrupted
    # run had in memory when it died.
    # Prefer a dedup_key stored directly in the metadata line (fast: one
    # sequential read of a single file). Falls back to reopening the
    # individual text file only for older records written before this field
    # existed - reopening hundreds of thousands of individual small files one
    # at a time is what made resume unusably slow in practice (100 minutes
    # wall clock, 6 minutes of actual CPU, on a resume against a ~440k-doc
    # cleaned corpus - disk I/O for that many individual file opens, not CPU,
    # was the bottleneck).
    seen_hashes = set()
    cleaned_text_dir = cleaned_dir / "text"
    for meta in read_metadata(cleaned_dir / "metadata.jsonl"):
        key = meta.get("dedup_key")
        if key is None:
            text_path = cleaned_text_dir / f"{meta['document_id']}.txt"
            if text_path.exists():
                key = dedup_key(text_path.read_text(encoding="utf-8"))
        if key is not None:
            seen_hashes.add(key)

    stats = None
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            stats = None

    return processed_ids, seen_hashes, stats


def write_doc(out_dir: Path, doc_id: str, text: str, meta: dict) -> None:
    (out_dir / "text").mkdir(parents=True, exist_ok=True)
    (out_dir / "text" / f"{doc_id}.txt").write_text(text, encoding="utf-8")
    with (out_dir / "metadata.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

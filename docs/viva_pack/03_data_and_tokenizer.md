# Phase 1 - cleaning, splitting, tokenizer

Part of the LMA individual project viva pack. Line numbers below match the files in the repository exactly, so any line cited here can be opened directly in the IDE.

Files in this pack:

- `scripts/clean_lib.py`
- `hindi/data/pipeline/clean_hindi.py`
- `hindi/data/pipeline/split_hindi.py`
- `hindi/tokenizer/train_tokenizer.py`
- `hindi/tokenizer/analyze_tokenizer.py`


## scripts/clean_lib.py

```python
   1  """Shared cleaning/normalization/dedup logic for the Hindi and Bhojpuri pipelines.
   2  
   3  Shared code only - each language's raw/interim/cleaned/splits data stays in its own
   4  directory tree and is never mixed here. Implements the pipeline described in
   5  report/phase1/phase1_strategy.md section 6:
   6  
   7    raw text -> NFC normalize + strip zero-width/control chars -> language-ID filter
   8    -> boilerplate/short-doc/malformed-Unicode removal -> exact + near-duplicate removal
   9  """
  10  
  11  import hashlib
  12  import json
  13  import os
  14  import re
  15  import time
  16  import unicodedata
  17  from pathlib import Path
  18  
  19  try:
  20      from langdetect import DetectorFactory, LangDetectException, detect_langs
  21  
  22      DetectorFactory.seed = 0
  23  except ImportError:
  24      detect_langs = None
  25      LangDetectException = Exception
  26  
  27  # Zero-width and control characters that are NOT meaningful in Devanagari text:
  28  # LRM/RLM (‎/‏, directional marks, unused in Devanagari), BOM
  29  # (﻿), and C0/DEL control codes. Deliberately excludes ZWNJ (‌) and
  30  # ZWJ (‍) - those control conjunct ligature formation in real Devanagari
  31  # orthography (e.g. half-form vs. full-conjunct rendering) and stripping them
  32  # can change a word's correct written form, not just its whitespace.
  33  _ZERO_WIDTH_AND_CONTROL = re.compile(
  34      "[‎‏﻿\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
  35  )
  36  _INLINE_WHITESPACE_RUN = re.compile(r"[ \t]+")
  37  _BLANK_LINE_RUN = re.compile(r"\n{3,}")
  38  _NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)
  39  _WHITESPACE_RUN = re.compile(r"\s+")
  40  _REPLACEMENT_CHAR = "�"
  41  # A token counts as "non-language" only if it is PURE Latin script (letters
  42  # plus internal hyphen/apostrophe and trailing punctuation) with no digits and
  43  # no Devanagari - e.g. "IRCTC", "Delhi-NCR", "Musk's". Deliberately narrow:
  44  # digits stay per the course requirement, and mixed tokens (e.g. "5G",
  45  # "iPhone14") are left alone since they're not pure foreign-script noise.
  46  _PURE_LATIN_TOKEN = re.compile(r"^[A-Za-z]+(?:[.\-'][A-Za-z]+)*[.,!?;:\"'()\[\]]*$")
  47  
  48  DEVANAGARI_BLOCK = (0x0900, 0x097F)
  49  
  50  # Lexical markers for the Hindi/Bhojpuri cross-contamination check. Both
  51  # languages are Devanagari and lexically close, so script-ratio alone (see
  52  # devanagari_ratio) cannot tell them apart - a Hindi document isn't "wrong
  53  # script," it's wrong language. These are common function words/verb endings
  54  # that differ between the two (Bhojpuri: बा/बाटे/बाड़ें/बानी copula forms,
  55  # एगो/गइल/भइल/कइल etc.; Hindi: है/हैं copula, एक, गया, हुआ, किया). This is a
  56  # coarse signal for REPORTING contamination in both directions, not a hard
  57  # pass/fail filter - a short, genuine document may simply not contain any of
  58  # these specific words, and that must not be read as evidence of the wrong
  59  # language.
  60  BHOJPURI_MARKERS = [
  61      "बा", "बाटे", "बाड़ें", "बानी", "एगो", "गइल", "भइल", "कइल",
  62      "रउआ", "रउरा", "तोहार", "हमनी", "जवन", "खातिर", "करे ला",
  63  ]
  64  HINDI_MARKERS = ["है", "हैं", "एक", "गया", "हुआ", "किया", "आप", "तुम्हारा"]
  65  
  66  
  67  def marker_counts(text: str) -> tuple[int, int]:
  68      """Returns (bhojpuri_marker_hits, hindi_marker_hits) - raw occurrence
  69      counts of the word lists above, word-boundary matched."""
  70      bho = sum(len(re.findall(rf"(?<!\w){re.escape(m)}(?!\w)", text)) for m in BHOJPURI_MARKERS)
  71      hin = sum(len(re.findall(rf"(?<!\w){re.escape(m)}(?!\w)", text)) for m in HINDI_MARKERS)
  72      return bho, hin
  73  
  74  
  75  def normalize_text(text: str) -> str:
  76      """NFC-normalize and strip zero-width/control characters only (preserves
  77      Devanagari and all other real text, per phase1_strategy.md section 6)."""
  78      text = unicodedata.normalize("NFC", text)
  79      text = _ZERO_WIDTH_AND_CONTROL.sub("", text)
  80      text = _INLINE_WHITESPACE_RUN.sub(" ", text)
  81      text = _BLANK_LINE_RUN.sub("\n\n", text)
  82      return text.strip()
  83  
  84  
  85  def devanagari_ratio(text: str) -> float:
  86      """Fraction of non-whitespace characters in the Devanagari Unicode block."""
  87      chars = [c for c in text if not c.isspace()]
  88      if not chars:
  89          return 0.0
  90      dev = sum(1 for c in chars if DEVANAGARI_BLOCK[0] <= ord(c) <= DEVANAGARI_BLOCK[1])
  91      return dev / len(chars)
  92  
  93  
  94  def malformed_ratio(text: str) -> float:
  95      """Fraction of characters that are the Unicode replacement char (decoding
  96      failures / mojibake)."""
  97      if not text:
  98          return 1.0
  99      return text.count(_REPLACEMENT_CHAR) / len(text)
 100  
 101  
 102  def is_too_short(text: str, min_chars: int) -> bool:
 103      """WHY: a tiny document (a stub, a nav-menu fragment that survived
 104      extraction, a truncated fetch) contributes noise, not signal, and
 105      langdetect is also unreliable on very short text - filtering by length
 106      before language-ID avoids wasting a langdetect call on text too short
 107      to classify meaningfully anyway.
 108      INPUT: text (post-normalization); min_chars threshold (per-language,
 109      e.g. MIN_CHARS in clean_hindi.py/clean_bhojpuri.py).
 110      OUTPUT: True if text should be dropped as too short."""
 111      return len(text) < min_chars
 112  
 113  
 114  def langdetect_verdict(text: str, expected_iso: str, sample_chars: int = 3000, min_prob: float = 0.85):
 115      """Returns True/False if langdetect can render a confident verdict, or None
 116      if it can't (too little/ambiguous text, or - notably - if langdetect has no
 117      label for the expected language at all, e.g. Bhojpuri ("bho") isn't in
 118      langdetect's supported-language list; callers must not trust a None as
 119      either pass or fail)."""
 120      if detect_langs is None:
 121          return None
 122      sample = text[:sample_chars]
 123      try:
 124          results = detect_langs(sample)
 125      except LangDetectException:
 126          return None
 127      if not results:
 128          return None
 129      top = results[0]
 130      return top.lang == expected_iso and top.prob >= min_prob
 131  
 132  
 133  def strip_latin_tokens(text: str) -> tuple[str, int]:
 134      """Remove whitespace-delimited tokens that are purely Latin-script (course
 135      requirement: strip non-language words, digits stay). Devanagari tokens,
 136      digit tokens, and mixed-script/alphanumeric tokens are all left untouched.
 137      Line structure is preserved; a line that was entirely a removed token
 138      becomes an empty line (harmless - normalize_text's blank-line collapsing
 139      handles runs of these). Returns (cleaned_text, tokens_removed)."""
 140      removed = 0
 141      out_lines = []
 142      for line in text.split("\n"):
 143          kept = []
 144          for tok in line.split(" "):
 145              if tok and _PURE_LATIN_TOKEN.match(tok):
 146                  removed += 1
 147                  continue
 148              kept.append(tok)
 149          out_lines.append(" ".join(kept))
 150      return "\n".join(out_lines), removed
 151  
 152  
 153  def dedup_key(text: str) -> str:
 154      """Hash of whitespace-collapsed, punctuation-stripped text. This is what
 155      phase1_strategy.md section 6 means by 'near-duplicate removal (normalized-
 156      text hash)' - it catches exact dupes and formatting/whitespace variants of
 157      the same document, not fuzzy/paraphrase similarity."""
 158      key = _WHITESPACE_RUN.sub(" ", text).strip()
 159      key = _NON_WORD.sub("", key).lower()
 160      return hashlib.sha1(key.encode("utf-8")).hexdigest()
 161  
 162  
 163  def read_metadata(meta_path: Path):
 164      """WHY: this is clean_hindi.py/clean_bhojpuri.py's primary input reader,
 165      so it must tolerate the one known real-world failure mode (an
 166      interleaved-write malformed line from concurrent crawler appends,
 167      see the inline comment below) rather than crash a multi-hour clean run
 168      over a single bad line.
 169      INPUT: path to a raw/{manual,downloaded}/metadata.jsonl file (may not
 170      exist, e.g. a language with no downloaded sources yet).
 171      OUTPUT: generator yielding each valid line's parsed dict."""
 172      if not meta_path.exists():
 173          return
 174      with meta_path.open("r", encoding="utf-8") as f:
 175          for lineno, line in enumerate(f, 1):
 176              line = line.strip()
 177              if not line:
 178                  continue
 179              try:
 180                  yield json.loads(line)
 181              except json.JSONDecodeError:
 182                  # Two crawler processes appending to the same metadata.jsonl
 183                  # concurrently can interleave writes mid-line (no file lock on
 184                  # the shared append). Skip the casualty instead of crashing
 185                  # the whole cleaning run over one malformed record.
 186                  print(f"  [warn] {meta_path}:{lineno} malformed JSON line, skipping")
 187  
 188  
 189  STALE_LOCK_SECONDS = 15 * 60  # a lock this old means its owner crashed/was killed, not that it's slow
 190  
 191  
 192  def acquire_single_instance_lock(lock_path: Path, retries: int = 5, wait: float = 0.2) -> None:
 193      """Refuse to run if another process already holds this lock. Background
 194      launches in this environment have been observed to spawn two OS processes
 195      per logical command - without this guard, two clean_*.py instances for
 196      the same language would both write to data/interim/ and data/cleaned/
 197      concurrently and corrupt them the same way concurrent crawlers corrupted
 198      metadata.jsonl (see scripts/crawl_sites.py site_lock).
 199  
 200      Uses os.O_CREAT | os.O_EXCL, which is an atomic create-if-absent at the
 201      OS level, not a check-then-write (an earlier version of this function did
 202      `if lock_path.exists(): ... else: write_text(...)`, which is two separate
 203      syscalls - the two duplicate-launch processes above can both pass the
 204      `exists()` check before either has written the file, so both proceed and
 205      both reset+rewrite data/interim/data/cleaned concurrently; this is exactly
 206      what happened during Phase 1 clean_hindi.py development. O_EXCL closes
 207      that window: only one of the two racing opens can succeed).
 208  
 209      PID-tagged with a staleness timeout so a genuinely crashed/killed owner
 210      doesn't permanently block reruns. Plain acquire/release (not a context
 211      manager) so callers don't need to re-indent an existing function body
 212      into a `with` block; release_single_instance_lock() is best-effort
 213      cleanup - if the process dies before calling it, the staleness timeout
 214      reclaims the lock instead."""
 215      lock_path.parent.mkdir(parents=True, exist_ok=True)
 216      for _ in range(retries):
 217          try:
 218              fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
 219              os.write(fd, str(os.getpid()).encode("utf-8"))
 220              os.close(fd)
 221              return
 222          except (FileExistsError, PermissionError):
 223              # Windows can raise PermissionError instead of FileExistsError
 224              # under create/delete contention - both mean "lock already exists."
 225              if not lock_path.exists():
 226                  time.sleep(wait)
 227                  continue
 228              age = time.time() - lock_path.stat().st_mtime
 229              if age < STALE_LOCK_SECONDS:
 230                  raise SystemExit(
 231                      f"another process already holds {lock_path} (age {age:.0f}s) - "
 232                      f"refusing to run concurrently. If that process is dead, delete "
 233                      f"the lock file and rerun."
 234                  )
 235              print(f"stale lock (age {age:.0f}s) from a crashed run, reclaiming {lock_path}")
 236              lock_path.unlink(missing_ok=True)
 237      raise SystemExit(f"could not acquire {lock_path} after {retries} attempts")
 238  
 239  
 240  def release_single_instance_lock(lock_path: Path) -> None:
 241      """WHY: pairs with acquire_single_instance_lock; best-effort only
 242      because if the owning process dies before calling this, the lock's
 243      staleness timeout (STALE_LOCK_SECONDS) reclaims it instead - callers
 244      don't need a try/finally around every possible crash path.
 245      INPUT: lock_path (the same path passed to acquire).
 246      OUTPUT: none; missing_ok=True means calling this twice, or on a lock
 247      that was never acquired, is not an error."""
 248      lock_path.unlink(missing_ok=True)
 249  
 250  
 251  def recompute_disk_truth_stats(stats: dict, interim_dir: Path, cleaned_dir: Path, dropped_log_path: Path) -> dict:
 252      """Overwrite the doc/char counters in `stats` with values recomputed
 253      directly from data/interim/, data/cleaned/, and the dropped-document log
 254      on disk, rather than the incrementally-tracked in-memory counters.
 255      Across several --resume cycles (this project has had a few, from
 256      background jobs getting killed by session teardowns), the carried-forward
 257      in-memory counters drifted from reality - docs_in/docs_cleaned printed
 258      after one such run were ~1.5-2x the true unique document count, even
 259      though the actual output files had zero duplicate document_ids. The
 260      output files and the (synchronously, durably appended) dropped-log are
 261      the correct sources of truth; only the summary counters need correction.
 262      chars_in is left as tracked (not recoverable exactly from disk, since raw
 263      pre-normalization character counts aren't persisted anywhere) - approximate."""
 264      interim_n = interim_c = 0
 265      for meta in read_metadata(interim_dir / "metadata.jsonl"):
 266          interim_n += 1
 267          interim_c += meta.get("characters", 0)
 268      cleaned_n = cleaned_c = 0
 269      for meta in read_metadata(cleaned_dir / "metadata.jsonl"):
 270          cleaned_n += 1
 271          cleaned_c += meta.get("characters", 0)
 272  
 273      dropped_reason_by_id = {}
 274      if dropped_log_path.exists():
 275          with dropped_log_path.open("r", encoding="utf-8") as f:
 276              for line in f:
 277                  line = line.strip()
 278                  if not line:
 279                      continue
 280                  try:
 281                      rec = json.loads(line)
 282                      dropped_reason_by_id[rec["document_id"]] = rec["reason"]
 283                  except (json.JSONDecodeError, KeyError):
 284                      continue
 285      dropped_counts = {}
 286      for reason in dropped_reason_by_id.values():
 287          dropped_counts[reason] = dropped_counts.get(reason, 0) + 1
 288  
 289      stats["docs_in"] = interim_n + dropped_counts.get("malformed_unicode", 0) + dropped_counts.get("too_short", 0)
 290      stats["docs_cleaned"] = cleaned_n
 291      stats["chars_cleaned"] = cleaned_c
 292      stats["docs_interim"] = interim_n
 293      stats["chars_interim"] = interim_c
 294      stats["dropped_malformed"] = dropped_counts.get("malformed_unicode", 0)
 295      stats["dropped_short"] = dropped_counts.get("too_short", 0)
 296      stats["dropped_duplicate"] = dropped_counts.get("duplicate", 0)
 297      for reason in list(stats.get("dropped_language", {})):
 298          stats["dropped_language"][reason] = dropped_counts.get(reason, 0)
 299      return stats
 300  
 301  
 302  def reset_dir(out_dir: Path) -> None:
 303      """Clear a stage output directory (text/ + metadata.jsonl) so pipeline
 304      stages can be re-run from scratch rather than silently appending to stale
 305      output."""
 306      text_dir = out_dir / "text"
 307      text_dir.mkdir(parents=True, exist_ok=True)
 308      for old in text_dir.glob("*.txt"):
 309          old.unlink()
 310      meta_path = out_dir / "metadata.jsonl"
 311      if meta_path.exists():
 312          meta_path.unlink()
 313  
 314  
 315  def load_resume_state(interim_dir: Path, cleaned_dir: Path, dropped_log_path: Path, stats_path: Path):
 316      """Reconstruct in-memory state from a previous, interrupted clean_*.py run
 317      so --resume can continue from where it stopped instead of reprocessing the
 318      full raw corpus from doc 0. Cost is proportional to the partial run's
 319      output (interim + cleaned + dropped-log), not the full raw input, so this
 320      is cheap even when the interruption happened most of the way through -
 321      this environment has been observed to kill background jobs on session
 322      teardown well before a multi-hour clean run finishes.
 323  
 324      Returns (processed_ids, seen_hashes, stats_or_none).
 325      """
 326      processed_ids = set()
 327      for meta in read_metadata(interim_dir / "metadata.jsonl"):
 328          processed_ids.add(meta["document_id"])
 329      if dropped_log_path.exists():
 330          with dropped_log_path.open("r", encoding="utf-8") as f:
 331              for line in f:
 332                  line = line.strip()
 333                  if not line:
 334                      continue
 335                  try:
 336                      processed_ids.add(json.loads(line)["document_id"])
 337                  except (json.JSONDecodeError, KeyError):
 338                      continue
 339  
 340      # seen_hashes must be recomputed from the actual cleaned text (not stored
 341      # anywhere) because dedup_key is a pure function of post-clean text -
 342      # rereading data/cleaned/ reproduces exactly the hashes the interrupted
 343      # run had in memory when it died.
 344      # Prefer a dedup_key stored directly in the metadata line (fast: one
 345      # sequential read of a single file). Falls back to reopening the
 346      # individual text file only for older records written before this field
 347      # existed - reopening hundreds of thousands of individual small files one
 348      # at a time is what made resume unusably slow in practice (100 minutes
 349      # wall clock, 6 minutes of actual CPU, on a resume against a ~440k-doc
 350      # cleaned corpus - disk I/O for that many individual file opens, not CPU,
 351      # was the bottleneck).
 352      seen_hashes = set()
 353      cleaned_text_dir = cleaned_dir / "text"
 354      for meta in read_metadata(cleaned_dir / "metadata.jsonl"):
 355          key = meta.get("dedup_key")
 356          if key is None:
 357              text_path = cleaned_text_dir / f"{meta['document_id']}.txt"
 358              if text_path.exists():
 359                  key = dedup_key(text_path.read_text(encoding="utf-8"))
 360          if key is not None:
 361              seen_hashes.add(key)
 362  
 363      stats = None
 364      if stats_path.exists():
 365          try:
 366              stats = json.loads(stats_path.read_text(encoding="utf-8"))
 367          except json.JSONDecodeError:
 368              stats = None
 369  
 370      return processed_ids, seen_hashes, stats
 371  
 372  
 373  def write_doc(out_dir: Path, doc_id: str, text: str, meta: dict) -> None:
 374      """WHY: clean_hindi.py/clean_bhojpuri.py write to interim/ and cleaned/
 375      single-threaded (no concurrent-writer risk the way raw/ ingestion
 376      scripts have, since only one clean_*.py instance holds the stage's
 377      lock at a time - see acquire_single_instance_lock), so a plain append
 378      is safe here, unlike append_metadata_locked used elsewhere.
 379      INPUT: out_dir (interim/ or cleaned/); doc_id; text (post-processing
 380      for this stage); meta (dict, cleaning_status field distinguishes
 381      interim vs cleaned).
 382      OUTPUT: none (disk side effects only)."""
 383      (out_dir / "text").mkdir(parents=True, exist_ok=True)
 384      (out_dir / "text" / f"{doc_id}.txt").write_text(text, encoding="utf-8")
 385      with (out_dir / "metadata.jsonl").open("a", encoding="utf-8") as f:
 386          f.write(json.dumps(meta, ensure_ascii=False) + "\n")
```


## hindi/data/pipeline/clean_hindi.py

```python
   1  """Clean Hindi raw text (downloaded + manual) into data/interim/ and data/cleaned/.
   2  
   3  Pipeline stage per report/phase1/phase1_strategy.md section 6 (plus the
   4  non-language-word removal stage added per course ruling - digits stay):
   5    raw text -> NFC normalize + strip zero-width/control chars -> strip pure-
   6    Latin-script tokens -> malformed-Unicode and short-doc removal ->
   7    [data/interim/ snapshot here] -> language-ID filter -> exact + near-duplicate
   8    removal -> data/cleaned/
   9  
  10  Run after download_hindi.py (and ingest_manual.py, once manual data exists).
  11  Always rebuilds data/interim/ and data/cleaned/ from data/raw/ rather than
  12  appending, so it is safe to re-run whenever raw data changes.
  13  """
  14  
  15  import argparse
  16  import json
  17  import sys
  18  from pathlib import Path
  19  
  20  sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
  21  from clean_lib import (  # noqa: E402
  22      acquire_single_instance_lock,
  23      dedup_key,
  24      devanagari_ratio,
  25      is_too_short,
  26      langdetect_verdict,
  27      load_resume_state,
  28      malformed_ratio,
  29      marker_counts,
  30      normalize_text,
  31      read_metadata,
  32      recompute_disk_truth_stats,
  33      release_single_instance_lock,
  34      reset_dir,
  35      strip_latin_tokens,
  36      write_doc,
  37  )
  38  
  39  CHECKPOINT_EVERY = 2000
  40  LOCK_PATH = Path(__file__).resolve().parents[1] / "metadata" / ".clean.lock"
  41  
  42  LANG = "hindi"
  43  LANG_ISO = "hi"
  44  DATA_DIR = Path(__file__).resolve().parents[1]
  45  # Manual processed BEFORE downloaded: dedup below keeps first-seen and drops
  46  # later duplicates, so this ordering means a manual copy always wins a
  47  # cross-source duplicate tie against a downloaded copy of the same text
  48  # (e.g. a scraped article that also appears in a bulk corpus), never the
  49  # reverse. Never share this ordering assumption without this comment - it's
  50  # load-bearing for the dedup tie-break rule, not incidental.
  51  RAW_DIRS = [DATA_DIR / "raw" / "manual", DATA_DIR / "raw" / "downloaded"]
  52  INTERIM_DIR = DATA_DIR / "interim"
  53  CLEANED_DIR = DATA_DIR / "cleaned"
  54  STATS_PATH = DATA_DIR / "metadata" / "clean_stats.json"
  55  DROPPED_LOG_PATH = DATA_DIR / "metadata" / "dropped_documents.jsonl"
  56  
  57  MIN_CHARS = 200
  58  MIN_DEVANAGARI_RATIO = 0.5
  59  MAX_MALFORMED_RATIO = 0.01
  60  LANGDETECT_MIN_PROB = 0.85
  61  
  62  
  63  def language_ok(text: str) -> tuple[bool, str]:
  64      """Hindi has full langdetect support, so trust its verdict as the primary
  65      check. Devanagari-ratio is a cheap, script-level pre-filter.
  66  
  67      langdetect has no 'bho' label, so Bhojpuri text - also Devanagari, with
  68      heavy vocabulary overlap with Hindi - gets misclassified as 'hi' by
  69      langdetect and would otherwise pass straight into the Hindi corpus. That
  70      violates the course's no-shared-documents-across-languages rule, not just
  71      a style/quality issue, so it must be rejected here rather than merely
  72      logged. The lexical marker check runs before langdetect for that reason:
  73      a document that reads as Bhojpuri should never reach the Hindi corpus
  74      regardless of what langdetect would have said about it."""
  75      if devanagari_ratio(text) < MIN_DEVANAGARI_RATIO:
  76          return False, "low_devanagari_ratio"
  77      bho_hits, hin_hits = marker_counts(text)
  78      if bho_hits >= 2 and bho_hits > hin_hits:
  79          return False, "likely_bhojpuri"
  80      verdict = langdetect_verdict(text, LANG_ISO, min_prob=LANGDETECT_MIN_PROB)
  81      if verdict is False:
  82          return False, "langdetect_mismatch"
  83      # verdict is None -> langdetect couldn't decide (too short/ambiguous text);
  84      # fall back to the Devanagari-ratio pass above instead of dropping it.
  85      return True, ""
  86  
  87  
  88  def main():
  89      """WHY: the single entry point that turns raw/{manual,downloaded}/ into
  90      cleaned/ via the pipeline documented in the module docstring
  91      (normalize -> strip Latin tokens -> malformed/short-doc drop ->
  92      language-ID + Bhojpuri-contamination filter -> dedup). Without
  93      --resume it always rebuilds interim/ and cleaned/ from scratch (see
  94      reset_dir in clean_lib.py) so a rerun after raw/ changes is never
  95      silently stale.
  96      WHAT: iterates every raw doc (manual before downloaded, so a
  97      cross-source duplicate tie always keeps the manual copy - see the
  98      RAW_DIRS comment above), applies language_ok() and dedup_key()
  99      filtering, writes survivors to cleaned/, and recomputes final stats
 100      from disk (not carried-forward in-memory counters, which have drifted
 101      from reality across --resume cycles in the past).
 102      INPUT: CLI args (--min-chars, --resume) in; reads raw/manual/ and
 103      raw/downloaded/.
 104      OUTPUT: writes data/interim/, data/cleaned/, and
 105      metadata/clean_stats.json; no return value."""
 106      parser = argparse.ArgumentParser(description="Clean Hindi raw text (Model H)")
 107      parser.add_argument("--min-chars", type=int, default=MIN_CHARS)
 108      parser.add_argument("--resume", action="store_true",
 109                           help="Continue an interrupted run instead of rebuilding from "
 110                                "scratch: skip docs already in data/interim/ or the dropped "
 111                                "log, reload dedup hashes from data/cleaned/, and resume the "
 112                                "running stats counters.")
 113      args = parser.parse_args()
 114  
 115      acquire_single_instance_lock(LOCK_PATH)
 116  
 117      processed_ids = set()
 118      seen_hashes = set()
 119      resumed_stats = None
 120      if args.resume:
 121          processed_ids, seen_hashes, resumed_stats = load_resume_state(
 122              INTERIM_DIR, CLEANED_DIR, DROPPED_LOG_PATH, STATS_PATH)
 123          print(f"[{LANG}] resuming: {len(processed_ids)} docs already processed, "
 124                f"{len(seen_hashes)} dedup hashes reloaded")
 125      else:
 126          reset_dir(INTERIM_DIR)
 127          reset_dir(CLEANED_DIR)
 128          DROPPED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
 129          if DROPPED_LOG_PATH.exists():
 130              DROPPED_LOG_PATH.unlink()
 131  
 132      stats = {
 133          "docs_in": 0,
 134          "docs_cleaned": 0,
 135          "dropped_malformed": 0,
 136          "dropped_short": 0,
 137          "dropped_language": {
 138              "low_devanagari_ratio": 0,
 139              "likely_bhojpuri": 0,
 140              "langdetect_mismatch": 0,
 141          },
 142          "dropped_duplicate": 0,
 143          "chars_in": 0,
 144          "chars_cleaned": 0,
 145          "latin_tokens_removed": 0,
 146          "bhojpuri_reject_caveat": (
 147              "langdetect has no 'bho' label, so it cannot itself catch Bhojpuri text "
 148              "misclassified as Hindi (they share vocabulary and script). "
 149              "dropped_language.likely_bhojpuri is an active reject filter (see "
 150              "clean_lib.marker_counts / language_ok below), not just a count: any "
 151              "document where Bhojpuri lexical markers (बा/बाटे/बाड़ें/एगो/गइल/भइल/"
 152              "कइल/etc., >=2 hits) outnumber Hindi markers (है/हैं/एक/गया/हुआ/किया/etc.) "
 153              "is dropped before reaching data/cleaned/. This is a coarse lexical "
 154              "heuristic, not a language model, so it will not catch every case - "
 155              "documented as a known limitation, not a solved problem."
 156          ),
 157      }
 158      if resumed_stats is not None:
 159          stats.update(resumed_stats)
 160  
 161      for raw_dir in RAW_DIRS:
 162          meta_path = raw_dir / "metadata.jsonl"
 163          for meta in read_metadata(meta_path):
 164              doc_id = meta["document_id"]
 165              if doc_id in processed_ids:
 166                  continue
 167              stats["docs_in"] += 1
 168              if stats["docs_in"] % CHECKPOINT_EVERY == 0:
 169                  STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
 170                  STATS_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
 171              text_path = raw_dir / "text" / f"{doc_id}.txt"
 172              if not text_path.exists():
 173                  continue
 174              raw_text = text_path.read_text(encoding="utf-8", errors="ignore")
 175              stats["chars_in"] += len(raw_text)
 176  
 177              text = normalize_text(raw_text)
 178              text, n_removed = strip_latin_tokens(text)
 179              text = normalize_text(text)  # re-collapse whitespace left by removed tokens
 180              stats["latin_tokens_removed"] += n_removed
 181  
 182              if malformed_ratio(text) > MAX_MALFORMED_RATIO:
 183                  stats["dropped_malformed"] += 1
 184                  _log_dropped(doc_id, "malformed_unicode")
 185                  continue
 186              if is_too_short(text, args.min_chars):
 187                  stats["dropped_short"] += 1
 188                  _log_dropped(doc_id, "too_short")
 189                  continue
 190  
 191              # data/interim/ snapshot: normalized text that passed the cheap
 192              # structural checks, before language-ID and dedup are applied.
 193              interim_meta = dict(meta)
 194              interim_meta["characters"] = len(text)
 195              interim_meta["cleaning_status"] = "interim"
 196              write_doc(INTERIM_DIR, doc_id, text, interim_meta)
 197  
 198              ok, reason = language_ok(text)
 199              if not ok:
 200                  stats["dropped_language"][reason] = stats["dropped_language"].get(reason, 0) + 1
 201                  _log_dropped(doc_id, reason)
 202                  continue
 203  
 204              key = dedup_key(text)
 205              if key in seen_hashes:
 206                  stats["dropped_duplicate"] += 1
 207                  _log_dropped(doc_id, "duplicate")
 208                  continue
 209              seen_hashes.add(key)
 210  
 211              cleaned_meta = dict(meta)
 212              cleaned_meta["characters"] = len(text)
 213              cleaned_meta["cleaning_status"] = "cleaned"
 214              cleaned_meta["dedup_key"] = key
 215              write_doc(CLEANED_DIR, doc_id, text, cleaned_meta)
 216              stats["docs_cleaned"] += 1
 217              stats["chars_cleaned"] += len(text)
 218  
 219      stats = recompute_disk_truth_stats(stats, INTERIM_DIR, CLEANED_DIR, DROPPED_LOG_PATH)
 220      STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
 221      STATS_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
 222  
 223      print(f"[{LANG}] {stats['docs_cleaned']}/{stats['docs_in']} docs kept "
 224            f"({stats['chars_cleaned']} / {stats['chars_in']} chars)")
 225      print(json.dumps(stats, indent=2))
 226      release_single_instance_lock(LOCK_PATH)
 227  
 228  
 229  def _log_dropped(doc_id: str, reason: str) -> None:
 230      """WHY: a durable, per-document record of why each dropped doc was
 231      dropped - both for --resume (load_resume_state in clean_lib.py treats
 232      a doc_id in this log as already-processed, so a resumed run doesn't
 233      re-drop and re-log it) and for auditability (a reader can look up any
 234      specific doc_id and see its fate, not just aggregate counts).
 235      INPUT: doc_id; reason (one of the dropped_language keys, or
 236      "malformed_unicode"/"too_short"/"duplicate").
 237      OUTPUT: none (appends one JSON line to DROPPED_LOG_PATH)."""
 238      with DROPPED_LOG_PATH.open("a", encoding="utf-8") as f:
 239          f.write(json.dumps({"document_id": doc_id, "reason": reason}, ensure_ascii=False) + "\n")
 240  
 241  
 242  if __name__ == "__main__":
 243      main()
```


## hindi/data/pipeline/split_hindi.py

```python
   1  """Split cleaned Hindi documents into train/validation/test, per
   2  report/phase1/phase1_strategy.md section 7.
   3  
   4  - Splits at the document level (never sentence/chunk level) to avoid leakage.
   5  - ~96/2/2 train/val/test by document count, seeded for reproducibility.
   6  - Writes both the actual text files (data/splits/{split}/text/) and explicit
   7    manifests (data/splits/{split}_manifest.json) listing every document's
   8    metadata, as required by the spec.
   9  
  10  Run after clean_hindi.py. Always rebuilds data/splits/ from data/cleaned/ rather
  11  than appending, so it is safe to re-run whenever the cleaned corpus changes
  12  (e.g. once manual data is added).
  13  """
  14  
  15  import argparse
  16  import json
  17  import os
  18  import random
  19  import shutil
  20  import sys
  21  from pathlib import Path
  22  
  23  sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
  24  from clean_lib import acquire_single_instance_lock, release_single_instance_lock  # noqa: E402
  25  
  26  DATA_DIR = Path(__file__).resolve().parents[1]
  27  CLEANED_DIR = DATA_DIR / "cleaned"
  28  SPLITS_DIR = DATA_DIR / "splits"
  29  LOCK_PATH = DATA_DIR / "metadata" / ".split.lock"
  30  LANG = "hindi"
  31  
  32  SPLIT_RATIOS = {"train": 0.96, "validation": 0.02, "test": 0.02}
  33  DEFAULT_SEED = 42
  34  
  35  
  36  def link_or_copy(src: Path, dst: Path) -> None:
  37      """Hardlink src to dst (near-instant, byte-identical, same filesystem
  38      only) instead of shutil.copyfile - copying ~500k+ individual small files
  39      one syscall at a time is the dominant cost of this script on Windows/
  40      NTFS (measured at ~2,500 files/min on the Bhojpuri run, i.e. hours for a
  41      full split; Hindi's cleaned pool is larger). Falls back to a real copy
  42      only if hardlinking fails (e.g. src/dst end up on different volumes)."""
  43      try:
  44          os.link(src, dst)
  45      except OSError:
  46          shutil.copyfile(src, dst)
  47  
  48  
  49  def main():
  50      """WHY: splits at the DOCUMENT level with a fixed seed, never
  51      sentence/chunk level, so no document's text can appear split across
  52      train and validation/test (leakage). Always rebuilds data/splits/ from
  53      data/cleaned/ rather than appending, so it's safe to rerun whenever the
  54      cleaned corpus changes (e.g. after the cross-language-duplicate fix).
  55      WHAT: shuffles cleaned doc metadata with the given seed, assigns each
  56      doc to train/validation/test per SPLIT_RATIOS (~96/2/2), hardlinks
  57      (falling back to copy) each doc's text into the right split dir, and
  58      writes a manifest per split.
  59      INPUT: CLI arg --seed; reads data/cleaned/metadata.jsonl.
  60      OUTPUT: writes data/splits/{train,validation,test}/text/ and
  61      data/splits/{split}_manifest.json; no return value."""
  62      parser = argparse.ArgumentParser(description="Split cleaned Hindi docs into train/val/test")
  63      parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
  64      args = parser.parse_args()
  65  
  66      acquire_single_instance_lock(LOCK_PATH)
  67      try:
  68          meta_path = CLEANED_DIR / "metadata.jsonl"
  69          if not meta_path.exists():
  70              print(f"[{LANG}] No cleaned metadata at {meta_path}; run clean_hindi.py first.")
  71              return
  72          docs = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
  73          if not docs:
  74              print(f"[{LANG}] No cleaned documents found; run clean_hindi.py first.")
  75              return
  76  
  77          rng = random.Random(args.seed)
  78          rng.shuffle(docs)
  79  
  80          n = len(docs)
  81          n_val = max(1, round(n * SPLIT_RATIOS["validation"]))
  82          n_test = max(1, round(n * SPLIT_RATIOS["test"]))
  83          n_train = n - n_val - n_test
  84  
  85          assignment = (
  86              [("train", d) for d in docs[:n_train]]
  87              + [("validation", d) for d in docs[n_train : n_train + n_val]]
  88              + [("test", d) for d in docs[n_train + n_val :]]
  89          )
  90  
  91          for split in SPLIT_RATIOS:
  92              text_dir = SPLITS_DIR / split / "text"
  93              text_dir.mkdir(parents=True, exist_ok=True)
  94              for old in text_dir.glob("*.txt"):
  95                  old.unlink()
  96  
  97          manifests = {split: [] for split in SPLIT_RATIOS}
  98          stats = {split: {"docs": 0, "chars": 0} for split in SPLIT_RATIOS}
  99  
 100          for split, meta in assignment:
 101              doc_id = meta["document_id"]
 102              src = CLEANED_DIR / "text" / f"{doc_id}.txt"
 103              if not src.exists():
 104                  continue
 105              dst = SPLITS_DIR / split / "text" / f"{doc_id}.txt"
 106              link_or_copy(src, dst)
 107              split_meta = dict(meta)
 108              split_meta["split"] = split
 109              manifests[split].append(split_meta)
 110              stats[split]["docs"] += 1
 111              stats[split]["chars"] += split_meta.get("characters", 0)
 112  
 113          for split in SPLIT_RATIOS:
 114              manifest_path = SPLITS_DIR / f"{split}_manifest.json"
 115              manifest_path.write_text(json.dumps(manifests[split], indent=2, ensure_ascii=False), encoding="utf-8")
 116  
 117          print(f"[{LANG}] split {n} documents (seed={args.seed}):")
 118          for split in SPLIT_RATIOS:
 119              print(f"  {split}: {stats[split]['docs']} docs, {stats[split]['chars']} chars")
 120      finally:
 121          release_single_instance_lock(LOCK_PATH)
 122  
 123  
 124  if __name__ == "__main__":
 125      main()
```


## hindi/tokenizer/train_tokenizer.py

```python
   1  """Train a Hindi SentencePiece tokenizer at a given vocab size.
   2  
   3  Mirrors bhojpuri/tokenizer/train_tokenizer.py exactly (same logic, Hindi
   4  paths/model-name prefix). Trains only on data/splits/train/ - never
   5  validation or test, so fertility/coverage/UNK-rate measurements on held-out
   6  text (see analyze_tokenizer.py) are a genuine held-out evaluation, not
   7  train-set leakage.
   8  
   9  Settings are fixed across the vocab-size sweep (5k/6k/8k/10k, per a
  10  course-TA ruling capping vocab size to keep the embedding table from
  11  dominating the ~25M-parameter budget - see
  12  report/phase1/tokenizer_analysis.md) so the sweep only varies vocab_size:
  13    - model_type=unigram
  14    - character_coverage=0.9995
  15    - byte_fallback=True (structurally makes UNK rate 0.0 - see
  16      analyze_tokenizer.py's byte_fallback_rate for the metric that actually
  17      matters instead)
  18    - split_digits=True
  19    - normalization_rule_name=identity (no case-folding/width-normalization
  20      that could alter Devanagari text)
  21    - input_sentence_size=2_000_000, shuffled sampling
  22  """
  23  
  24  import argparse
  25  import sys
  26  from concurrent.futures import ThreadPoolExecutor
  27  from pathlib import Path
  28  
  29  import sentencepiece as spm
  30  
  31  sys.stdout.reconfigure(encoding="utf-8")  # see analyze_tokenizer.py for why
  32  
  33  sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
  34  from clean_lib import acquire_single_instance_lock, release_single_instance_lock  # noqa: E402
  35  
  36  LANG_DIR = Path(__file__).resolve().parents[1]
  37  TRAIN_TEXT_DIR = LANG_DIR / "data" / "splits" / "train" / "text"
  38  TOKENIZER_DIR = Path(__file__).resolve().parent
  39  MERGED_CORPUS = TOKENIZER_DIR / "_train_corpus.txt"
  40  LOCK_PATH = TOKENIZER_DIR / ".train.lock"
  41  READ_WORKERS = 32
  42  
  43  
  44  def _read(path: Path) -> str:
  45      """WHY: factored out to a plain function (not a lambda/inline) so it's
  46      picklable/mappable by ThreadPoolExecutor.map in build_merged_corpus.
  47      WHAT: reads one train-split text file fully into memory.
  48      INPUT: path to a single .txt file.
  49      OUTPUT: its full contents as one string."""
  50      return path.read_text(encoding="utf-8")
  51  
  52  
  53  def build_merged_corpus() -> Path:
  54      """Flatten every train-split document into one newline-per-line corpus
  55      file, which is what SentencePieceTrainer expects as `input`. Cached
  56      across the vocab-size sweep so this (the slow part, given ~500k+ small
  57      files - Hindi's cleaned pool is larger than Bhojpuri's) only runs once.
  58  
  59      Reads are thread-pooled: on Windows/NTFS, per-file open/read of ~500k
  60      small files serially was measured at ~24.5KB/s wall-clock (a ~2.5h ETA
  61      for this step alone, on the Bhojpuri run) - each read is I/O-wait
  62      dominated, not CPU-bound, so it releases the GIL and threading gives a
  63      large real speedup here. Writing stays single-threaded (via
  64      executor.map, which preserves input order) so the merged corpus is
  65      still written in a deterministic order."""
  66      if MERGED_CORPUS.exists():
  67          return MERGED_CORPUS
  68      files = sorted(TRAIN_TEXT_DIR.glob("*.txt"))
  69      if not files:
  70          raise SystemExit(f"no train text files found at {TRAIN_TEXT_DIR} - run split_hindi.py first")
  71      with MERGED_CORPUS.open("w", encoding="utf-8") as out, ThreadPoolExecutor(READ_WORKERS) as pool:
  72          for text in pool.map(_read, files, chunksize=64):
  73              for line in text.split("\n"):
  74                  line = line.strip()
  75                  if line:
  76                      out.write(line + "\n")
  77      print(f"built merged train corpus: {MERGED_CORPUS} ({len(files):,} docs)")
  78      return MERGED_CORPUS
  79  
  80  
  81  def main():
  82      """WHY: single-instance-locked (see acquire_single_instance_lock) since
  83      this environment has been observed to launch two OS processes per
  84      background command - without the lock, two concurrent trainings would
  85      both build/read the merged corpus and both write the same output
  86      .model/.vocab path, corrupting either the corpus file or the model.
  87      WHAT: builds (or reuses the cached) merged train corpus, then trains
  88      one SentencePiece unigram model at the requested vocab size with the
  89      fixed settings documented in the module docstring.
  90      INPUT: CLI args --vocab-size (required), --threads.
  91      OUTPUT: writes hindi_unigram_{vocab_size}.model/.vocab; no return
  92      value."""
  93      parser = argparse.ArgumentParser(description="Train a Hindi SentencePiece tokenizer")
  94      parser.add_argument("--vocab-size", type=int, required=True)
  95      parser.add_argument("--threads", type=int, default=4)
  96      args = parser.parse_args()
  97  
  98      acquire_single_instance_lock(LOCK_PATH)
  99      try:
 100          corpus = build_merged_corpus()
 101          model_prefix = str(TOKENIZER_DIR / f"hindi_unigram_{args.vocab_size}")
 102  
 103          spm.SentencePieceTrainer.Train(
 104              input=str(corpus),
 105              model_prefix=model_prefix,
 106              vocab_size=args.vocab_size,
 107              model_type="unigram",
 108              character_coverage=0.9995,
 109              byte_fallback=True,
 110              split_digits=True,
 111              normalization_rule_name="identity",
 112              input_sentence_size=2_000_000,
 113              shuffle_input_sentence=True,
 114              num_threads=args.threads,
 115          )
 116          print(f"trained {model_prefix}.model / {model_prefix}.vocab")
 117      finally:
 118          release_single_instance_lock(LOCK_PATH)
 119  
 120  
 121  if __name__ == "__main__":
 122      main()
```


## hindi/tokenizer/analyze_tokenizer.py

```python
   1  """Measure a trained Hindi SentencePiece tokenizer on held-out text.
   2  
   3  Mirrors bhojpuri/tokenizer/analyze_tokenizer.py exactly (same logic, Hindi
   4  paths/model-name prefix). Evaluated on data/splits/validation/ (never train,
   5  never test - validation is the correct split for a model-selection decision
   6  like "which vocab size", keeping test genuinely unseen for Phase 2/3
   7  evaluation).
   8  
   9  Reports the numbers needed to justify a vocab-size choice:
  10    - chars/token (the ratio the rest of Phase 1/2 planning depends on)
  11    - fertility (tokens per whitespace-delimited word)
  12    - UNK rate (structurally ~0.0 with byte_fallback on - not a useful signal
  13      on its own, reported only for completeness)
  14    - byte-fallback rate (fraction of emitted tokens that are raw-byte
  15      fallback pieces, i.e. text the vocab genuinely could not represent as a
  16      normal subword - this is the metric that actually stands in for "UNK
  17      rate" here)
  18    - vocab utilization (distinct piece ids used at least once / vocab_size)
  19    - top tokens by frequency
  20    - 3 sample tokenizations with piece strings and ids
  21  """
  22  
  23  import argparse
  24  import collections
  25  import re
  26  import sys
  27  from pathlib import Path
  28  
  29  import sentencepiece as spm
  30  
  31  # stdout can default to a non-UTF-8 codepage on Windows (cp1252) when not
  32  # attached to a real console (e.g. redirected to a log file via `> log.txt`)
  33  # - printing Devanagari token pieces/samples then crashes with
  34  # UnicodeEncodeError partway through the report. Force UTF-8 explicitly so
  35  # this script is safe to run either interactively or redirected.
  36  sys.stdout.reconfigure(encoding="utf-8")
  37  
  38  LANG_DIR = Path(__file__).resolve().parents[1]
  39  VAL_TEXT_DIR = LANG_DIR / "data" / "splits" / "validation" / "text"
  40  TOKENIZER_DIR = Path(__file__).resolve().parent
  41  _BYTE_FALLBACK_PIECE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
  42  
  43  
  44  def load_validation_text(max_docs: int) -> list[str]:
  45      """WHY: analysis must run on text the tokenizer never trained on
  46      (validation, not train) so fertility/utilization/byte-fallback numbers
  47      reflect genuine held-out generalization, not memorized training text.
  48      WHAT: reads up to max_docs .txt files from data/splits/validation/,
  49      skipping any that are empty after stripping.
  50      INPUT: max_docs cap on how many files to read (sampling, not the full
  51      validation split, since 2,000 docs is enough for stable rate estimates
  52      and reading all ~14k would be needlessly slow for a sanity-check pass).
  53      OUTPUT: list of document text strings."""
  54      files = sorted(VAL_TEXT_DIR.glob("*.txt"))
  55      if not files:
  56          raise SystemExit(f"no validation text files at {VAL_TEXT_DIR} - run split_hindi.py first")
  57      texts = []
  58      for fp in files[:max_docs]:
  59          text = fp.read_text(encoding="utf-8").strip()
  60          if text:
  61              texts.append(text)
  62      return texts
  63  
  64  
  65  def analyze(model_path: Path, texts: list[str], n_samples: int = 3) -> dict:
  66      """WHY: bundles every metric this project's vocab-size decision is
  67      based on (see the module docstring's list) into one pass over the
  68      validation sample, so a single function call produces everything
  69      tokenizer_analysis.md quotes for one vocab size.
  70      WHAT: encodes each text with the SentencePiece model (both as ids and
  71      as piece strings, since UNK/byte-fallback detection needs the string
  72      form but chars/token needs only counts), accumulates running totals,
  73      and separately re-encodes the first n_samples texts to keep example
  74      (text, pieces, ids) triples for the report.
  75      INPUT: model_path (a trained .model file); texts (from
  76      load_validation_text); n_samples (how many worked examples to keep).
  77      OUTPUT: dict of every metric plus 'samples' - see the return statement
  78      for the exact keys; this is what print_report formats."""
  79      sp = spm.SentencePieceProcessor(model_file=str(model_path))
  80      vocab_size = sp.vocab_size()
  81      unk_id = sp.unk_id()
  82  
  83      total_chars = 0
  84      total_tokens = 0
  85      total_words = 0
  86      unk_count = 0
  87      byte_fallback_count = 0
  88      piece_freq = collections.Counter()
  89      used_ids = set()
  90  
  91      for text in texts:
  92          ids = sp.encode(text, out_type=int)
  93          pieces = sp.encode(text, out_type=str)
  94          total_chars += len(text)
  95          total_tokens += len(ids)
  96          total_words += len(text.split())
  97          for tid, piece in zip(ids, pieces):
  98              used_ids.add(tid)
  99              piece_freq[piece] += 1
 100              if tid == unk_id:
 101                  unk_count += 1
 102              if _BYTE_FALLBACK_PIECE.match(piece):
 103                  byte_fallback_count += 1
 104  
 105      samples = []
 106      for text in texts[:n_samples]:
 107          ids = sp.encode(text, out_type=int)
 108          pieces = sp.encode(text, out_type=str)
 109          samples.append({
 110              "text": text[:80] + ("..." if len(text) > 80 else ""),
 111              "pieces": pieces[:30],
 112              "ids": ids[:30],
 113          })
 114  
 115      return {
 116          "vocab_size": vocab_size,
 117          "n_docs_evaluated": len(texts),
 118          "chars_per_token": total_chars / total_tokens if total_tokens else float("nan"),
 119          "fertility_tokens_per_word": total_tokens / total_words if total_words else float("nan"),
 120          "unk_rate": unk_count / total_tokens if total_tokens else float("nan"),
 121          "byte_fallback_rate": byte_fallback_count / total_tokens if total_tokens else float("nan"),
 122          "vocab_utilization": len(used_ids) / vocab_size,
 123          "top_tokens": piece_freq.most_common(15),
 124          "samples": samples,
 125      }
 126  
 127  
 128  def print_report(result: dict):
 129      """WHY: keeps formatting separate from computation (analyze() returns
 130      plain data, this only renders it) so the dict can also be consumed
 131      programmatically elsewhere without re-parsing printed text.
 132      WHAT: prints every metric from analyze()'s return dict in the fixed
 133      layout tokenizer_analysis.md's numbers were transcribed from.
 134      INPUT: result dict from analyze().
 135      OUTPUT: none (prints to stdout only)."""
 136      print(f"\n{'=' * 70}")
 137      print(f"vocab_size={result['vocab_size']}  (evaluated on {result['n_docs_evaluated']} validation docs)")
 138      print(f"{'=' * 70}")
 139      print(f"  chars/token:        {result['chars_per_token']:.4f}")
 140      print(f"  fertility (tok/wd): {result['fertility_tokens_per_word']:.4f}")
 141      print(f"  UNK rate:           {result['unk_rate']:.6f}  (structurally ~0 with byte_fallback on)")
 142      print(f"  byte-fallback rate: {result['byte_fallback_rate']:.6f}")
 143      print(f"  vocab utilization:  {result['vocab_utilization']:.4f}")
 144      print(f"  top tokens: {result['top_tokens']}")
 145      print("  sample tokenizations:")
 146      for s in result["samples"]:
 147          print(f"    text:   {s['text']}")
 148          print(f"    pieces: {s['pieces']}")
 149          print(f"    ids:    {s['ids']}")
 150  
 151  
 152  def main():
 153      """WHY: CLI entry point so the sweep driver (run_hindi.bat /
 154      run_hindi_tokenizer_sweep.sh) can analyze one vocab size at a time
 155      right after training it.
 156      WHAT: loads the requested vocab size's model, loads validation text,
 157      runs analyze(), prints the report.
 158      INPUT: CLI args --vocab-size (required), --max-docs (default 2000).
 159      OUTPUT: none (prints only); exits with an error if that vocab size
 160      hasn't been trained yet."""
 161      parser = argparse.ArgumentParser(description="Analyze a trained Hindi tokenizer on held-out text")
 162      parser.add_argument("--vocab-size", type=int, required=True)
 163      parser.add_argument("--max-docs", type=int, default=2000)
 164      args = parser.parse_args()
 165  
 166      model_path = TOKENIZER_DIR / f"hindi_unigram_{args.vocab_size}.model"
 167      if not model_path.exists():
 168          raise SystemExit(f"{model_path} not found - run train_tokenizer.py --vocab-size {args.vocab_size} first")
 169  
 170      texts = load_validation_text(args.max_docs)
 171      result = analyze(model_path, texts)
 172      print_report(result)
 173  
 174  
 175  if __name__ == "__main__":
 176      main()
```

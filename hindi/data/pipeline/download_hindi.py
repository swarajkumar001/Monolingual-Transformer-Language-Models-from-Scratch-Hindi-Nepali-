"""Download Hindi (Model H) downloaded-source text into data/raw/downloaded/.

Verified real sources (checked against the HF Hub / datasets-server APIs before
writing this script, not assumed):
  - wikimedia/wikipedia, config 20231101.hi : Hindi Wikipedia dump via HF's parquet
    mirror (raw XML dump is ~239MB compressed on dumps.wikimedia.org).
  - ai4bharat/sangraha, config "verified", split "hin" : the human-verified Hindi
    partition of Sangraha (a 251B-token, 22-language curated Indic corpus). Schema
    confirmed as {doc_id, type, text}. This config/split combination is large, so
    this script streams it and stops at --max-chars-per-source rather than pulling
    it in full - the goal is to reach the ~500M *token* target for Hindi overall,
    not to ingest the entire corpus.
  - ai4bharat/sangraha, config "unverified/hin" : the larger, perplexity-filtered
    (not human-verified) Hindi partition of Sangraha. Schema confirmed as
    {doc_id, text}. Added when 300M chars from the verified partition alone
    wasn't enough to approach the ~500M-token target.
  - HuggingFaceFW/fineweb-2, config "hin_Deva" : CommonCrawl-derived Hindi web
    text. Schema confirmed as {text, id, dump, url, date, file_path, language,
    language_score, language_script, minhash_cluster_size, top_langs}. Being
    CommonCrawl-derived, it can contain closely-related-language contamination
    (e.g. Bhojpuri pages tagged Hindi) - the language_score field is used as a
    coarse pre-filter, and the real Hindi/Bhojpuri distinction still happens in
    the language-ID cleaning stage, not here.

This script only downloads and records metadata. It does not clean, normalize, or
deduplicate - that happens in the later pipeline stages so each step stays testable
in isolation.
"""

import argparse
import json
import hashlib
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

LANG = "hindi"
LANG_ID = "hi"
OUT_DIR = Path(__file__).resolve().parents[1] / "raw" / "downloaded"


def document_id(source_name: str, raw_id: str) -> str:
    """WHY: doc_ids must be stable across reruns (so --append doesn't create
    duplicates) and collision-safe across sources, without depending on the
    upstream dataset's own id scheme being globally unique.
    WHAT: hashes "{source_name}:{raw_id}" and prefixes with the language +
    source_type, matching the id convention every other language/source in
    this project uses (see clean_lib.py's dedup logic, which trusts this
    prefix to route Bhojpuri-vs-Hindi checks correctly).
    INPUT: source_name (e.g. "wikipedia"), raw_id (the upstream dataset's own
    row/doc id, any string).
    OUTPUT: "hi_downloaded_{source_name}_{10-char hex digest}"."""
    digest = hashlib.sha1(f"{source_name}:{raw_id}".encode("utf-8")).hexdigest()[:10]
    return f"hi_downloaded_{source_name}_{digest}"


def write_document(out_dir: Path, doc_id: str, text: str, meta: dict) -> None:
    """WHY: every raw document needs both its text (for later pipeline
    stages to read) and its provenance metadata (source_type, license, etc.
    - what validate_phase1.py and the cleaning stage both rely on) persisted
    together, atomically enough that a later stage never sees one without
    the other for the same doc_id.
    WHAT: writes {out_dir}/text/{doc_id}.txt, then appends one JSON line to
    {out_dir}/metadata.jsonl.
    INPUT: out_dir (raw/downloaded), doc_id, text (str), meta (dict matching
    the schema documented inline in each download_* function below).
    OUTPUT: none (disk side effects only)."""
    (out_dir / "text").mkdir(parents=True, exist_ok=True)
    (out_dir / "text" / f"{doc_id}.txt").write_text(text, encoding="utf-8")
    meta_path = out_dir / "metadata.jsonl"
    with meta_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")


def download_wikipedia(out_dir: Path, max_chars: int | None) -> tuple[int, int]:
    """WHY/WHAT: see module docstring for why this source (HF parquet mirror
    of the Hindi Wikipedia dump) is used and its schema. Streams rather than
    downloading in full so a --max-chars cap can stop early without paying
    for the whole dataset.
    INPUT: out_dir to write into; max_chars cap in characters, or None for
    the full dump (~239MB compressed upstream, so finite either way).
    OUTPUT: (doc_count, char_count) written this call."""
    source_name = "wikipedia"
    print(f"[{LANG}] streaming wikimedia/wikipedia 20231101.hi ...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train", streaming=True)
    count, chars = 0, 0
    for row in tqdm(ds, desc="hi-wikipedia"):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        raw_id = row.get("id", str(count))
        doc_id = document_id(source_name, raw_id)
        meta = {
            "document_id": doc_id,
            "language": LANG,
            "source_type": "downloaded",
            "source_name": "wikimedia/wikipedia:20231101.hi",
            "source_description": "Hindi Wikipedia dump (HF parquet mirror, 2023-11-01 snapshot)",
            "collection_method": "download",
            "license_or_usage_note": "CC-BY-SA-4.0",
            "url": row.get("url") or "",
            "characters": len(text),
            "split": None,
            "cleaning_status": "raw",
        }
        write_document(out_dir, doc_id, text, meta)
        count += 1
        chars += len(text)
        if max_chars is not None and chars >= max_chars:
            break
    print(f"[{LANG}] Wikipedia: wrote {count} documents, {chars} characters")
    return count, chars


def download_sangraha(out_dir: Path, max_chars: int | None) -> tuple[int, int]:
    """WHY/WHAT: see module docstring - Sangraha's human-verified Hindi
    partition (config "verified", split "hin"). Capped by default since the
    full partition is far more than needed for the ~500M-token target.
    INPUT: out_dir to write into; max_chars cap in characters (default
    300M, see main()'s argparse default).
    OUTPUT: (doc_count, char_count) written this call."""
    source_name = "sangraha"
    print(f"[{LANG}] streaming ai4bharat/sangraha (verified/hin) ...")
    ds = load_dataset("ai4bharat/sangraha", data_dir="verified/hin", split="train", streaming=True)
    count, chars = 0, 0
    for row in tqdm(ds, desc="sangraha-hin"):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        raw_id = row.get("doc_id", str(count))
        doc_id = document_id(source_name, raw_id)
        meta = {
            "document_id": doc_id,
            "language": LANG,
            "source_type": "downloaded",
            "source_name": f"ai4bharat/sangraha:verified/hin:{row.get('type', 'unknown')}",
            "source_description": "Sangraha verified Hindi partition (curated Indic pretraining corpus)",
            "collection_method": "download",
            "license_or_usage_note": "CC-BY-4.0",
            "url": "",
            "characters": len(text),
            "split": None,
            "cleaning_status": "raw",
        }
        write_document(out_dir, doc_id, text, meta)
        count += 1
        chars += len(text)
        if max_chars is not None and chars >= max_chars:
            break
    print(f"[{LANG}] Sangraha: wrote {count} documents, {chars} characters")
    return count, chars


def download_sangraha_unverified(out_dir: Path, max_chars: int | None) -> tuple[int, int]:
    """WHY/WHAT: see module docstring - Sangraha's larger, perplexity-filtered
    (not human-verified) Hindi partition, added because the verified
    partition alone wasn't enough volume toward the ~500M-token target.
    INPUT: out_dir to write into; max_chars cap in characters (default 500M).
    OUTPUT: (doc_count, char_count) written this call."""
    source_name = "sangraha_unverified"
    print(f"[{LANG}] streaming ai4bharat/sangraha (unverified/hin) ...")
    ds = load_dataset("ai4bharat/sangraha", data_dir="unverified/hin", split="train", streaming=True)
    count, chars = 0, 0
    for row in tqdm(ds, desc="sangraha-unverified-hin"):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        raw_id = row.get("doc_id", str(count))
        doc_id = document_id(source_name, raw_id)
        meta = {
            "document_id": doc_id,
            "language": LANG,
            "source_type": "downloaded",
            "source_name": "ai4bharat/sangraha:unverified/hin",
            "source_description": "Sangraha unverified Hindi partition (perplexity-filtered, not human-verified)",
            "collection_method": "download",
            "license_or_usage_note": "CC-BY-4.0",
            "url": "",
            "characters": len(text),
            "split": None,
            "cleaning_status": "raw",
        }
        write_document(out_dir, doc_id, text, meta)
        count += 1
        chars += len(text)
        if max_chars is not None and chars >= max_chars:
            break
    print(f"[{LANG}] Sangraha (unverified): wrote {count} documents, {chars} characters")
    return count, chars


def download_fineweb2(out_dir: Path, max_chars: int | None, min_language_score: float = 0.9) -> tuple[int, int]:
    """WHY/WHAT: see module docstring - CommonCrawl-derived Hindi web text.
    Filters on the dataset's own language_score as a cheap pre-filter
    against closely-related-language contamination (e.g. Bhojpuri pages
    tagged Hindi); this is not a substitute for clean_hindi.py's real
    lexical-marker contamination filter, just a first pass that discards
    the most confidently-mislabeled rows before they're even written.
    INPUT: out_dir to write into; max_chars cap; min_language_score
    threshold (rows below this are skipped, not written).
    OUTPUT: (doc_count, char_count) written this call."""
    source_name = "fineweb2"
    print(f"[{LANG}] streaming HuggingFaceFW/fineweb-2 (hin_Deva) ...")
    ds = load_dataset("HuggingFaceFW/fineweb-2", name="hin_Deva", split="train", streaming=True)
    count, chars, skipped_low_score = 0, 0, 0
    for row in tqdm(ds, desc="fineweb2-hin_Deva"):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        if (row.get("language_score") or 0) < min_language_score:
            skipped_low_score += 1
            continue
        raw_id = row.get("id", str(count))
        doc_id = document_id(source_name, raw_id)
        meta = {
            "document_id": doc_id,
            "language": LANG,
            "source_type": "downloaded",
            "source_name": "HuggingFaceFW/fineweb-2:hin_Deva",
            "source_description": "FineWeb-2 Hindi partition (CommonCrawl-derived web text)",
            "collection_method": "download",
            "license_or_usage_note": "ODC-By (dataset); underlying page licenses vary",
            "url": row.get("url") or "",
            "characters": len(text),
            "split": None,
            "cleaning_status": "raw",
        }
        write_document(out_dir, doc_id, text, meta)
        count += 1
        chars += len(text)
        if max_chars is not None and chars >= max_chars:
            break
    print(f"[{LANG}] FineWeb-2: wrote {count} documents, {chars} characters "
          f"(skipped {skipped_low_score} below language_score {min_language_score})")
    return count, chars


def main():
    """WHY: a single CLI entry point so re-running with --sources/--append
    lets one add a new downloaded source later without re-pulling ones
    already on disk (each source is independently resumable by simply not
    re-listing it).
    WHAT: parses args, optionally wipes raw/downloaded/ (unless --append),
    then calls whichever download_* functions were requested in --sources
    and prints running totals.
    INPUT/OUTPUT: CLI args in, disk writes out (see write_document); no
    return value."""
    parser = argparse.ArgumentParser(description="Download Hindi source text (Model H)")
    parser.add_argument("--max-chars-wikipedia", type=int, default=None,
                         help="Char budget for Wikipedia (omit for the full dump)")
    parser.add_argument("--max-chars-sangraha", type=int, default=300_000_000,
                         help="Char budget for Sangraha verified (default caps it; Sangraha is huge)")
    parser.add_argument("--max-chars-sangraha-unverified", type=int, default=500_000_000)
    parser.add_argument("--max-chars-fineweb2", type=int, default=500_000_000)
    parser.add_argument("--sources", nargs="+", default=["wikipedia", "sangraha"],
                         choices=["wikipedia", "sangraha", "sangraha_unverified", "fineweb2"])
    parser.add_argument("--append", action="store_true",
                         help="Append to existing raw/downloaded/ instead of wiping it first "
                              "(use when adding new sources without re-pulling ones already done)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = OUT_DIR / "metadata.jsonl"
    if meta_path.exists() and not args.append:
        meta_path.unlink()

    total_docs, total_chars = 0, 0
    if "wikipedia" in args.sources:
        d, c = download_wikipedia(OUT_DIR, args.max_chars_wikipedia)
        total_docs += d
        total_chars += c
    if "sangraha" in args.sources:
        d, c = download_sangraha(OUT_DIR, args.max_chars_sangraha)
        total_docs += d
        total_chars += c
    if "sangraha_unverified" in args.sources:
        d, c = download_sangraha_unverified(OUT_DIR, args.max_chars_sangraha_unverified)
        total_docs += d
        total_chars += c
    if "fineweb2" in args.sources:
        d, c = download_fineweb2(OUT_DIR, args.max_chars_fineweb2)
        total_docs += d
        total_chars += c

    print(f"[{LANG}] TOTAL documents downloaded this run: {total_docs}, characters: {total_chars}")
    print(f"[{LANG}] Output: {OUT_DIR}")


if __name__ == "__main__":
    main()

"""Ingest the self-extracted Hindi Wikipedia dump into data/raw/manual/.

Per the course ruling this project follows: an HF-mirrored/parsed dataset
(e.g. wikimedia/wikipedia:20231101.hi, already in data/raw/downloaded/) does
NOT count as manual collection, but a dump downloaded and extracted by the
student themselves does. This script ingests the output of:

  curl -L -C - -o hiwiki-latest-pages-articles.xml.bz2 \
      https://dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2
  python -m wikiextractor.WikiExtractor <dump>.bz2 --json -o data/external/wikidump/extracted

as source_type="manual", collection_method="wikipedia_dump", license CC-BY-SA-4.0.

Mirrors bhojpuri/data/pipeline/ingest_wikipedia_dump.py exactly (same logic,
Hindi paths/ids). The existing wikimedia/wikipedia:20231101.hi HF-downloaded
records in data/raw/downloaded/ must be dropped separately, after this script
has run successfully and been verified - not before, and not by this script
(same reasoning as the Bhojpuri version: don't rewrite a raw/downloaded/
metadata.jsonl that another process may be reading concurrently).
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

EXTRACTED_DIR = Path(__file__).resolve().parents[1] / "external" / "wikidump" / "extracted"
OUT_DIR = Path(__file__).resolve().parents[1] / "raw" / "manual"
SOURCE_NAME = "hi_wikipedia_dump"
LICENSE_NOTE = (
    "Self-extracted from the official Hindi Wikipedia XML dump "
    "(dumps.wikimedia.org/hiwiki/latest), via WikiExtractor - not the HF "
    "wikimedia/wikipedia mirror. CC-BY-SA-4.0. Counts as manual collection per "
    "the course ruling that a self-extracted dump (unlike an HF-parsed dataset) "
    "qualifies. Overlaps with wikimedia/wikipedia:20231101.hi in "
    "data/raw/downloaded/ - that HF mirror is dropped once this script has run "
    "and been verified, see ingest_wikipedia_dump.py module docstring."
)


def append_metadata_locked(meta_path: Path, record: dict, retries: int = 200, wait: float = 0.05) -> None:
    """WHY: this script can run concurrently with other writers to the same
    raw/manual/metadata.jsonl (crawlers, other ingest scripts); a plain
    unlocked append can interleave two processes' writes mid-line and
    corrupt the file (this happened earlier this project with crawler
    metadata - see scripts/crawl_sites.py). An os.O_CREAT|os.O_EXCL lock
    file is atomic at the OS level, unlike a check-then-write pattern.
    WHAT: acquires an exclusive lock file next to meta_path (retrying up to
    `retries` times), appends one JSON line, then releases the lock.
    INPUT: meta_path (the metadata.jsonl to append to); record (dict to
    serialize as one JSON line); retries/wait control the acquire loop.
    OUTPUT: none (disk side effect only); raises TimeoutError if the lock
    can't be acquired within retries * wait seconds."""
    lock_path = meta_path.with_suffix(meta_path.suffix + ".writelock")
    line = json.dumps(record, ensure_ascii=False) + "\n"
    for _ in range(retries):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except (FileExistsError, PermissionError):
            # Windows can raise PermissionError instead of FileExistsError
            # under lock-file create/delete contention - both mean "retry".
            time.sleep(wait)
    else:
        raise TimeoutError(f"could not acquire write lock for {meta_path} after {retries} attempts")
    try:
        with meta_path.open("a", encoding="utf-8") as f:
            f.write(line)
    finally:
        lock_path.unlink(missing_ok=True)


def write_document(doc_id: str, text: str, meta: dict) -> None:
    """WHY/WHAT: same role as the other ingest scripts' write_document, but
    uses append_metadata_locked instead of a plain append since this
    script's output directory (raw/manual/) is written concurrently by
    other sources (crawlers, ingest_manual.py).
    INPUT: doc_id, text (str), meta (dict; source_type="manual",
    collection_method="wikipedia_dump" per the module docstring above).
    OUTPUT: none (disk side effects only)."""
    (OUT_DIR / "text").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "text" / f"{doc_id}.txt").write_text(text, encoding="utf-8")
    append_metadata_locked(OUT_DIR / "metadata.jsonl", meta)


def main():
    """WHY: separates "download the XML dump + run WikiExtractor" (a manual,
    one-time shell step documented in the module docstring) from "ingest
    the extracted JSON into this project's raw/manual/ schema" (this
    script) - keeps the extraction tool swappable without touching the
    schema-writing logic.
    WHAT: walks every WikiExtractor output shard under EXTRACTED_DIR,
    skips too-short stubs, and writes each remaining article via
    write_document.
    INPUT/OUTPUT: reads EXTRACTED_DIR, writes to OUT_DIR; no return value;
    exits early with a message if EXTRACTED_DIR doesn't exist yet."""
    parser = argparse.ArgumentParser(description="Ingest self-extracted Hindi Wikipedia dump")
    args = parser.parse_args()

    if not EXTRACTED_DIR.exists():
        print(f"[hi_wiki_dump] {EXTRACTED_DIR} not found - run WikiExtractor first")
        return

    count = chars = skipped_short = 0
    for shard_path in sorted(EXTRACTED_DIR.rglob("wiki_*")):
        with shard_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = (row.get("text") or "").strip()
                if len(text) < 50:
                    skipped_short += 1
                    continue
                raw_id = row.get("id", "")
                digest = hashlib.sha1(f"{SOURCE_NAME}:{raw_id}".encode("utf-8")).hexdigest()[:10]
                doc_id = f"hi_manual_{SOURCE_NAME}_{digest}"
                meta = {
                    "document_id": doc_id,
                    "language": "hindi",
                    "source_type": "manual",
                    "source_name": SOURCE_NAME,
                    "source_description": row.get("title", ""),
                    "collection_method": "wikipedia_dump",
                    "license_or_usage_note": LICENSE_NOTE,
                    "url": row.get("url") or "",
                    "characters": len(text),
                    "split": None,
                    "cleaning_status": "raw",
                }
                write_document(doc_id, text, meta)
                count += 1
                chars += len(text)

    print(f"[hi_wiki_dump] wrote {count:,} docs, {chars:,} chars (skipped {skipped_short} too-short stubs)")


if __name__ == "__main__":
    main()

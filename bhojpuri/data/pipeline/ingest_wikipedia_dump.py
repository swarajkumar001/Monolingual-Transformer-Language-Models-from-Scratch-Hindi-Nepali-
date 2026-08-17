"""Ingest the self-extracted Bhojpuri Wikipedia dump into data/raw/manual/.

Per the course ruling this project follows: an HF-mirrored/parsed dataset
(e.g. wikimedia/wikipedia:20231101.bh, already in data/raw/downloaded/) does
NOT count as manual collection, but a dump downloaded and extracted by the
student themselves does. This script ingests the output of:

  wget https://dumps.wikimedia.org/bhwiki/latest/bhwiki-latest-pages-articles.xml.bz2
  python -m wikiextractor.WikiExtractor <dump>.bz2 --json -o data/external/wikidump/extracted

as source_type="manual", collection_method="wikipedia_dump", license CC-BY-SA-4.0.

NOTE - not yet done, tracked as a follow-up: the existing
wikimedia/wikipedia:20231101.bh HF-downloaded records in data/raw/downloaded/
should be dropped once it's safe to rewrite that file (a clean_bhojpuri.py
--resume run is reading from it as this script runs; removing lines from a
file mid-read by another process is the same corruption risk this project
already hit once with concurrent writes, so this is deferred to the next
full non-resume clean rerun rather than done now against a live reader).
Until that drop happens, both copies exist and will contribute a large
duplicate mass at the dedup stage (manual copy wins the tie-break, so the
downloaded copy is what gets dropped there) - not double-counted in the
FINAL cleaned corpus, just redundant raw storage in the meantime.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

EXTRACTED_DIR = Path(__file__).resolve().parents[1] / "external" / "wikidump" / "extracted"
OUT_DIR = Path(__file__).resolve().parents[1] / "raw" / "manual"
SOURCE_NAME = "bho_wikipedia_dump"
LICENSE_NOTE = (
    "Self-extracted from the official Bhojpuri Wikipedia XML dump "
    "(dumps.wikimedia.org/bhwiki/latest), via WikiExtractor - not the HF "
    "wikimedia/wikipedia mirror. CC-BY-SA-4.0. Counts as manual collection per "
    "the course ruling that a self-extracted dump (unlike an HF-parsed dataset) "
    "qualifies. Overlaps with wikimedia/wikipedia:20231101.bh in "
    "data/raw/downloaded/ - see ingest_wikipedia_dump.py module docstring."
)


def append_metadata_locked(meta_path: Path, record: dict, retries: int = 200, wait: float = 0.05) -> None:
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
    (OUT_DIR / "text").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "text" / f"{doc_id}.txt").write_text(text, encoding="utf-8")
    append_metadata_locked(OUT_DIR / "metadata.jsonl", meta)


def main():
    parser = argparse.ArgumentParser(description="Ingest self-extracted Bhojpuri Wikipedia dump")
    args = parser.parse_args()

    if not EXTRACTED_DIR.exists():
        print(f"[bho_wiki_dump] {EXTRACTED_DIR} not found - run WikiExtractor first")
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
                doc_id = f"bho_manual_{SOURCE_NAME}_{digest}"
                meta = {
                    "document_id": doc_id,
                    "language": "bhojpuri",
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

    print(f"[bho_wiki_dump] wrote {count:,} docs, {chars:,} chars (skipped {skipped_short} too-short stubs)")


if __name__ == "__main__":
    main()

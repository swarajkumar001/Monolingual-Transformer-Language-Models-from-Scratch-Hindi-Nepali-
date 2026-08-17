"""Ingest the BHLTR (JNU Bhojpuri Language Technological Resources) monolingual
corpus, cloned locally at data/external/BHLTR (github.com/shashwatup9k/BHLTR),
into data/raw/downloaded/.

Treated as source_type "downloaded" (not "manual"): this is a pre-aggregated
external corpus pulled in bulk via git clone, not text the user personally
scraped/OCR'd/typed themselves - same reasoning as why BhojpuriCorpus and the
HF-mirrored Wikipedia dump are "downloaded" rather than "manual".

KNOWN OVERLAP CAVEAT: Satyam810/BhojpuriCorpus (already ingested by
download_bhojpuri.py) has a "jnu_bhltr" partition (79,884 docs) that is very
likely this same underlying corpus, re-packaged/re-segmented. Ingesting both
risks a meaningful chunk of near-duplicate content; document-level dedup in
clean_bhojpuri.py catches EXACT normalized-text matches but may not catch
differently-segmented duplicates (this file is one-sentence-per-line, the
BhojpuriCorpus partition may be chunked differently). Not resolved here -
flagged in metadata (license_or_usage_note) and left for the Phase 1 report
to discuss, not silently hidden.

One line -> one document (source file is already one Bhojpuri sentence per
line, matching the sentence-level granularity the rest of the Bhojpuri
corpus already uses per report/phase1/phase1_strategy.md measurements).
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

BHLTR_DIR = Path(__file__).resolve().parents[1] / "external" / "BHLTR"
OUT_DIR = Path(__file__).resolve().parents[1] / "raw" / "downloaded"
LICENSE_NOTE = (
    "BHLTR (github.com/shashwatup9k/BHLTR), CC-BY-NC-SA-4.0 (see repo license.md) - "
    "non-commercial, acceptable for coursework use, same license class as the "
    "jnu_bhltr partition of Satyam810/BhojpuriCorpus already in this corpus. "
    "CAVEAT: likely substantial content overlap with that same jnu_bhltr partition "
    "(same underlying JNU corpus, different packaging/segmentation) - see "
    "ingest_bhltr.py module docstring. Not fully deduplicated against it beyond "
    "the standard exact-normalized-text dedup pass."
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


def ingest_file(path: Path, source_name: str) -> tuple[int, int]:
    if not path.exists():
        print(f"[bhltr] {path} not found, skipping")
        return 0, 0
    count = chars = 0
    with path.open(encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            text = line.strip()
            if len(text) < 5:
                continue
            digest = hashlib.sha1(f"{source_name}:{i}".encode("utf-8")).hexdigest()[:10]
            doc_id = f"bho_downloaded_{source_name}_{digest}"
            meta = {
                "document_id": doc_id,
                "language": "bhojpuri",
                "source_type": "downloaded",
                "source_name": source_name,
                "source_description": "BHLTR JNU Bhojpuri monolingual corpus (git-cloned)",
                "collection_method": "download",
                "license_or_usage_note": LICENSE_NOTE,
                "url": "https://github.com/shashwatup9k/BHLTR",
                "characters": len(text),
                "split": None,
                "cleaning_status": "raw",
            }
            write_document(doc_id, text, meta)
            count += 1
            chars += len(text)
    print(f"[bhltr] {source_name}: {count:,} docs, {chars:,} chars")
    return count, chars


def main():
    parser = argparse.ArgumentParser(description="Ingest BHLTR JNU Bhojpuri corpus")
    args = parser.parse_args()
    n1, c1 = ingest_file(BHLTR_DIR / "mono-bho-corpus" / "monolingual.bho", "bhltr_monolingual")
    n2, c2 = ingest_file(BHLTR_DIR / "mono-bho-corpus" / "monolingual-v0.2.bho", "bhltr_monolingual_v02")
    print(f"[bhltr] TOTAL: {n1 + n2:,} docs, {c1 + c2:,} chars")


if __name__ == "__main__":
    main()

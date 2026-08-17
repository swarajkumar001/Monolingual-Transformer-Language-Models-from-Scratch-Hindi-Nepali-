"""Ingest select files from a prior personal data-collection effort
(C:\\Users\\swara\\Downloads\\Harsh-...\\Harsh\\bhojpuri data_\\raw) into this
project's data/raw/ tree. Confirmed by the project owner to be their own
work under a different folder name, not a third party's.

Only two of the five source files are ingested here, after inspection:

  indiccorp.txt     -> downloaded. Legitimate public AI4Bharat IndicCorp
                       text, distinct from Sangraha (already in this
                       project) and FineWeb-2. Has ZERO document boundaries
                       (a single unbroken block of text) so this script
                       chunks it on Devanagari sentence-terminator
                       punctuation (danda U+0964, double-danda U+0965, ? !)
                       into paragraph-sized documents.
  scraped_manual.txt -> manual. Small (~725k chars) genuine scrape,
                       mostly of anjoria.com per scrape_metadata.jsonl (a
                       site this project already crawls properly via
                       crawl_sites.py) - kept anyway since dedup will
                       naturally drop the overlap and keep only what's
                       actually new; the per-URL metadata doesn't line up
                       1:1 with the concatenated text (191 URLs sum to
                       548,921 chars vs 725,015 in the file), so this is
                       also sentence-chunked rather than split per-URL.

Deliberately NOT ingested, for reasons recorded in the Phase 1 report:
  hf_aggregated.txt   - confirmed synthetic/LLM-generated content on
                         inspection (instruction-tuning-style templates,
                         generic AI-generated web-design advice, and a
                         literal "nan" pandas/numpy null-artifact embedded
                         mid-sentence - a real data-quality defect, not
                         authentic Bhojpuri).
  wikipedia.txt       - redundant with this project's own Wikipedia dump
                         extraction (bho_wikipedia_dump, source_type=manual,
                         better-provenance than a bare unattributed .txt).
  scraped_extended.txt - state file claims bhojpuriplanet.com/bhojpuri.blog,
                         but actual content is Bhojpuri Wikipedia articles
                         with leaked MediaWiki UI text ("[edit | edit
                         source]") - the crawler apparently wandered onto
                         Wikipedia rather than the intended sites. Redundant
                         with the Wikipedia dump for the same reason as above.
"""

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

HARSH_BASE = Path(r"C:\Users\swara\Downloads\Harsh-20260815T052922Z-1-001\Harsh\bhojpuri data_\raw")
OUT_ROOT = Path(__file__).resolve().parents[1] / "raw"

# Devanagari danda (।), double danda (॥), plus ASCII sentence enders as a
# fallback for any Latin-script fragments mixed into the source text.
_SENTENCE_END = re.compile(r"[।॥!?]")
TARGET_CHUNK_CHARS = 800  # roughly matches the document size of other sources in this corpus


def chunk_by_sentence(text: str, target_chars: int = TARGET_CHUNK_CHARS):
    """Split a boundary-free text blob into paragraph-sized chunks on
    sentence-ending punctuation, accumulating sentences until each chunk is
    roughly target_chars long. Never splits mid-sentence."""
    sentences = []
    start = 0
    for m in _SENTENCE_END.finditer(text):
        end = m.end()
        sentences.append(text[start:end].strip())
        start = end
    if start < len(text):
        tail = text[start:].strip()
        if tail:
            sentences.append(tail)

    chunks = []
    buf = []
    buf_len = 0
    for s in sentences:
        if not s:
            continue
        buf.append(s)
        buf_len += len(s)
        if buf_len >= target_chars:
            chunks.append(" ".join(buf))
            buf, buf_len = [], 0
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def append_metadata_locked(meta_path: Path, record: dict, retries: int = 200, wait: float = 0.05) -> None:
    lock_path = meta_path.with_suffix(meta_path.suffix + ".writelock")
    line = json.dumps(record, ensure_ascii=False) + "\n"
    for _ in range(retries):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except (FileExistsError, PermissionError):
            time.sleep(wait)
    else:
        raise TimeoutError(f"could not acquire write lock for {meta_path} after {retries} attempts")
    try:
        with meta_path.open("a", encoding="utf-8") as f:
            f.write(line)
    finally:
        lock_path.unlink(missing_ok=True)


def write_document(out_dir: Path, doc_id: str, text: str, meta: dict) -> None:
    (out_dir / "text").mkdir(parents=True, exist_ok=True)
    (out_dir / "text" / f"{doc_id}.txt").write_text(text, encoding="utf-8")
    append_metadata_locked(out_dir / "metadata.jsonl", meta)


def ingest_file(filename: str, source_name: str, source_type: str, license_note: str,
                 collection_method: str, min_chars: int = 40) -> tuple[int, int]:
    path = HARSH_BASE / filename
    if not path.exists():
        print(f"[harsh:{source_name}] {path} not found, skipping")
        return 0, 0
    text = path.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_by_sentence(text)

    out_dir = OUT_ROOT / source_type
    count = chars = 0
    for i, chunk in enumerate(chunks):
        if len(chunk) < min_chars:
            continue
        digest = hashlib.sha1(f"{source_name}:{i}".encode("utf-8")).hexdigest()[:10]
        doc_id = f"bho_{source_type}_{source_name}_{digest}"
        meta = {
            "document_id": doc_id,
            "language": "bhojpuri",
            "source_type": source_type,
            "source_name": source_name,
            "source_description": f"Sentence-chunked from {filename} (personal prior collection)",
            "collection_method": collection_method,
            "license_or_usage_note": license_note,
            "url": "",
            "characters": len(chunk),
            "split": None,
            "cleaning_status": "raw",
        }
        write_document(out_dir, doc_id, chunk, meta)
        count += 1
        chars += len(chunk)
    print(f"[harsh:{source_name}] {filename}: {count:,} chunks, {chars:,} chars "
          f"(from {len(text):,} raw chars, {len(chunks):,} total chunks before min-chars filter)")
    return count, chars


def main():
    parser = argparse.ArgumentParser(description="Ingest select files from the Harsh personal-collection folder")
    args = parser.parse_args()

    n1, c1 = ingest_file(
        "indiccorp.txt", "harsh_indiccorp", "downloaded",
        "AI4Bharat IndicCorp (public corpus), personally downloaded prior to this session. "
        "Distinct resource from Sangraha and FineWeb-2 already in this project, though likely "
        "includes some Wikipedia-derived text (overlap handled by standard dedup). "
        "No per-document boundaries in the source file - sentence-chunked here.",
        "download",
    )
    n2, c2 = ingest_file(
        "scraped_manual.txt", "harsh_scraped_manual", "manual",
        "Personal scrape, predominantly anjoria.com per the original scrape_metadata.jsonl "
        "(a site already crawled properly via crawl_sites.py in this project - expect heavy "
        "overlap, handled by standard dedup). Per-URL boundaries could not be reconstructed "
        "(metadata char-length sum does not match the concatenated file), so sentence-chunked "
        "rather than split per source URL. Collected for coursework research use, not for redistribution.",
        "web_scrape",
        min_chars=20,
    )
    print(f"\nTOTAL: {n1+n2:,} docs, {c1+c2:,} chars")


if __name__ == "__main__":
    main()

"""Fetch OCR'd text from archive.org items into {lang}/data/raw/manual/.

Uses the archive.org advancedsearch API to find items matching a query, then
pulls each item's existing OCR text (the "*_djvu.txt" file archive.org
generates for scanned items) via the metadata API. Records whatever
rights/licenseurl fields archive.org actually reports for that item - not an
assumed license; most historical scans have neither field set, which is
recorded honestly as "no explicit rights statement on item - unverified"
rather than guessed at. Items with no OCR text file are skipped, not padded.

Shared across languages - only --lang/--query differ, no data is shared.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANG_ISO = {"hindi": "hi", "bhojpuri": "bho"}

SEARCH_URL = "https://archive.org/advancedsearch.php"
METADATA_URL = "https://archive.org/metadata/{identifier}"
DOWNLOAD_URL = "https://archive.org/download/{identifier}/{filename}"
HEADERS = {"User-Agent": "Mozilla/5.0 (research collection; coursework LM training corpus)"}


def search_identifiers(query: str, max_items: int, retries: int = 5) -> list[str]:
    identifiers = []
    start = 0
    while len(identifiers) < max_items:
        rows = min(50, max_items - len(identifiers))
        params = {"q": query, "fl[]": "identifier", "rows": rows, "start": start, "output": "json"}
        docs = None
        for attempt in range(retries):
            try:
                r = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=90)
                r.raise_for_status()
                docs = r.json()["response"]["docs"]
                break
            except Exception as e:
                wait = min(60, 5 * (attempt + 1))
                print(f"    search request failed ({e}); retrying in {wait}s "
                      f"(attempt {attempt + 1}/{retries})")
                time.sleep(wait)
        if docs is None:
            print(f"    search giving up after {retries} attempts at start={start}; "
                  f"keeping {len(identifiers)} identifiers found so far")
            break
        if not docs:
            break
        identifiers.extend(d["identifier"] for d in docs)
        start += len(docs)
        if len(docs) < rows:
            break
    return identifiers[:max_items]


def fetch_item_text(identifier: str, retries: int = 3):
    """Returns (text_or_None, metadata_dict_or_None)."""
    meta = None
    for attempt in range(retries):
        try:
            r = requests.get(METADATA_URL.format(identifier=identifier), headers=HEADERS, timeout=60)
            r.raise_for_status()
            meta = r.json()
            break
        except Exception:
            if attempt == retries - 1:
                return None, None
            time.sleep(2 * (attempt + 1))

    files = meta.get("files", [])
    txt_file = next((f["name"] for f in files if f["name"].endswith("_djvu.txt")), None)
    if not txt_file:
        return None, meta
    try:
        r = requests.get(DOWNLOAD_URL.format(identifier=identifier, filename=txt_file), headers=HEADERS, timeout=90)
        r.raise_for_status()
        r.encoding = r.encoding or "utf-8"
        return r.text, meta
    except Exception:
        return None, meta


def main():
    parser = argparse.ArgumentParser(description="Fetch archive.org OCR text into a language's raw/manual/")
    parser.add_argument("--lang", required=True, choices=["hindi", "bhojpuri"])
    parser.add_argument("--query", required=True, help="archive.org advancedsearch query")
    parser.add_argument("--max-items", type=int, default=50)
    parser.add_argument("--min-chars", type=int, default=500)
    args = parser.parse_args()

    lang = args.lang
    lang_iso = LANG_ISO[lang]
    out_dir = PROJECT_ROOT / lang / "data" / "raw" / "manual"
    out_dir.mkdir(parents=True, exist_ok=True)

    identifiers = search_identifiers(args.query, args.max_items)
    print(f"[{lang}] query matched {len(identifiers)} candidate items, fetching...")

    count = skipped_no_ocr = skipped_short = skipped_error = 0
    for i, identifier in enumerate(identifiers, 1):
        text, meta = fetch_item_text(identifier)
        if meta is None:
            skipped_error += 1
            print(f"  [{i}/{len(identifiers)}] {identifier}: metadata fetch failed")
            continue
        if text is None:
            skipped_no_ocr += 1
            continue
        if len(text.strip()) < args.min_chars:
            skipped_short += 1
            continue

        m = meta.get("metadata", {})
        digest = hashlib.sha1(identifier.encode("utf-8")).hexdigest()[:10]
        doc_id = f"{lang_iso}_manual_archiveorg_{digest}"
        record = {
            "document_id": doc_id,
            "language": lang,
            "source_type": "manual",
            "source_name": f"archive.org:{identifier}",
            "source_description": m.get("title", identifier),
            "collection_method": "archive_org_ocr_text",
            "license_or_usage_note": (
                m.get("licenseurl") or m.get("rights")
                or "no explicit rights statement on item - unverified"
            ),
            "url": f"https://archive.org/details/{identifier}",
            "characters": len(text),
            "split": None,
            "cleaning_status": "raw",
        }
        (out_dir / "text").mkdir(parents=True, exist_ok=True)
        (out_dir / "text" / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        with (out_dir / "metadata.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1
        print(f"  [{i}/{len(identifiers)}] {identifier}: {len(text)} chars")
        time.sleep(0.5)  # be polite to archive.org

    print(f"[{lang}] archive.org: wrote {count} documents "
          f"(skipped: {skipped_no_ocr} no-OCR, {skipped_short} too-short, {skipped_error} errors)")
    print(f"[{lang}] Output: {out_dir}")


if __name__ == "__main__":
    main()

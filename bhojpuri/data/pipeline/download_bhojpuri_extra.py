"""Download additional Bhojpuri (Model L) downloaded-source text into
data/raw/downloaded/, on top of what download_bhojpuri.py already collected
(Satyam810/BhojpuriCorpus, wikimedia/wikipedia:20231101.bh, FineWeb-2 bho_Deva).

Kept as a separate script rather than extending download_bhojpuri.py so the
already-working, already-run pipeline stays untouched - this only ever
appends new documents, never wipes or re-derives the existing ~412k
downloaded docs.

Each source is independent and wrapped in try/except that prints the REAL
exception (not a swallowed warning) and moves on - per source, this script
reports whether the dataset actually has Bhojpuri text or not, since several
of the requested sources were unconfirmed at request time.

Shares the metadata.jsonl schema with download_bhojpuri.py exactly, so
source_name is what distinguishes provenance (already proven sufficient for
per-source attribution - see e.g. the existing 13 tagged BhojpuriCorpus
partitions). Uses a short-lived cross-process append lock so multiple source
downloads can run concurrently without corrupting the shared metadata.jsonl -
same failure mode and same fix as scripts/crawl_sites.py's concurrent-crawler
corruption (interleaved writes across processes to one shared file).
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

LANG = "bhojpuri"
OUT_DIR = Path(__file__).resolve().parents[1] / "raw" / "downloaded"
META_PATH = OUT_DIR / "metadata.jsonl"


def document_id(source_name: str, raw_id: str) -> str:
    digest = hashlib.sha1(f"{source_name}:{raw_id}".encode("utf-8")).hexdigest()[:10]
    return f"bho_downloaded_{source_name}_{digest}"


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


def write_document(out_dir: Path, doc_id: str, text: str, meta: dict) -> None:
    (out_dir / "text").mkdir(parents=True, exist_ok=True)
    (out_dir / "text" / f"{doc_id}.txt").write_text(text, encoding="utf-8")
    append_metadata_locked(out_dir / "metadata.jsonl", meta)


def _ingest_rows(ds, source_name, source_desc, license_note, max_docs, text_field="text", url_field="url"):
    count = 0
    chars = 0
    for row in tqdm(ds, desc=source_name):
        text = (row.get(text_field) or "").strip()
        if not text:
            continue
        raw_id = row.get("id", str(count))
        doc_id = document_id(source_name, str(raw_id))
        meta = {
            "document_id": doc_id,
            "language": LANG,
            "source_type": "downloaded",
            "source_name": source_name,
            "source_description": source_desc,
            "collection_method": "download",
            "license_or_usage_note": license_note,
            "url": row.get(url_field) or "",
            "characters": len(text),
            "split": None,
            "cleaning_status": "raw",
        }
        write_document(OUT_DIR, doc_id, text, meta)
        count += 1
        chars += len(text)
        if max_docs is not None and count >= max_docs:
            break
    return count, chars


def try_culturax(max_docs):
    name = "uonlp/CulturaX:bho"
    try:
        ds = load_dataset("uonlp/CulturaX", "bho", split="train", streaming=True)
        n, c = _ingest_rows(ds, name, "CulturaX multilingual pretraining corpus, Bhojpuri config",
                             "per CulturaX terms (mC4 + OSCAR derived, research use)", max_docs)
        print(f"[OK] {name}: {n} docs, {c:,} chars")
        return n, c
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 0, 0


def try_madlad400(max_docs):
    for repo_id in ["allenai/MADLAD-400", "google/madlad-400"]:
        for config in ["bho", "bho_Deva"]:
            name = f"{repo_id}:{config}"
            try:
                ds = load_dataset(repo_id, config, split="clean", streaming=True)
                n, c = _ingest_rows(ds, f"madlad400_{config}", f"MADLAD-400 ({repo_id}, config {config})",
                                     "ODC-BY-1.0", max_docs)
                print(f"[OK] {name}: {n} docs, {c:,} chars")
                return n, c
            except Exception as e:
                print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    return 0, 0


def try_sangraha(max_docs):
    total_n, total_c = 0, 0
    for data_dir in ["verified/bho", "unverified/bho"]:
        name = f"ai4bharat/sangraha:{data_dir}"
        try:
            ds = load_dataset("ai4bharat/sangraha", data_dir=data_dir, split="train", streaming=True)
            n, c = _ingest_rows(ds, f"sangraha_{data_dir.replace('/', '_')}",
                                 f"AI4Bharat Sangraha, {data_dir}", "per Sangraha terms", max_docs)
            print(f"[OK] {name}: {n} docs, {c:,} chars")
            total_n += n
            total_c += c
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    return total_n, total_c


def try_cc100(max_docs):
    name = "statmt/cc100:bho"
    try:
        ds = load_dataset("statmt/cc100", lang="bho", split="train", streaming=True)
        n, c = _ingest_rows(ds, name, "CC-100 Bhojpuri (CommonCrawl-derived)", "CC0/permissive per statmt", max_docs)
        print(f"[OK] {name}: {n} docs, {c:,} chars")
        return n, c
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 0, 0


def try_glot500(max_docs):
    name = "cis-lmu/Glot500:bho_Deva"
    try:
        ds = load_dataset("cis-lmu/Glot500", "bho_Deva", split="train", streaming=True)
        n, c = _ingest_rows(ds, name, "Glot500-c Bhojpuri Devanagari", "per Glot500 terms (mixed sources)", max_docs)
        print(f"[OK] {name}: {n} docs, {c:,} chars")
        return n, c
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 0, 0


def try_bloomlm(max_docs):
    name = "sil-ai/bloom-lm:bho"
    try:
        ds = load_dataset("sil-ai/bloom-lm", "bho", split="train", streaming=True)
        n, c = _ingest_rows(ds, name, "SIL Bloom Library children's books, Bhojpuri", "CC-BY-4.0 (per Bloom Library)", max_docs)
        print(f"[OK] {name}: {n} docs, {c:,} chars")
        return n, c
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 0, 0


SOURCES = {
    "culturax": try_culturax,
    "madlad400": try_madlad400,
    "sangraha": try_sangraha,
    "cc100": try_cc100,
    "glot500": try_glot500,
    "bloomlm": try_bloomlm,
}


def main():
    parser = argparse.ArgumentParser(description="Download additional Bhojpuri sources (Model L)")
    parser.add_argument("--sources", nargs="+", default=list(SOURCES.keys()), choices=list(SOURCES.keys()))
    parser.add_argument("--max-docs-per-source", type=int, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for src in args.sources:
        print(f"\n=== {src} ===")
        n, c = SOURCES[src](args.max_docs_per_source)
        results[src] = (n, c)

    print("\n=== SUMMARY ===")
    total_n = total_c = 0
    for src, (n, c) in results.items():
        status = "OK" if n > 0 else "NO DATA (see [FAIL]/[OK] lines above for reason)"
        print(f"  {src:12s} {n:>8,} docs  {c:>14,} chars   [{status}]")
        total_n += n
        total_c += c
    print(f"  TOTAL new: {total_n:,} docs, {total_c:,} chars")


if __name__ == "__main__":
    main()

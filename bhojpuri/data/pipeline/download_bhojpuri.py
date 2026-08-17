"""Download Bhojpuri (Model L) downloaded-source text into data/raw/downloaded/.

Verified real sources (checked against the HF Hub API before writing this script,
not assumed):
  - Satyam810/BhojpuriCorpus : 386,032 docs, ~25M estimated tokens, aggregated and
    already-deduplicated Bhojpuri text (Wikipedia + web crawl + ASR transcripts +
    literary/academic corpora). Per-partition licenses are mostly CC-BY-4.0 / CC0 /
    CC-BY-SA-4.0, with two partitions (jnu_bhltr, iit_bhu) under CC-BY-NC-SA-4.0
    (non-commercial - acceptable for this coursework use, noted in metadata).
  - wikimedia/wikipedia, config 20231101.bh : the Bhojpuri Wikipedia dump via HF's
    parquet mirror. Pulled independently of BhojpuriCorpus's own (smaller, 7,603-doc)
    wikipedia partition; exact-duplicate documents between the two are expected and
    are removed later by the document-level dedup step, not by this script.

AI4Bharat Sangraha was investigated and does NOT include a Bhojpuri split (its
language tag list has no "bho"), so it is intentionally not used here.

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

LANG = "bhojpuri"
LANG_ID = "bho"
OUT_DIR = Path(__file__).resolve().parents[1] / "raw" / "downloaded"

# Per-partition license map for Satyam810/BhojpuriCorpus, taken from its dataset
# card's "Source Distribution" table (not guessed).
BHOJPURICORPUS_PARTITION_LICENSE = {
    "fish_food": "CC-BY-4.0",
    "jnu_bhltr": "CC-BY-NC-SA-4.0",
    "eng_bho": "CC-BY-4.0",
    "bhojpuri_lr_v3": "CC-BY-4.0",
    "hplt": "CC0",
    "wikipedia": "CC-BY-SA-4.0",
    "kumar_bhojpuri": "CC-BY-4.0",
    "iit_bhu": "CC-BY-NC-SA-4.0",
    "fleurs_bhojpuri": "CC-BY-4.0",
    "google_smol": "CC-BY-4.0",
    "glotcc": "CC-BY-4.0",
    "rural_women": "CC-BY-4.0",
    "ud_bhtb": "CC-BY-SA-4.0",
}


def document_id(source_name: str, raw_id: str) -> str:
    """Build a stable, collision-resistant document id for the metadata schema."""
    digest = hashlib.sha1(f"{source_name}:{raw_id}".encode("utf-8")).hexdigest()[:10]
    return f"bho_downloaded_{source_name}_{digest}"


def write_document(out_dir: Path, doc_id: str, text: str, meta: dict) -> None:
    (out_dir / "text").mkdir(parents=True, exist_ok=True)
    (out_dir / "text" / f"{doc_id}.txt").write_text(text, encoding="utf-8")
    meta_path = out_dir / "metadata.jsonl"
    with meta_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")


def download_bhojpuricorpus(out_dir: Path, max_docs: int | None) -> int:
    source_name = "bhojpuricorpus"
    print(f"[{LANG}] streaming Satyam810/BhojpuriCorpus ...")
    ds = load_dataset("Satyam810/BhojpuriCorpus", split="train", streaming=True)
    count = 0
    for row in tqdm(ds, desc="BhojpuriCorpus"):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        raw_id = row.get("id", str(count))
        partition = row.get("source", "unknown")
        license_ = BHOJPURICORPUS_PARTITION_LICENSE.get(partition, row.get("license", "unknown"))
        doc_id = document_id(source_name, raw_id)
        meta = {
            "document_id": doc_id,
            "language": LANG,
            "source_type": "downloaded",
            "source_name": f"Satyam810/BhojpuriCorpus:{partition}",
            "source_description": "Aggregated Bhojpuri pretraining corpus (Wikipedia, web crawl, ASR transcripts, literary/academic text)",
            "collection_method": "download",
            "license_or_usage_note": license_,
            "url": row.get("url") or "",
            "characters": len(text),
            "split": None,
            "cleaning_status": "raw",
        }
        write_document(out_dir, doc_id, text, meta)
        count += 1
        if max_docs is not None and count >= max_docs:
            break
    print(f"[{LANG}] BhojpuriCorpus: wrote {count} documents")
    return count


def download_wikipedia(out_dir: Path, max_docs: int | None) -> int:
    source_name = "wikipedia"
    print(f"[{LANG}] streaming wikimedia/wikipedia 20231101.bh ...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.bh", split="train", streaming=True)
    count = 0
    for row in tqdm(ds, desc="bh-wikipedia"):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        raw_id = row.get("id", str(count))
        doc_id = document_id(source_name, raw_id)
        meta = {
            "document_id": doc_id,
            "language": LANG,
            "source_type": "downloaded",
            "source_name": "wikimedia/wikipedia:20231101.bh",
            "source_description": "Bhojpuri Wikipedia dump (HF parquet mirror, 2023-11-01 snapshot)",
            "collection_method": "download",
            "license_or_usage_note": "CC-BY-SA-4.0",
            "url": row.get("url") or "",
            "characters": len(text),
            "split": None,
            "cleaning_status": "raw",
        }
        write_document(out_dir, doc_id, text, meta)
        count += 1
        if max_docs is not None and count >= max_docs:
            break
    print(f"[{LANG}] Wikipedia: wrote {count} documents")
    return count


def main():
    parser = argparse.ArgumentParser(description="Download Bhojpuri source text (Model L)")
    parser.add_argument("--max-docs-per-source", type=int, default=None,
                         help="Cap docs per source (omit for full download; both sources are small)")
    parser.add_argument("--sources", nargs="+", default=["bhojpuricorpus", "wikipedia"],
                         choices=["bhojpuricorpus", "wikipedia"])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Fresh metadata file per run (this script is meant to be re-run from scratch,
    # not appended to across multiple partial runs).
    meta_path = OUT_DIR / "metadata.jsonl"
    if meta_path.exists():
        meta_path.unlink()

    total = 0
    if "bhojpuricorpus" in args.sources:
        total += download_bhojpuricorpus(OUT_DIR, args.max_docs_per_source)
    if "wikipedia" in args.sources:
        total += download_wikipedia(OUT_DIR, args.max_docs_per_source)

    print(f"[{LANG}] TOTAL documents downloaded: {total}")
    print(f"[{LANG}] Output: {OUT_DIR}")


if __name__ == "__main__":
    main()

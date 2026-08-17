"""One-time backfill: add a `dedup_key` field to every existing record in
data/cleaned/metadata.jsonl that doesn't already have one, by reading its
text file and computing the hash once. After this runs, clean_*.py's
--resume no longer needs to reopen individual cleaned text files to rebuild
its dedup-hash set (see clean_lib.load_resume_state) - it can read the hash
straight out of metadata.jsonl instead, which is what made resume unusably
slow at ~440k+ cleaned docs (100 minutes wall clock for 6 minutes of actual
CPU work, disk I/O for individual file opens being the bottleneck).

Rewrites metadata.jsonl in place (via a temp file + atomic replace) rather
than appending, since every existing line needs the new field added.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_lib import dedup_key  # noqa: E402

WORKERS = 32  # I/O-bound (individual small-file opens); GIL releases during file reads,
              # so threading gives real parallelism here despite the GIL - the earlier
              # sequential version was the actual bottleneck (100 min for ~440k files).


def _read_and_hash(args):
    idx, meta, text_dir = args
    key = meta.get("dedup_key")
    if key is not None:
        return idx, meta, False, False
    text_path = text_dir / f"{meta['document_id']}.txt"
    if not text_path.exists():
        return idx, meta, False, True
    meta["dedup_key"] = dedup_key(text_path.read_text(encoding="utf-8"))
    return idx, meta, True, False


def backfill(cleaned_dir: Path) -> None:
    meta_path = cleaned_dir / "metadata.jsonl"
    text_dir = cleaned_dir / "text"
    tmp_path = meta_path.with_suffix(".jsonl.tmp")

    records = []
    with meta_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[{cleaned_dir}] loaded {len(records):,} records, backfilling with {WORKERS} threads...")

    start = time.time()
    already_had = added = missing_text = 0
    results = [None] * len(records)
    work = [(i, m, text_dir) for i, m in enumerate(records)]
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for idx, meta, was_added, was_missing in pool.map(_read_and_hash, work):
            results[idx] = meta
            already_had += 0 if (was_added or was_missing) else 1
            added += 1 if was_added else 0
            missing_text += 1 if was_missing else 0
            done += 1
            if done % 50000 == 0:
                elapsed = time.time() - start
                print(f"  ...{done:,}/{len(records):,} processed ({elapsed:.0f}s elapsed, "
                      f"{done/elapsed:.0f} docs/s)")

    with tmp_path.open("w", encoding="utf-8") as fout:
        for meta in results:
            fout.write(json.dumps(meta, ensure_ascii=False) + "\n")
    tmp_path.replace(meta_path)

    print(f"[{cleaned_dir}] total={len(records):,} already_had_key={already_had:,} "
          f"backfilled={added:,} missing_text_file={missing_text:,} "
          f"({time.time()-start:.0f}s total)")


def main():
    parser = argparse.ArgumentParser(description="Backfill dedup_key into an existing cleaned/metadata.jsonl")
    parser.add_argument("--lang", required=True, choices=["hindi", "bhojpuri"])
    args = parser.parse_args()
    cleaned_dir = Path(__file__).resolve().parent.parent / args.lang / "data" / "cleaned"
    backfill(cleaned_dir)


if __name__ == "__main__":
    main()

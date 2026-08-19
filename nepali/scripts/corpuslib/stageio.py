"""
I/O for pipeline stages S03 onward.

``ShardWriter`` (see :mod:`jsonlio`) is deliberately narrow: one writer owns
one ``(lang, lane, source_id)`` triple, because that is what makes the raw
collection layer's provenance guarantees enforceable. A pipeline stage after
collection does the opposite thing -- it reads documents from *every* source
and lane together, drops some, and writes the rest back out as one pool -- so
reusing ``ShardWriter`` there would fight its own validation on every record.

This module is the equivalent for that shape: plain sharded JSONL, written
atomically (via :func:`jsonlio.write_jsonl`, one ``.part`` -> rename per
shard), with a stage manifest recording what went in, what came out, and why
the difference is what it is. Every stage script writes its stage manifest
before touching the next stage's input, so a stage can always be re-run in
isolation and inspected without re-running the ones before it.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import jsonlio

STAGE_MANIFEST_NAME = "_stage_manifest.json"
#: Where persisted copies of stage manifests live once run_pipeline.py's
#: rolling cleanup has deleted the stage's own (large) output directory.
STATS_DIRNAME = "stats"


def iter_raw_documents(raw_dir: Path, *, lane: str | None = None) -> Iterator[dict[str, Any]]:
    """Yield every collected document across all sources (and both lanes).

    This is the one place S03 (the first post-collection stage) reads from --
    every enabled source's shards, regardless of which collector produced
    them, found the same way :class:`~corpuslib.ledger.Ledger` finds them: by
    walking for ``_manifest.json`` markers rather than assuming a fixed set of
    source directories.
    """
    for source_dir in jsonlio.iter_source_dirs(raw_dir):
        manifest = jsonlio.read_json(source_dir / jsonlio.MANIFEST_NAME, default={})
        if lane is not None and manifest.get("lane") != lane:
            continue
        yield from jsonlio.read_shards(source_dir)


def read_stage(stage_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield every record a previous stage wrote."""
    yield from jsonlio.read_shards(stage_dir)


def stage_manifest(stage_dir: Path) -> dict[str, Any]:
    """A stage's manifest -- from its own directory, or from the persisted
    copy under ``interim/stats/`` if that directory has since been deleted
    by run_pipeline.py's rolling per-stage cleanup (see :func:`persist_stats`).
    """
    stage_dir = Path(stage_dir)
    manifest = jsonlio.read_json(stage_dir / STAGE_MANIFEST_NAME, default=None)
    if manifest is not None:
        return manifest
    persisted = stage_dir.parent / STATS_DIRNAME / f"{stage_dir.name}.json"
    return jsonlio.read_json(persisted, default={}) or {}


def persist_stats(interim_dir: Path, stage_output_name: str) -> None:
    """Copy one stage output's manifest into ``interim/stats/`` so later
    steps (chiefly ``s14_corpus_stats.py``'s retention funnel) can still read
    it after the stage's own -- much larger -- output directory is deleted.
    """
    src = Path(interim_dir) / stage_output_name / STAGE_MANIFEST_NAME
    if not src.exists():
        return
    dest_dir = Path(interim_dir) / STATS_DIRNAME
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest_dir / f"{stage_output_name}.json")


def write_stage(
    stage_dir: Path,
    records: Iterable[dict[str, Any]],
    *,
    docs_per_shard: int = 25_000,
) -> int:
    """Write ``records`` as fresh shards under ``stage_dir``. Returns the count.

    Always a full, fresh write -- there is no append or resume mode here, for
    the same reason :mod:`jsonlio` gives ``ShardWriter`` none: a stage that
    can silently double its own output is a stage nobody can trust the
    retention numbers from.
    """
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    for stale in stage_dir.glob("part-*.jsonl"):
        stale.unlink()
    for stale in stage_dir.glob("*.part"):
        stale.unlink()

    n = 0
    batch: list[dict[str, Any]] = []
    shard_index = 0
    for record in records:
        batch.append(record)
        if len(batch) >= docs_per_shard:
            jsonlio.write_jsonl(stage_dir / f"part-{shard_index:05d}.jsonl", batch)
            n += len(batch)
            shard_index += 1
            batch = []
    if batch:
        jsonlio.write_jsonl(stage_dir / f"part-{shard_index:05d}.jsonl", batch)
        n += len(batch)
    return n


def write_stage_manifest(stage_dir: Path, payload: dict[str, Any]) -> None:
    jsonlio.atomic_write_json(Path(stage_dir) / STAGE_MANIFEST_NAME, payload)


def write_samples(
    samples_dir: Path,
    reason: str,
    kept: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
    *,
    n: int = 20,
) -> None:
    """Write up to ``n`` kept and ``n`` dropped documents for one drop reason.

    The plan's review discipline for every filtering stage: a reviewer reads
    real kept/dropped text side by side rather than trusting a retention
    percentage on its own.
    """
    samples_dir = Path(samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)
    safe_reason = "".join(c if c.isalnum() or c in "_-" else "_" for c in reason)[:80]
    jsonlio.write_jsonl(samples_dir / f"{safe_reason}__kept.jsonl", kept[:n])
    jsonlio.write_jsonl(samples_dir / f"{safe_reason}__dropped.jsonl", dropped[:n])

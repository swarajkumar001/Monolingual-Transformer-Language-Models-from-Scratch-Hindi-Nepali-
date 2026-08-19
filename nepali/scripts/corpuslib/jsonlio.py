"""
Crash-safe, append-proof JSONL I/O for corpus shards.

The previous pipeline lost a week of trust to a single default argument:
``write_jsonl(path, records, mode='a')``. Re-running any collector silently
doubled its output, so every raw file held each document twice and every
downstream count was inflated. Nothing in this module can append to a
finished file:

* Writers own a *directory* of numbered shards, never a single growing file.
* Each shard is written to ``.part`` and atomically renamed on close, so an
  interrupted run leaves no half-written shard for the next stage to read.
* Opening a writer over an existing directory requires an explicit policy --
  ``fresh`` (delete prior shards) or ``resume`` (continue after the last
  complete shard). There is no implicit third option.

A ``_manifest.json`` sits beside the shards recording counts, byte sizes and
the source registry entry, so any stage can report on a source without
re-reading its text.

If an open shard's output directory is deleted out from under a running
writer -- observed once, when an unrelated `rm -rf` on `data/raw` collided
with a live overnight run -- the rename that commits the shard fails with
``FileNotFoundError`` and its documents are unrecoverable (they lived only in
an unlinked inode, freed the moment the handle closed). ``ShardWriter`` rolls
those documents back out of its own counters rather than reporting them as
collected, and stops the source after two such losses in a row rather than
spending its whole time budget re-losing shards one after another.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import schema

MANIFEST_NAME = "_manifest.json"
SHARD_GLOB = "part-*.jsonl"

#: How many shards in a row may be lost to a vanished output directory before
#: :class:`ShardWriter` gives up. One loss is tolerated and self-healed (some
#: other process may have transiently touched the tree); a second consecutive
#: loss means something is actively deleting `data/raw` while we write to it,
#: and continuing would just keep manufacturing unrecoverable data.
MAX_CONSECUTIVE_SHARD_LOSSES = 2


class ShardWriteError(RuntimeError):
    """A shard's temp file disappeared before it could be committed.

    This means something outside this process removed the writer's output
    directory (or the specific `.part` file) while a shard was open. Any
    documents already written to that shard lived only in the now-unlinked
    inode and are gone the moment the handle closes -- there is no path left
    to read them back from. ``ShardWriter`` rolls those documents back out of
    its own stats so a manifest never reports words that are not on disk.
    """


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON to ``path`` such that readers never observe a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


@dataclass(slots=True)
class ShardStats:
    docs: int = 0
    words: int = 0
    chars: int = 0
    bytes: int = 0
    shards: int = 0
    #: Documents/words that were written and counted in-memory but whose
    #: shard could not be committed to disk (see ``ShardWriteError``). Kept
    #: separate from ``docs``/``words`` so a manifest never claims progress
    #: that the corresponding bytes on disk cannot back up.
    lost_docs: int = 0
    lost_words: int = 0


class ShardWriter:
    """Writes validated :class:`~corpuslib.schema.Document` records as shards.

    Parameters
    ----------
    out_dir:
        Directory that will contain ``part-00000.jsonl`` .. and the manifest.
        The directory belongs exclusively to one source; never share it.
    lang / lane / source_id:
        The provenance triple. Every record is checked against these, so a
        collector cannot write a Nepali document into the Hindi corpus or
        mislabel a downloaded dump as manual.
    mode:
        ``"fresh"`` removes any existing shards first; ``"resume"`` keeps
        complete shards and continues numbering after them.
    docs_per_shard:
        Shard rollover threshold. Smaller shards mean less work lost to an
        interrupted run and let downstream stages parallelise cleanly.
    """

    def __init__(
        self,
        out_dir: Path,
        *,
        lang: str,
        lane: str,
        source_id: str,
        mode: str = "fresh",
        docs_per_shard: int = 25_000,
        flush_every: int = 500,
    ) -> None:
        if mode not in {"fresh", "resume"}:
            raise ValueError(f"mode must be 'fresh' or 'resume', got {mode!r}")
        if lane not in schema.VALID_LANES:
            raise ValueError(f"unknown lane {lane!r}")

        self.out_dir = Path(out_dir)
        self.lang = lang
        self.lane = lane
        self.source_id = source_id
        self.docs_per_shard = docs_per_shard
        self.flush_every = flush_every

        self.out_dir.mkdir(parents=True, exist_ok=True)
        if mode == "fresh":
            self._clear()

        self.stats = ShardStats()
        self._seen: set[str] = set()
        self._shard_index = 0
        self._handle = None
        self._docs_in_shard = 0
        self._since_flush = 0
        # Per-shard shadow counters, reset on every _open_shard(). If the
        # in-progress shard is lost, these are what get subtracted back out
        # of self.stats and folded into self.stats.lost_docs/lost_words.
        self._shard_doc_ids: list[str] = []
        self._shard_words = 0
        self._shard_chars = 0
        self._consecutive_shard_losses = 0

        if mode == "resume":
            self._adopt_existing()

    # -- lifecycle ---------------------------------------------------------

    def _clear(self) -> None:
        for stale in self.out_dir.glob(SHARD_GLOB):
            stale.unlink()
        for stale in self.out_dir.glob("*.part"):
            stale.unlink()
        (self.out_dir / MANIFEST_NAME).unlink(missing_ok=True)

    def _adopt_existing(self) -> None:
        """Re-read complete shards so a resumed run does not re-emit documents."""
        for stale in self.out_dir.glob("*.part"):
            stale.unlink()  # incomplete by definition
        shards = sorted(self.out_dir.glob(SHARD_GLOB))
        for shard in shards:
            with shard.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._seen.add(record["doc_id"])
                    self.stats.docs += 1
                    self.stats.words += record.get("n_words", 0)
                    self.stats.chars += record.get("n_chars", 0)
            self.stats.bytes += shard.stat().st_size
        self.stats.shards = len(shards)
        self._shard_index = len(shards)

    def _open_shard(self) -> None:
        # Self-heal a directory that vanished out from under us (see
        # _close_shard's ShardWriteError handling below): if it is still
        # missing, the fresh open below fails loudly and immediately rather
        # than silently drifting into another lost shard.
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._pending_path = self.out_dir / f"part-{self._shard_index:05d}.jsonl"
        self._pending_tmp = self.out_dir / f"part-{self._shard_index:05d}.jsonl.part"
        self._handle = self._pending_tmp.open("w", encoding="utf-8")
        self._docs_in_shard = 0
        self._shard_doc_ids = []
        self._shard_words = 0
        self._shard_chars = 0

    def _close_shard(self) -> None:
        if self._handle is None:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._handle = None
        if self._docs_in_shard == 0:
            self._pending_tmp.unlink(missing_ok=True)
            return
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            os.replace(self._pending_tmp, self._pending_path)
        except FileNotFoundError as exc:
            # The temp file (or its directory) is gone: something outside
            # this process deleted data/raw while the shard was open. The
            # documents already "written" to it existed only in that unlinked
            # inode and were freed the instant handle.close() ran above --
            # there is nothing left to rename or recover. Roll the shard's
            # documents back out of the live stats (so the manifest never
            # reports words that are not on disk) and record the loss instead
            # of letting it masquerade as counted progress.
            self.stats.docs -= self._docs_in_shard
            self.stats.words -= self._shard_words
            self.stats.chars -= self._shard_chars
            self.stats.lost_docs += self._docs_in_shard
            self.stats.lost_words += self._shard_words
            self._seen.difference_update(self._shard_doc_ids)
            self._shard_index += 1  # never reuse a burned shard number
            self._consecutive_shard_losses += 1
            if self._consecutive_shard_losses >= MAX_CONSECUTIVE_SHARD_LOSSES:
                raise ShardWriteError(
                    f"{self.source_id}: lost {self._consecutive_shard_losses} "
                    f"consecutive shards under {self.out_dir} -- its directory "
                    f"keeps disappearing while this writer holds shards open, "
                    f"most likely because another process is deleting data/raw "
                    f"during collection. Stopping this source rather than "
                    f"continuing to write into a directory that will not hold "
                    f"still."
                ) from exc
            return
        self.stats.bytes += self._pending_path.stat().st_size
        self.stats.shards += 1
        self._shard_index += 1
        self._consecutive_shard_losses = 0

    # -- writing -----------------------------------------------------------

    def write(self, doc: schema.Document) -> bool:
        """Persist ``doc``. Returns False if it was a within-source duplicate.

        Exact duplicates are dropped here rather than in a later dedup stage
        because they are free to detect at write time and because letting them
        reach disk is what inflates every downstream count.
        """
        if doc.lang != self.lang:
            raise schema.SchemaError(
                f"{doc.doc_id}: {doc.lang!r} document offered to the {self.lang!r} writer"
            )
        if doc.lane != self.lane:
            raise schema.SchemaError(
                f"{doc.doc_id}: lane {doc.lane!r} != writer lane {self.lane!r} "
                f"(source {self.source_id!r} is registered to one lane only)"
            )
        if doc.source_id != self.source_id:
            raise schema.SchemaError(
                f"{doc.doc_id}: source_id {doc.source_id!r} != writer {self.source_id!r}"
            )

        if doc.doc_id in self._seen:
            return False

        record = doc.to_dict()
        schema.validate(record, expect_lang=self.lang)

        if self._handle is None or self._docs_in_shard >= self.docs_per_shard:
            self._close_shard()
            self._open_shard()

        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._seen.add(doc.doc_id)
        self._docs_in_shard += 1
        self._shard_doc_ids.append(doc.doc_id)
        self._shard_words += doc.n_words
        self._shard_chars += doc.n_chars
        self.stats.docs += 1
        self.stats.words += doc.n_words
        self.stats.chars += doc.n_chars

        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self._handle.flush()
            self._since_flush = 0
        return True

    def write_manifest(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        manifest = {
            "source_id": self.source_id,
            "lang": self.lang,
            "lane": self.lane,
            "docs": self.stats.docs,
            "words_estimate": self.stats.words,
            "chars": self.stats.chars,
            "bytes": self.stats.bytes,
            "shards": self.stats.shards,
            "lost_docs": self.stats.lost_docs,
            "lost_words": self.stats.lost_words,
            "schema_version": schema.SCHEMA_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if extra:
            manifest.update(extra)
        atomic_write_json(self.out_dir / MANIFEST_NAME, manifest)
        return manifest

    def close(self, extra_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        """Flush the current shard and write the manifest. Never raises.

        ``Collector.run()`` calls this from its ``finally`` clause, which is
        the last line of defence for the "a source's failure never crashes
        the run" contract in ``base.py``. A ``ShardWriteError`` raised here
        (the final shard happening to be the one that loses its directory)
        has nothing left to protect -- iteration is already over -- so it is
        swallowed rather than propagated; the loss is already folded into
        ``self.stats.lost_docs``/``lost_words`` by the time it is raised, and
        that is what the manifest reports.
        """
        try:
            self._close_shard()
        except ShardWriteError:
            pass
        return self.write_manifest(extra_manifest)

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# -- reading ---------------------------------------------------------------


def read_shards(directory: Path) -> Iterator[dict[str, Any]]:
    """Yield every record from a shard directory, in deterministic order."""
    for shard in sorted(Path(directory).glob(SHARD_GLOB)):
        with shard.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield every record from a single JSONL file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records) -> int:
    """Write ``records`` to ``path`` atomically, truncating any prior content.

    There is no ``mode`` parameter. Appending to a corpus file is never the
    right behaviour for this pipeline and the option is therefore not offered.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    written = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = record.to_dict() if hasattr(record, "to_dict") else record
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return written


def count_lines(path: Path) -> int:
    if not Path(path).exists():
        return 0
    total = 0
    with Path(path).open("rb") as handle:
        for _ in handle:
            total += 1
    return total


def iter_source_dirs(root: Path) -> Iterator[Path]:
    """Yield every shard directory beneath ``root`` that carries a manifest."""
    for manifest in sorted(Path(root).rglob(MANIFEST_NAME)):
        yield manifest.parent


def collect_manifests(root: Path) -> list[dict[str, Any]]:
    """Load every source manifest under ``root``, newest field set first."""
    out = []
    for directory in iter_source_dirs(root):
        payload = read_json(directory / MANIFEST_NAME)
        if payload:
            payload["path"] = str(directory)
            out.append(payload)
    return out


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def free_disk_gb(path: Path) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024 ** 3)

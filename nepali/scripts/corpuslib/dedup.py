"""
Deduplication, with cross-lane collisions resolved in favour of manual text.

Exact duplicates are already impossible within a source -- ``doc_id`` is a
content hash and the shard writer rejects repeats -- so this module handles the
two cases that survive: identical text arriving from *different* sources, and
near-duplicates.

The rule that matters
---------------------
When a near-duplicate pair straddles the manual and downloaded lanes, the
**manual copy is kept and the downloaded copy dropped**. Sangraha and IndicCorp
are built from web crawls that already include Wikipedia and Indian news sites,
so overlap with our own scraping is expected and was measured by the user at a
significant rate. A lane-blind dedup pass would delete manual documents at
random and silently erode the 20% requirement -- which is precisely how the
previous run's manual share evaporated.

Beyond preserving the requirement, the resolution is also correct on the
merits: for a duplicated document we would rather keep the copy whose
extraction and cleaning we control and can describe in the report.

Parameters follow the lecture's worked example: 100 permutations, ``b=20``
bands of ``r=5`` rows, giving a similarity threshold near ``(1/b)^(1/r) ≈
0.55``. At that setting a 0.8-similar pair is caught ~99.96% of the time while
a 0.3-similar pair is caught ~4.7% of the time.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from . import schema

_WORD_RE = re.compile(r"\S+")

DEFAULT_NUM_PERM = 100
DEFAULT_BANDS = 20
DEFAULT_ROWS = 5
DEFAULT_SHINGLE = 5

#: Below this many documents, process-pool startup overhead outweighs the
#: gain -- just hash inline.
MIN_DOCS_FOR_PARALLEL = 5_000

#: Default cap on parallel signature workers -- see compute_signatures().
DEFAULT_MAX_WORKERS = 4


@dataclass
class DedupStats:
    input_docs: int = 0
    exact_removed: int = 0
    near_removed: int = 0
    cross_lane_pairs: int = 0
    manual_saved_by_priority: int = 0
    removed_by_lane: Counter[str] = field(default_factory=Counter)
    kept: int = 0

    def to_dict(self) -> dict[str, Any]:
        total_removed = self.exact_removed + self.near_removed
        return {
            "input_docs": self.input_docs,
            "exact_duplicates_removed": self.exact_removed,
            "near_duplicates_removed": self.near_removed,
            "total_removed": total_removed,
            "kept": self.kept,
            "retention": round(self.kept / self.input_docs, 4) if self.input_docs else 0.0,
            "cross_lane_duplicate_pairs": self.cross_lane_pairs,
            "manual_docs_preserved_by_priority": self.manual_saved_by_priority,
            "removed_by_lane": dict(self.removed_by_lane),
        }


def shingles(text: str, k: int = DEFAULT_SHINGLE) -> set[str]:
    """Word-level k-shingles.

    Word shingles rather than character shingles: Devanagari words carry more
    signal per unit than characters, and k=5 words is the middle of the 5-9
    range production pipelines use.
    """
    words = _WORD_RE.findall(text)
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def _lane_rank(lane: str) -> int:
    """Sort key making manual documents win duplicate resolution."""
    return 0 if lane == schema.LANE_MANUAL else 1


def remove_exact(
    records: Iterable[dict[str, Any]], stats: DedupStats | None = None
) -> list[dict[str, Any]]:
    """Drop repeated ``doc_id`` values, keeping the manual copy of each.

    Records are grouped rather than streamed because the manual-priority rule
    needs to see every copy of a document before choosing which to keep.
    """
    stats = stats if stats is not None else DedupStats()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        stats.input_docs += 1
        groups[record["doc_id"]].append(record)

    kept: list[dict[str, Any]] = []
    for copies in groups.values():
        if len(copies) == 1:
            kept.append(copies[0])
            continue
        copies.sort(key=lambda r: (_lane_rank(r["lane"]), r["source_id"]))
        winner = copies[0]
        lanes = {c["lane"] for c in copies}
        if len(lanes) > 1:
            stats.cross_lane_pairs += 1
            if winner["lane"] == schema.LANE_MANUAL:
                stats.manual_saved_by_priority += 1
        for loser in copies[1:]:
            stats.exact_removed += 1
            stats.removed_by_lane[loser["lane"]] += 1
        winner.setdefault("meta", {})["exact_duplicate_copies"] = len(copies)
        kept.append(winner)
    return kept


class MinHasher:
    """MinHash signatures without a heavyweight dependency.

    ``datasketch`` is a fine library, but a signature here is 100 integers
    derived from cheap hashes, and implementing it directly keeps the
    dependency surface small and makes the parameters visible in the report.

    Each permutation's ``(a*h + b) mod p`` is computed with Python's
    arbitrary-precision ints -- ``a`` and ``h`` are each up to ~2^61, so the
    product is up to ~2^122 and would silently overflow a fixed-width numpy
    int. Only the *result* (already reduced mod the Mersenne prime, so it
    fits safely under 2^61) is stored in a numpy ``uint64`` array instead of
    a Python tuple -- ~800 bytes per signature instead of ~2.9KB, which is
    the difference between a few million documents' worth of signatures
    fitting comfortably in memory and not, when two languages' pipelines run
    at once.
    """

    _MERSENNE = (1 << 61) - 1

    def __init__(self, num_perm: int = DEFAULT_NUM_PERM, seed: int = 20260817) -> None:
        import random

        self.num_perm = num_perm
        rng = random.Random(seed)
        # a*x + b mod p, the standard universal hash family.
        self._a = [rng.randrange(1, self._MERSENNE) for _ in range(num_perm)]
        self._b = [rng.randrange(0, self._MERSENNE) for _ in range(num_perm)]

    def signature(self, tokens: set[str]) -> np.ndarray:
        if not tokens:
            return np.full(self.num_perm, self._MERSENNE, dtype=np.uint64)
        base = [
            int.from_bytes(hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest(), "big")
            for t in tokens
        ]
        sig = np.empty(self.num_perm, dtype=np.uint64)
        for i, (a, b) in enumerate(zip(self._a, self._b)):
            sig[i] = min(((a * h + b) % self._MERSENNE) for h in base)
        return sig


def estimate_similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.count_nonzero(sig_a == sig_b)) / len(sig_a)


def _band_key(signature: np.ndarray, start: int, end: int) -> tuple:
    """A hashable dict key for one LSH band -- numpy arrays aren't hashable."""
    return tuple(signature[start:end].tolist())


# ---------------------------------------------------------------------------
# Parallel signature computation
# ---------------------------------------------------------------------------
# Computing a MinHash signature for one document (shingle it, hash every
# shingle, take 100 running minimums) does not depend on any other document,
# so it is the one part of dedup that parallelises without touching the
# algorithm's output. The LSH bucket insertion that follows still has to run
# in a single pass, in order, because manual-priority depends on seeing
# earlier documents first -- only the expensive hashing is farmed out.
#
# Each worker process builds its own MinHasher once (via the pool
# initializer) rather than re-deriving the 100 hash-family coefficients per
# document, and rather than pickling the hasher itself across the process
# boundary.

_worker_hasher: "MinHasher | None" = None
_worker_shingle_size: int = DEFAULT_SHINGLE


def _init_worker(num_perm: int, seed: int, shingle_size: int) -> None:
    global _worker_hasher, _worker_shingle_size
    _worker_hasher = MinHasher(num_perm, seed=seed)
    _worker_shingle_size = shingle_size


def _signature_task(text: str) -> np.ndarray:
    assert _worker_hasher is not None
    return _worker_hasher.signature(shingles(text, _worker_shingle_size))


def compute_signatures(
    texts: list[str],
    *,
    num_perm: int = DEFAULT_NUM_PERM,
    shingle_size: int = DEFAULT_SHINGLE,
    seed: int = 20260817,
    workers: int | None = None,
) -> list[np.ndarray]:
    """MinHash signatures for every text, in input order.

    Deterministic regardless of ``workers`` -- each signature depends only on
    its own text and the fixed ``(num_perm, seed)`` hash family, never on
    what else is in the batch. Falls back to inline hashing for small batches
    or ``workers=1``, where process-pool startup would cost more than it saves.

    Default worker count is capped at ``DEFAULT_MAX_WORKERS`` (4) rather than
    "every core but one" -- this module has no way to know whether Hindi and
    Nepali are being run at the same time in two separate terminals, and each
    worker process holds its own chunk of document text plus a fresh
    ``MinHasher``, so "all cores" per language becomes "all cores twice" and
    real memory/CPU contention if both are running. Pass ``workers=`` (or
    ``--workers`` on the stage scripts) explicitly for more, if you know only
    one language is running.
    """
    if workers is None:
        workers = max(1, min(DEFAULT_MAX_WORKERS, (os.cpu_count() or 2) - 1))
    if workers <= 1 or len(texts) < MIN_DOCS_FOR_PARALLEL:
        hasher = MinHasher(num_perm, seed=seed)
        return [hasher.signature(shingles(t, shingle_size)) for t in texts]

    chunksize = max(1, len(texts) // (workers * 8))
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(num_perm, seed, shingle_size),
    ) as pool:
        return list(pool.map(_signature_task, texts, chunksize=chunksize))


def build_lsh_buckets(
    signatures: list[np.ndarray], *, bands: int, rows: int
) -> list[dict[tuple, list[int]]]:
    """Index signatures into ``bands`` LSH buckets, keyed by band-slice."""
    buckets: list[dict[tuple, list[int]]] = [defaultdict(list) for _ in range(bands)]
    for index, signature in enumerate(signatures):
        for band_index in range(bands):
            key = _band_key(signature, band_index * rows, (band_index + 1) * rows)
            buckets[band_index][key].append(index)
    return buckets


def query_matches(
    query_signatures: list[np.ndarray],
    index_signatures: list[np.ndarray],
    buckets: list[dict[tuple, list[int]]],
    *,
    bands: int,
    rows: int,
    threshold: float,
) -> list[int | None]:
    """For each query signature, the index of a >=threshold match, or ``None``.

    Pure lookup against an already-built index -- cheap compared to hashing,
    so this stays single-threaded.
    """
    matches: list[int | None] = []
    for signature in query_signatures:
        candidates: set[int] = set()
        for band_index in range(bands):
            key = _band_key(signature, band_index * rows, (band_index + 1) * rows)
            candidates.update(buckets[band_index].get(key, ()))
        hit = None
        for candidate in candidates:
            if estimate_similarity(signature, index_signatures[candidate]) >= threshold:
                hit = candidate
                break
        matches.append(hit)
    return matches


def remove_near_duplicates(
    records: list[dict[str, Any]],
    *,
    num_perm: int = DEFAULT_NUM_PERM,
    bands: int = DEFAULT_BANDS,
    rows: int = DEFAULT_ROWS,
    shingle_size: int = DEFAULT_SHINGLE,
    threshold: float = 0.55,
    stats: DedupStats | None = None,
    progress_every: int = 20_000,
    on_progress=None,
    workers: int | None = None,
) -> list[dict[str, Any]]:
    """Remove near-duplicates via MinHash + LSH banding, manual-priority first.

    Documents are processed in manual-first order so that when a cluster of
    near-duplicates forms, the manual representative is the one already in the
    index and the downloaded copies are the ones rejected. Signatures for the
    whole batch are computed up front, in parallel across CPU cores -- the
    LSH insertion loop that follows only needs to run in the required order.
    """
    if bands * rows != num_perm:
        raise ValueError(f"bands*rows ({bands}*{rows}) must equal num_perm ({num_perm})")

    stats = stats if stats is not None else DedupStats()
    if not stats.input_docs:
        stats.input_docs = len(records)

    # Manual first, so manual documents claim the LSH buckets.
    ordered = sorted(records, key=lambda r: (_lane_rank(r["lane"]), r["doc_id"]))
    all_signatures = compute_signatures(
        [r["text"] for r in ordered],
        num_perm=num_perm, shingle_size=shingle_size, workers=workers,
    )

    buckets: list[dict[tuple, list[int]]] = [defaultdict(list) for _ in range(bands)]
    signatures: dict[int, np.ndarray] = {}
    kept: list[dict[str, Any]] = []

    for position, (record, signature) in enumerate(zip(ordered, all_signatures)):
        candidates: set[int] = set()
        band_keys = []
        for band_index in range(bands):
            key = _band_key(signature, band_index * rows, (band_index + 1) * rows)
            band_keys.append(key)
            candidates.update(buckets[band_index].get(key, ()))

        duplicate_of = None
        for candidate in candidates:
            if estimate_similarity(signature, signatures[candidate]) >= threshold:
                duplicate_of = candidate
                break

        if duplicate_of is not None:
            stats.near_removed += 1
            stats.removed_by_lane[record["lane"]] += 1
            incumbent = kept[duplicate_of] if duplicate_of < len(kept) else None
            if incumbent is not None and incumbent["lane"] != record["lane"]:
                stats.cross_lane_pairs += 1
                if incumbent["lane"] == schema.LANE_MANUAL:
                    stats.manual_saved_by_priority += 1
        else:
            index = len(kept)
            kept.append(record)
            signatures[index] = signature
            for band_index, key in enumerate(band_keys):
                buckets[band_index][key].append(index)

        if on_progress and position and position % progress_every == 0:
            on_progress(position, len(kept))

    stats.kept = len(kept)
    return kept


def find_contaminated(
    train: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    *,
    num_perm: int = DEFAULT_NUM_PERM,
    bands: int = DEFAULT_BANDS,
    rows: int = DEFAULT_ROWS,
    shingle_size: int = DEFAULT_SHINGLE,
    threshold: float = 0.55,
    workers: int | None = None,
) -> set[str]:
    """Return ``doc_id`` values in ``train`` that near-match a held-out document.

    Run after the splits are fixed. A paraphrased test document leaks nearly as
    much as an exact copy, so exact-hash comparison alone is insufficient.
    Thin convenience wrapper around :func:`compute_signatures` /
    :func:`build_lsh_buckets` / :func:`query_matches` -- callers that need to
    query the same holdout index twice (e.g. a post-removal rescan) should use
    those directly instead, so the holdout side is hashed only once. See
    ``s09_decontaminate.py``.
    """
    holdout_signatures = compute_signatures(
        [r["text"] for r in holdout], num_perm=num_perm, shingle_size=shingle_size, workers=workers,
    )
    buckets = build_lsh_buckets(holdout_signatures, bands=bands, rows=rows)
    train_signatures = compute_signatures(
        [r["text"] for r in train], num_perm=num_perm, shingle_size=shingle_size, workers=workers,
    )
    hits = query_matches(
        train_signatures, holdout_signatures, buckets, bands=bands, rows=rows, threshold=threshold,
    )
    return {train[i]["doc_id"] for i, hit in enumerate(hits) if hit is not None}

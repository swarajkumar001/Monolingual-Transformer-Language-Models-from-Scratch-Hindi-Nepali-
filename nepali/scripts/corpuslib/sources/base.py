"""
Source registry and the collector contract.

The single most important invariant in Phase 1 lives here: **a source belongs
to exactly one lane**. The previous run violated it by collecting Hindi
Wikipedia twice -- once as an HF dump tagged ``downloaded`` and once through
the MediaWiki API tagged ``manual`` -- producing two "independent" sources that
were 97.7% the same articles. The manual share was therefore fictional: any
deduplication pass would delete one side of every pair.

:class:`SourceSpec` makes that structurally impossible. Lane assignment is
declared once, in ``configs/sources.yaml``, and the shard writer refuses any
document whose lane disagrees with its source's registration.

Every collector is a :class:`Collector` subclass that:

* runs under a **word budget** and a **wall-clock budget**, so no single source
  can consume the overnight run;
* checkpoints its position to ``_state.json`` after every batch, so an
  interrupted run resumes instead of restarting;
* never raises out of :meth:`Collector.run` -- a failing source degrades to a
  logged ``BLOCKED`` result while its siblings continue.
"""

from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from .. import schema
from ..jsonlio import ShardWriter, atomic_write_json, read_json
from ..nethttp import PoliteSession

# Result statuses for a single source's collection attempt.
OK = "OK"
PARTIAL = "PARTIAL"        # ran, produced data, but did not reach its target
SKIPPED = "SKIPPED"        # deliberately not run (disabled, or unmet dependency)
FAILED = "FAILED"          # errored out; no usable data


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Immutable declaration of one corpus source."""

    source_id: str
    source_name: str
    lane: str
    collection_method: str
    enabled: bool = True
    target_words: int = 0
    max_seconds: int = 3600
    priority: int = 50          # lower runs earlier when capacity is limited
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.lane not in schema.VALID_LANES:
            raise ValueError(f"{self.source_id}: invalid lane {self.lane!r}")
        if self.collection_method not in schema.VALID_METHODS:
            raise ValueError(
                f"{self.source_id}: invalid collection_method {self.collection_method!r}"
            )


class SourceRegistry:
    """All declared sources for one language, with the lane invariant enforced."""

    def __init__(self, specs: list[SourceSpec]) -> None:
        self._by_id: dict[str, SourceSpec] = {}
        for spec in specs:
            if spec.source_id in self._by_id:
                raise ValueError(f"duplicate source_id {spec.source_id!r}")
            self._by_id[spec.source_id] = spec
        self._assert_lanes_disjoint()

    def _assert_lanes_disjoint(self) -> None:
        """Verify no *content family* straddles the manual/downloaded boundary.

        Sources declare a ``content_family`` in their params (e.g. ``wikipedia``,
        ``news``, ``web_crawl``). Two sources drawing the same family into
        different lanes is the exact defect that invalidated the previous run,
        so it is rejected at construction time rather than discovered after a
        night of collection.
        """
        family_lanes: dict[str, tuple[str, str]] = {}
        for spec in self._by_id.values():
            if not spec.enabled:
                continue
            family = spec.params.get("content_family")
            if not family:
                continue
            previous = family_lanes.get(family)
            if previous and previous[1] != spec.lane:
                raise ValueError(
                    f"content family {family!r} is claimed by {previous[0]!r} "
                    f"(lane={previous[1]}) and {spec.source_id!r} (lane={spec.lane}). "
                    f"A content family must belong to exactly one lane, otherwise "
                    f"the manual/downloaded ratio double-counts the same text."
                )
            family_lanes[family] = (spec.source_id, spec.lane)

    def __getitem__(self, source_id: str) -> SourceSpec:
        return self._by_id[source_id]

    def __contains__(self, source_id: str) -> bool:
        return source_id in self._by_id

    def all(self) -> list[SourceSpec]:
        return sorted(self._by_id.values(), key=lambda s: (s.priority, s.source_id))

    def enabled(self) -> list[SourceSpec]:
        return [s for s in self.all() if s.enabled]

    def by_lane(self, lane: str) -> list[SourceSpec]:
        return [s for s in self.all() if s.lane == lane]

    def lane_target_words(self, lane: str) -> int:
        return sum(s.target_words for s in self.by_lane(lane) if s.enabled)


@dataclass
class CollectionResult:
    """Outcome of one source's collection attempt."""

    source_id: str
    lane: str
    status: str
    docs: int = 0
    words: int = 0
    bytes: int = 0
    seconds: float = 0.0
    target_words: int = 0
    message: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def target_fraction(self) -> float:
        if not self.target_words:
            return 1.0
        return self.words / self.target_words

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "source_id": self.source_id,
            "lane": self.lane,
            "status": self.status,
            "docs": self.docs,
            "words_estimate": self.words,
            "bytes": self.bytes,
            "seconds": round(self.seconds, 1),
            "target_words": self.target_words,
            "target_fraction": round(self.target_fraction, 3),
            "message": self.message,
        }
        payload.update(self.detail)
        return payload


class Budget:
    """Joint word / time / document budget for a single source."""

    def __init__(
        self,
        *,
        target_words: int,
        max_seconds: float,
        max_docs: int = 0,
        headroom: float = 1.0,
    ) -> None:
        # Collect `headroom`x the target because cleaning and deduplication
        # remove material; sizing to the bare target guarantees a shortfall.
        self.target_words = int(target_words * headroom) if target_words else 0
        self.max_seconds = max_seconds
        self.max_docs = max_docs
        self.started = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed)

    def exhausted(self, *, words: int, docs: int) -> str | None:
        """Return the name of the exhausted limit, or ``None`` to continue."""
        if self.target_words and words >= self.target_words:
            return "word_target"
        if self.max_docs and docs >= self.max_docs:
            return "doc_limit"
        if self.elapsed >= self.max_seconds:
            return "time_limit"
        return None


class Collector(ABC):
    """Base class for every corpus source.

    Subclasses implement :meth:`iter_documents`, a generator of
    :class:`~corpuslib.schema.Document`. Everything else -- budgets, sharding,
    checkpointing, progress reporting, error containment -- is handled here so
    that each source file contains only the logic specific to its medium.
    """

    def __init__(
        self,
        spec: SourceSpec,
        *,
        lang: str,
        lang_code: str,
        out_root: Path,
        session: PoliteSession | None = None,
        resume: bool = False,
        headroom: float = 1.0,
        progress: Callable[[str, dict[str, Any]], None] | None = None,
        limit: int = 0,
    ) -> None:
        self.spec = spec
        self.lang = lang
        self.lang_code = lang_code
        self.session = session
        self.resume = resume
        self.headroom = headroom
        self.limit = limit
        self._progress_cb = progress

        lane_dir = "manual" if spec.lane == schema.LANE_MANUAL else "downloaded"
        self.out_dir = Path(out_root) / lane_dir / spec.source_id
        self.state_path = self.out_dir / "_state.json"

    # -- checkpointing -----------------------------------------------------

    def load_state(self) -> dict[str, Any]:
        return read_json(self.state_path, default={}) or {}

    def save_state(self, state: dict[str, Any]) -> None:
        atomic_write_json(self.state_path, state)

    # -- progress ----------------------------------------------------------

    def report(self, **fields: Any) -> None:
        if self._progress_cb is not None:
            self._progress_cb(self.spec.source_id, fields)

    # -- the contract subclasses implement ---------------------------------

    @abstractmethod
    def iter_documents(self, budget: Budget) -> Iterator[schema.Document]:
        """Yield documents until exhausted or until ``budget`` says stop.

        Implementations should check ``budget.remaining_seconds`` at natural
        boundaries (page fetch, shard read) so a source stops promptly rather
        than after one more expensive operation.
        """

    def preflight(self) -> str | None:
        """Return a reason to skip, or ``None`` to proceed.

        Used for dependency checks -- a missing Tesseract language pack, a
        wiki that does not exist for this language -- so that an unavailable
        source is reported as ``SKIPPED`` with a cause rather than ``FAILED``.
        """
        return None

    def make_document(self, text: str, **kwargs: Any) -> schema.Document:
        """Build a :class:`Document` carrying this source's provenance."""
        return schema.Document(
            text=text,
            lang=self.lang,
            lane=self.spec.lane,
            source_id=self.spec.source_id,
            source_name=self.spec.source_name,
            collection_method=self.spec.collection_method,
            **kwargs,
        )

    # -- driver ------------------------------------------------------------

    def run(self) -> CollectionResult:
        """Collect this source. Never raises; failures become FAILED results."""
        started = time.monotonic()
        result = CollectionResult(
            source_id=self.spec.source_id,
            lane=self.spec.lane,
            status=FAILED,
            target_words=self.spec.target_words,
        )

        skip_reason = None
        try:
            skip_reason = self.preflight()
        except Exception as exc:  # noqa: BLE001
            skip_reason = f"preflight error: {type(exc).__name__}: {exc}"

        if skip_reason:
            result.status = SKIPPED
            result.message = skip_reason
            result.seconds = time.monotonic() - started
            self.report(status=SKIPPED, message=skip_reason)
            return result

        budget = Budget(
            target_words=self.spec.target_words,
            max_seconds=self.spec.max_seconds,
            max_docs=self.limit,
            headroom=self.headroom,
        )

        writer = ShardWriter(
            self.out_dir,
            lang=self.lang,
            lane=self.spec.lane,
            source_id=self.spec.source_id,
            mode="resume" if self.resume else "fresh",
        )

        stop_reason = "source_exhausted"
        error_detail = ""
        duplicates = 0
        # On resume, ShardWriter._adopt_existing() has already folded prior
        # shards' docs/words into writer.stats before we get here -- if that
        # alone already meets the budget, skip iter_documents() entirely
        # rather than re-visiting a source with nothing left to contribute.
        # This matters most for dump-based collectors (wikidump), which have
        # no resume cursor and would otherwise re-download the whole dump
        # just to immediately discard it as duplicates.
        already_met = self.resume and budget.exhausted(
            words=writer.stats.words, docs=writer.stats.docs
        )
        try:
            if already_met:
                stop_reason = f"already_satisfied_on_resume ({already_met})"
            else:
                for doc in self.iter_documents(budget):
                    if not writer.write(doc):
                        duplicates += 1
                    limit_hit = budget.exhausted(
                        words=writer.stats.words, docs=writer.stats.docs
                    )
                    if writer.stats.docs % 250 == 0:
                        self.report(
                            status="RUNNING",
                            docs=writer.stats.docs,
                            words=writer.stats.words,
                            seconds=round(budget.elapsed, 1),
                        )
                    if limit_hit:
                        stop_reason = limit_hit
                        break
        except KeyboardInterrupt:
            stop_reason = "interrupted"
            raise
        except Exception as exc:  # noqa: BLE001 - containment is the point
            stop_reason = "error"
            error_detail = f"{type(exc).__name__}: {exc}"
            result.detail["traceback"] = traceback.format_exc(limit=6)
        finally:
            manifest = writer.close(
                {
                    "source_name": self.spec.source_name,
                    "collection_method": self.spec.collection_method,
                    "target_words": self.spec.target_words,
                    "stop_reason": stop_reason,
                    "within_source_duplicates_dropped": duplicates,
                    "params": self.spec.params,
                }
            )

        result.docs = manifest["docs"]
        result.words = manifest["words_estimate"]
        result.bytes = manifest["bytes"]
        result.seconds = time.monotonic() - started
        result.detail["stop_reason"] = stop_reason
        result.detail["duplicates_dropped"] = duplicates

        if error_detail and result.docs == 0:
            result.status = FAILED
            result.message = error_detail
        elif error_detail:
            result.status = PARTIAL
            result.message = f"partial: {error_detail}"
        elif self.spec.target_words and result.words < self.spec.target_words * 0.9:
            result.status = PARTIAL
            result.message = (
                f"reached {result.words:,} of {self.spec.target_words:,} target words "
                f"({stop_reason})"
            )
        else:
            result.status = OK
            result.message = f"stopped on {stop_reason}"

        self.report(
            status=result.status,
            docs=result.docs,
            words=result.words,
            seconds=round(result.seconds, 1),
            message=result.message,
        )
        return result

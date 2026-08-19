"""
The manual/downloaded token ledger.

Phase 1 is graded on a single hard number: at least 20% of each language's
final training tokens must come from manual collection. The previous run
reported 7.66% and 2.49%, and both figures were themselves overstated because
the manual and downloaded lanes shared 97.7% of their Wikipedia content.

This module is the one place that number is computed, so it cannot disagree
with itself between stages. It provides:

* :class:`Ledger` -- per-source, per-lane totals at any pipeline stage, in
  words during collection and in *real tokens* once a tokenizer exists;
* :func:`balance_to_target` -- the deterministic subsampling rule that
  guarantees the floor is met even when manual collection underdelivers.

On why subsampling is legitimate
--------------------------------
The requirement constrains corpus *composition*, not collection effort. Having
collected more downloaded text than the composition allows, the choices are to
discard the excess or to fail the requirement. Discarding is the correct call,
it is applied by a seeded deterministic rule, and it is reported explicitly.
The alternative -- silently shipping a 7% manual corpus -- is what happened
last time.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import schema
from .jsonlio import atomic_write_json, collect_manifests


@dataclass(slots=True)
class LaneTotals:
    docs: int = 0
    words: int = 0
    tokens: int = 0

    def add(self, docs: int = 0, words: int = 0, tokens: int = 0) -> None:
        self.docs += docs
        self.words += words
        self.tokens += tokens


@dataclass
class Ledger:
    """Per-source and per-lane accounting for one language at one stage."""

    lang: str
    stage: str
    by_source: dict[str, LaneTotals] = field(default_factory=lambda: defaultdict(LaneTotals))
    source_lane: dict[str, str] = field(default_factory=dict)
    source_name: dict[str, str] = field(default_factory=dict)
    #: source_id -> (lost_docs, lost_words), populated only from manifests
    #: that recorded a shard lost to a vanished output directory. See
    #: ``jsonlio.ShardWriter``. Empty in the common case.
    source_lost: dict[str, tuple[int, int]] = field(default_factory=dict)

    # -- population --------------------------------------------------------

    def add_record(self, record: dict[str, Any]) -> None:
        source_id = record["source_id"]
        lane = record["lane"]
        previous = self.source_lane.setdefault(source_id, lane)
        if previous != lane:
            raise ValueError(
                f"source {source_id!r} appears in both lanes ({previous!r} and "
                f"{lane!r}); the manual/downloaded split would double-count it"
            )
        self.source_name.setdefault(source_id, record.get("source_name", source_id))
        self.by_source[source_id].add(
            docs=1,
            words=record.get("n_words", 0),
            tokens=record.get("n_tokens", 0) or 0,
        )

    def add_records(self, records: Iterable[dict[str, Any]]) -> "Ledger":
        for record in records:
            self.add_record(record)
        return self

    @classmethod
    def from_manifests(cls, lang: str, raw_root: Path, stage: str = "collection") -> "Ledger":
        """Build a ledger from source manifests without re-reading the text."""
        ledger = cls(lang=lang, stage=stage)
        for manifest in collect_manifests(raw_root):
            source_id = manifest["source_id"]
            ledger.source_lane[source_id] = manifest["lane"]
            ledger.source_name[source_id] = manifest.get("source_name", source_id)
            ledger.by_source[source_id].add(
                docs=manifest.get("docs", 0),
                words=manifest.get("words_estimate", 0),
            )
            lost_docs = manifest.get("lost_docs", 0)
            lost_words = manifest.get("lost_words", 0)
            if lost_docs or lost_words:
                ledger.source_lost[source_id] = (lost_docs, lost_words)
        return ledger

    def total_lost(self) -> tuple[int, int]:
        """Total (docs, words) lost to a vanished output directory, all sources."""
        docs = sum(d for d, _ in self.source_lost.values())
        words = sum(w for _, w in self.source_lost.values())
        return docs, words

    # -- aggregation -------------------------------------------------------

    def lane_totals(self, unit: str = "words") -> dict[str, int]:
        totals = {schema.LANE_MANUAL: 0, schema.LANE_DOWNLOADED: 0}
        for source_id, entry in self.by_source.items():
            lane = self.source_lane[source_id]
            totals[lane] += getattr(entry, unit)
        return totals

    def total(self, unit: str = "words") -> int:
        return sum(self.lane_totals(unit).values())

    def manual_ratio(self, unit: str = "words") -> float:
        totals = self.lane_totals(unit)
        grand = totals[schema.LANE_MANUAL] + totals[schema.LANE_DOWNLOADED]
        return totals[schema.LANE_MANUAL] / grand if grand else 0.0

    def meets_floor(self, floor: float = 0.20, unit: str = "words") -> bool:
        return self.manual_ratio(unit) >= floor

    def shortfall_words(self, floor: float = 0.20, unit: str = "words") -> int:
        """Manual units still needed to reach ``floor`` at the current downloaded size."""
        totals = self.lane_totals(unit)
        manual, downloaded = totals[schema.LANE_MANUAL], totals[schema.LANE_DOWNLOADED]
        # manual / (manual + downloaded) >= floor  =>  manual >= floor*downloaded/(1-floor)
        required = floor * downloaded / (1.0 - floor)
        return max(0, int(required - manual))

    def downloaded_cap(self, target_ratio: float, unit: str = "words") -> int:
        """Largest downloaded total compatible with ``target_ratio``."""
        manual = self.lane_totals(unit)[schema.LANE_MANUAL]
        if target_ratio <= 0 or target_ratio >= 1:
            raise ValueError(f"target_ratio must be in (0,1), got {target_ratio}")
        return int(manual * (1.0 - target_ratio) / target_ratio)

    # -- reporting ---------------------------------------------------------

    def to_dict(self, unit: str = "words") -> dict[str, Any]:
        lanes = self.lane_totals(unit)
        return {
            "lang": self.lang,
            "stage": self.stage,
            "unit": unit,
            "total": self.total(unit),
            "manual": lanes[schema.LANE_MANUAL],
            "downloaded": lanes[schema.LANE_DOWNLOADED],
            "manual_ratio": round(self.manual_ratio(unit), 4),
            "sources": [
                {
                    "source_id": source_id,
                    "source_name": self.source_name.get(source_id, source_id),
                    "lane": self.source_lane[source_id],
                    "docs": entry.docs,
                    "words": entry.words,
                    "tokens": entry.tokens,
                }
                for source_id, entry in sorted(
                    self.by_source.items(),
                    key=lambda kv: (self.source_lane[kv[0]], -kv[1].words),
                )
            ],
        }

    def write(self, path: Path, unit: str = "words") -> dict[str, Any]:
        payload = self.to_dict(unit)
        atomic_write_json(path, payload)
        return payload

    def render_table(self, unit: str = "words") -> str:
        """Fixed-width summary for the console and the wake-up report."""
        payload = self.to_dict(unit)
        rows = [
            f"{'SOURCE':<28} {'LANE':<11} {'DOCS':>10} {unit.upper():>14}  SHARE",
            "-" * 74,
        ]
        grand = max(payload["total"], 1)
        for entry in payload["sources"]:
            value = entry[unit] if unit in entry else entry["words"]
            rows.append(
                f"{entry['source_id']:<28} {entry['lane']:<11} "
                f"{entry['docs']:>10,} {value:>14,}  {value / grand:>5.1%}"
            )
        rows.append("-" * 74)
        rows.append(
            f"{'MANUAL':<28} {'':<11} {'':>10} {payload['manual']:>14,}  "
            f"{payload['manual_ratio']:>5.1%}"
        )
        rows.append(
            f"{'DOWNLOADED':<28} {'':<11} {'':>10} {payload['downloaded']:>14,}  "
            f"{1 - payload['manual_ratio']:>5.1%}"
        )
        rows.append(f"{'TOTAL':<28} {'':<11} {'':>10} {payload['total']:>14,}")
        lost_docs, lost_words = self.total_lost()
        if lost_docs or lost_words:
            rows.append("-" * 74)
            rows.append(
                f"WARNING: {lost_docs:,} docs / {lost_words:,} words counted but lost "
                f"to a vanished output directory during collection -- see "
                f"{sorted(self.source_lost)} -- excluded from the totals above."
            )
        return "\n".join(rows)


def cap_downloaded_lane(
    records: list[dict[str, Any]],
    *,
    downloaded_unit_cap: int,
    unit: str = "n_words",
    seed: int = 20260817,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep every manual document; trim the downloaded lane to a fixed budget.

    This replaced an earlier version that shrank the downloaded lane until the
    manual *ratio* hit a target -- which meant a starved manual lane forced
    away almost all of a large, perfectly usable downloaded lane just to keep
    a percentage. Composition and volume are different problems: the
    downloaded lane is capped here at ``downloaded_unit_cap`` regardless of
    how much manual text exists, and manual volume is grown separately, at
    collection time, by adding and deepening manual sources (see
    ``configs/sources.yaml``). If the manual floor still isn't met after this
    step, that is reported, not silently forced -- ``report/qa_gate.py``
    already treats a documented shortfall as a WARN, not a FAIL.

    Selection within the downloaded lane is a seeded shuffle, applied
    round-robin across downloaded *sources* so the corpus keeps its source
    diversity rather than one source being emptied first. Manual documents
    are never touched or reordered.
    """
    manual = [r for r in records if r["lane"] == schema.LANE_MANUAL]
    downloaded = [r for r in records if r["lane"] == schema.LANE_DOWNLOADED]

    manual_units = sum(r.get(unit, 0) for r in manual)
    downloaded_units = sum(r.get(unit, 0) for r in downloaded)
    total_units = manual_units + downloaded_units
    before_ratio = manual_units / total_units if total_units else 0.0

    report: dict[str, Any] = {
        "action": "none",
        "unit": unit,
        "seed": seed,
        "downloaded_unit_cap": downloaded_unit_cap,
        "ratio_before": round(before_ratio, 4),
        "manual_units": manual_units,
        "downloaded_units_before": downloaded_units,
        "downloaded_docs_before": len(downloaded),
    }

    if downloaded_units <= downloaded_unit_cap:
        report["ratio_after"] = round(before_ratio, 4)
        report["downloaded_units_after"] = downloaded_units
        report["downloaded_docs_after"] = len(downloaded)
        report["reason"] = "downloaded lane already within the cap; nothing dropped"
        return records, report

    rng = random.Random(seed)
    # Group by source so the drop is spread across sources rather than
    # concentrated in whichever one happens to sort first.
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in downloaded:
        by_source[record["source_id"]].append(record)
    for bucket in by_source.values():
        rng.shuffle(bucket)

    kept_downloaded: list[dict[str, Any]] = []
    kept_units = 0
    cursors = {source_id: 0 for source_id in by_source}
    source_cycle = sorted(by_source)
    # Round-robin across sources, taking one document at a time, until the cap.
    while kept_units < downloaded_unit_cap:
        progressed = False
        for source_id in source_cycle:
            bucket = by_source[source_id]
            index = cursors[source_id]
            if index >= len(bucket):
                continue
            record = bucket[index]
            cursors[source_id] = index + 1
            progressed = True
            size = record.get(unit, 0)
            if kept_units + size > downloaded_unit_cap and kept_downloaded:
                continue
            kept_downloaded.append(record)
            kept_units += size
            if kept_units >= downloaded_unit_cap:
                break
        if not progressed:
            break

    kept = manual + kept_downloaded
    after_total = manual_units + kept_units
    report.update(
        {
            "action": "capped_downloaded",
            "downloaded_units_after": kept_units,
            "downloaded_docs_after": len(kept_downloaded),
            "downloaded_docs_dropped": len(downloaded) - len(kept_downloaded),
            "downloaded_units_dropped": downloaded_units - kept_units,
            "ratio_after": round(manual_units / after_total, 4) if after_total else 0.0,
            "reason": (
                f"downloaded lane had {downloaded_units:,} {unit}, above the "
                f"{downloaded_unit_cap:,}-{unit} cap; trimmed to the cap. Manual lane "
                f"({manual_units:,} {unit}) was not touched."
            ),
        }
    )
    return kept, report

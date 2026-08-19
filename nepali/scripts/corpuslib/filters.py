"""
Heuristic quality filters, with different postures for the two resource tiers.

The governing asymmetry: Hindi has surplus data, so a filter that discards a
borderline document costs nothing. Nepali does not, so every discarded document
is unrecoverable and the same threshold is the wrong threshold. Both postures
come from ``configs/cleaning.yaml``; the code is shared.

Every filter returns a *reason string* rather than a boolean, and the caller
tallies reasons. This turns "retention was 61%" into "retention was 61%, of
which 22 points were short documents and 14 points repeated lines" -- which is
the level of detail the dataset-statistics deliverable asks for, and the level
at which a broken filter becomes visible.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .textnorm import devanagari_ratio, symbol_ratio

KEEP = ""  # empty reason means the document passes

# Sentence terminators in Devanagari-script prose. Text with none of these
# across hundreds of characters is a list, a table dump or navigation debris.
_TERMINATORS = ("।", "?", "!", ".", "॥")

_WORD_RE = re.compile(r"\S+")
_URL_RE = re.compile(r"https?://|www\.")


@dataclass(slots=True)
class FilterConfig:
    """Thresholds for one language's posture. Loaded from cleaning.yaml."""

    min_chars: int = 200
    max_chars: int = 200_000
    min_words: int = 30
    max_symbol_ratio: float = 0.35
    min_devanagari_ratio: float = 0.60
    max_repeated_line_ratio: float = 0.30
    max_duplicate_word_ratio: float = 0.25
    min_mean_word_length: float = 1.8
    max_mean_word_length: float = 14.0
    min_terminators: int = 1
    max_url_density: float = 0.02
    min_unique_word_ratio: float = 0.18

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FilterConfig":
        known = {f for f in cls.__slots__}
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class FilterStats:
    """Tally of keep/drop decisions and their causes."""

    kept: int = 0
    dropped: int = 0
    reasons: Counter[str] = field(default_factory=Counter)
    kept_samples: list[str] = field(default_factory=list)
    dropped_samples: list[tuple[str, str]] = field(default_factory=list)

    def record(self, reason: str, text: str, *, sample_cap: int = 20) -> None:
        if reason == KEEP:
            self.kept += 1
            if len(self.kept_samples) < sample_cap:
                self.kept_samples.append(text[:400])
        else:
            self.dropped += 1
            self.reasons[reason.split(":")[0]] += 1
            if len(self.dropped_samples) < sample_cap:
                self.dropped_samples.append((reason, text[:400]))

    @property
    def total(self) -> int:
        return self.kept + self.dropped

    @property
    def retention(self) -> float:
        return self.kept / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_docs": self.total,
            "kept": self.kept,
            "dropped": self.dropped,
            "retention": round(self.retention, 4),
            "drop_reasons": dict(self.reasons.most_common()),
        }


def evaluate(text: str, config: FilterConfig) -> str:
    """Return ``KEEP`` or the reason this document should be dropped.

    Checks are ordered cheapest-first so that the common rejections cost a
    length comparison rather than a full tokenisation.
    """
    n_chars = len(text)
    if n_chars < config.min_chars:
        return f"too_short:{n_chars}"
    if n_chars > config.max_chars:
        return f"too_long:{n_chars}"

    words = _WORD_RE.findall(text)
    n_words = len(words)
    if n_words < config.min_words:
        return f"too_few_words:{n_words}"

    mean_word_length = sum(len(w) for w in words) / n_words
    if mean_word_length < config.min_mean_word_length:
        return f"words_too_short:{mean_word_length:.1f}"
    if mean_word_length > config.max_mean_word_length:
        # Long "words" indicate concatenated markup or base64 debris.
        return f"words_too_long:{mean_word_length:.1f}"

    ratio = devanagari_ratio(text)
    if ratio < config.min_devanagari_ratio:
        return f"low_devanagari:{ratio:.2f}"

    symbols = symbol_ratio(text)
    if symbols > config.max_symbol_ratio:
        return f"symbol_heavy:{symbols:.2f}"

    terminators = sum(text.count(t) for t in _TERMINATORS)
    if terminators < config.min_terminators:
        return f"no_sentence_end:{terminators}"

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        line_counts = Counter(lines)
        repeated = sum(count for count in line_counts.values() if count > 1)
        repeated_ratio = repeated / len(lines)
        if repeated_ratio > config.max_repeated_line_ratio:
            return f"repeated_lines:{repeated_ratio:.2f}"

    unique_ratio = len(set(words)) / n_words
    if unique_ratio < config.min_unique_word_ratio:
        # Highly repetitive text: SEO keyword stuffing or a stuck generator.
        return f"low_lexical_diversity:{unique_ratio:.2f}"

    url_density = len(_URL_RE.findall(text)) / n_words
    if url_density > config.max_url_density:
        return f"link_farm:{url_density:.3f}"

    return KEEP


def apply(
    records,
    config: FilterConfig,
    *,
    stats: FilterStats | None = None,
):
    """Yield records that pass ``config``, tallying decisions into ``stats``."""
    stats = stats if stats is not None else FilterStats()
    for record in records:
        reason = evaluate(record["text"], config)
        stats.record(reason, record["text"])
        if reason == KEEP:
            record.setdefault("meta", {})
            yield record

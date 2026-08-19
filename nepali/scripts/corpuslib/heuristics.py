"""
Quality-filter statistics and the S05 heuristic gate.

Every threshold this module checks against comes from ``configs/cleaning.yaml``
and is **fixed, not auto-tuned toward a retention target** -- tuning
thresholds until retention hits a chosen number optimises for volume, not
quality, and makes the reported retention figure circular (see
``executionplan.md`` S05). Hindi's thresholds are stricter than Nepali's
because Hindi has surplus data and Nepali does not: every Nepali document
this stage drops is unrecoverable.

:func:`evaluate` returns the *first* failing check, not all of them, so each
document contributes to exactly one drop-reason bucket -- which is what makes
the reviewable ``filter_samples/`` directory (20 kept + 20 dropped per reason)
meaningful rather than a document appearing under several reasons at once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .textnorm import devanagari_ratio, symbol_ratio

_WORD_RE = re.compile(r"\S+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
# Devanagari danda / double danda plus the Latin sentence enders that leak in
# from mixed-script web text. A coarse structural signal, not a real sentence
# splitter -- it exists to catch text that never forms sentences at all
# (menus, tag clouds, link lists).
_TERMINATOR_RE = re.compile(r"[।॥.!?]")


@dataclass(slots=True)
class HeuristicVerdict:
    passed: bool
    reason: str          # "accepted", or the name of the first failing check
    stats: dict[str, Any]


def repeated_line_ratio(text: str) -> float:
    """Fraction of non-blank lines that repeat another line verbatim.

    Boilerplate -- navigation menus, cookie notices, "share this article"
    blocks -- shows up as a handful of lines repeated many times, which a
    per-document average does not catch but a line-repetition ratio does.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return 0.0
    seen: dict[str, int] = {}
    for line in lines:
        seen[line] = seen.get(line, 0) + 1
    repeated = sum(count for count in seen.values() if count > 1)
    return repeated / len(lines)


def mean_word_length(text: str) -> float:
    words = _WORD_RE.findall(text)
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def count_terminators(text: str) -> int:
    return len(_TERMINATOR_RE.findall(text))


def url_density(text: str) -> float:
    """Fraction of characters belonging to URLs -- link farms read almost pure URL."""
    if not text:
        return 0.0
    url_chars = sum(len(m.group(0)) for m in _URL_RE.finditer(text))
    return url_chars / len(text)


def unique_word_ratio(text: str) -> float:
    """Distinct words / total words -- catches repeated-phrase spam and stutter."""
    words = _WORD_RE.findall(text)
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def evaluate(text: str, cfg: dict[str, Any]) -> HeuristicVerdict:
    """Run every configured heuristic check, short-circuiting on the first failure."""
    n_chars = len(text)
    words = _WORD_RE.findall(text)
    n_words = len(words)
    stats: dict[str, Any] = {"n_chars": n_chars, "n_words": n_words}

    def fail(reason: str) -> HeuristicVerdict:
        return HeuristicVerdict(False, reason, stats)

    min_chars = int(cfg.get("min_chars", 0))
    if n_chars < min_chars:
        return fail("too_short")
    max_chars = int(cfg.get("max_chars", 0) or 0)
    if max_chars and n_chars > max_chars:
        return fail("too_long")

    min_words = int(cfg.get("min_words", 0))
    if n_words < min_words:
        return fail("too_few_words")

    sym_ratio = symbol_ratio(text)
    stats["symbol_ratio"] = round(sym_ratio, 4)
    max_symbol_ratio = float(cfg.get("max_symbol_ratio", 1.0))
    if sym_ratio > max_symbol_ratio:
        return fail("too_symbolic")

    deva_ratio = devanagari_ratio(text)
    stats["devanagari_ratio"] = round(deva_ratio, 4)
    min_deva = float(cfg.get("min_devanagari_ratio", 0.0))
    if deva_ratio < min_deva:
        return fail("low_devanagari_ratio")

    rep_ratio = repeated_line_ratio(text)
    stats["repeated_line_ratio"] = round(rep_ratio, 4)
    max_repeated = float(cfg.get("max_repeated_line_ratio", 1.0))
    if rep_ratio > max_repeated:
        return fail("repeated_lines")

    mean_len = mean_word_length(text)
    stats["mean_word_length"] = round(mean_len, 3)
    min_mwl = float(cfg.get("min_mean_word_length", 0.0))
    max_mwl = float(cfg.get("max_mean_word_length", 0.0) or 0.0)
    if mean_len < min_mwl:
        return fail("mean_word_too_short")
    if max_mwl and mean_len > max_mwl:
        return fail("mean_word_too_long")

    n_terminators = count_terminators(text)
    stats["n_terminators"] = n_terminators
    min_terminators = int(cfg.get("min_terminators", 0))
    if n_terminators < min_terminators:
        return fail("too_few_sentences")

    url_ratio = url_density(text)
    stats["url_density"] = round(url_ratio, 4)
    max_url = float(cfg.get("max_url_density", 1.0))
    if url_ratio > max_url:
        return fail("link_farm")

    uniq_ratio = unique_word_ratio(text)
    stats["unique_word_ratio"] = round(uniq_ratio, 4)
    min_uniq = float(cfg.get("min_unique_word_ratio", 0.0))
    if uniq_ratio < min_uniq:
        return fail("low_lexical_diversity")

    return HeuristicVerdict(True, "accepted", stats)

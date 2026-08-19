"""
Shared tokenizer measurement -- fertility, UNK rate, character stats.

Used by both S12 (candidate selection during the vocabulary sweep) and S13
(final evaluation report), so the two never compute these numbers by two
different definitions.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

_WORD_RE = re.compile(r"\S+")


def measure(processor, texts: Iterable[str]) -> dict[str, Any]:
    """Fertility (tokens per whitespace word) and UNK rate over ``texts``.

    Fertility is measured against whitespace words, not characters -- the
    standard definition, and the one the brief's "average characters per
    token" figure complements rather than duplicates.
    """
    unk_piece = processor.id_to_piece(processor.unk_id())
    total_tokens = 0
    total_words = 0
    total_chars = 0
    unk_tokens = 0
    piece_counts: Counter[str] = Counter()

    for text in texts:
        words = _WORD_RE.findall(text)
        total_words += len(words)
        total_chars += len(text)
        pieces = processor.encode(text, out_type=str)
        total_tokens += len(pieces)
        unk_tokens += sum(1 for p in pieces if p == unk_piece)
        piece_counts.update(pieces)

    fertility = total_tokens / total_words if total_words else 0.0
    unk_rate = unk_tokens / total_tokens if total_tokens else 0.0
    chars_per_token = total_chars / total_tokens if total_tokens else 0.0

    return {
        "total_tokens": total_tokens,
        "total_words": total_words,
        "total_chars": total_chars,
        "fertility": fertility,
        "unk_tokens": unk_tokens,
        "unk_rate": unk_rate,
        "chars_per_token": chars_per_token,
        "piece_counts": piece_counts,
    }

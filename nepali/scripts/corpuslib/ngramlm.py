"""
A small order-5 word n-gram language model, for the S06 perplexity filter.

No external LM toolkit (KenLM, SRILM, ...) is a project dependency, and
pulling one in for a single filtering pass is a heavier install than the
filter is worth. Stupid backoff (Brants et al., 2007) is simple enough to
implement directly, needs no discounting/smoothing machinery, and scales to
however much reference text is available: when a 5-gram context has never
been seen, the model backs off to the (n-1)-gram estimate scaled by a fixed
penalty, all the way down to unigrams.

**The reference model must not be trained on the data it scores** -- that
would be circular, since text is always maximally probable under a model
trained on itself. S06 trains this on a held-out sample of edited prose
(Wikisource, OCR books) and scores every *other* document against it, so a
high score means "reads unlike edited prose", not "reads unlike itself".
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Iterable

_WORD_RE = re.compile(r"\S+")

BACKOFF_PENALTY = 0.4  # standard stupid-backoff constant (Brants et al., 2007)


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text)


class NGramLM:
    """Word-level n-gram model scored with stupid backoff."""

    def __init__(self, order: int = 5) -> None:
        self.order = order
        # ngram_counts[n]: {(w1..wn): count}; context_counts[n]: {(w1..w(n-1)): count}
        self.ngram_counts: list[dict[tuple[str, ...], int]] = [
            defaultdict(int) for _ in range(order + 1)
        ]
        self.vocab: set[str] = set()
        self.total_unigrams = 0
        self._fit_docs = 0

    def fit(self, token_lists: Iterable[list[str]]) -> None:
        for tokens in token_lists:
            self._fit_docs += 1
            self.vocab.update(tokens)
            self.total_unigrams += len(tokens)
            for n in range(1, self.order + 1):
                counts = self.ngram_counts[n]
                for i in range(len(tokens) - n + 1):
                    counts[tuple(tokens[i : i + n])] += 1

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _prob(self, history: tuple[str, ...], word: str) -> float:
        """Stupid-backoff probability estimate for ``word`` given ``history``."""
        max_order = min(self.order, len(history) + 1)
        for n in range(max_order, 1, -1):
            context = history[len(history) - (n - 1) :]
            context_count = self.ngram_counts[n - 1].get(context, 0)
            if context_count == 0:
                continue
            ngram_count = self.ngram_counts[n].get(context + (word,), 0)
            if ngram_count == 0:
                continue
            penalty = BACKOFF_PENALTY ** (self.order - n)
            return penalty * (ngram_count / context_count)

        # Unigram floor, add-one smoothed over the training vocabulary so an
        # unseen word gets a small nonzero probability rather than log(0).
        unigram_count = self.ngram_counts[1].get((word,), 0)
        penalty = BACKOFF_PENALTY ** (self.order - 1)
        return penalty * (unigram_count + 1) / (self.total_unigrams + self.vocab_size + 1)

    def score(self, tokens: list[str]) -> float:
        """Average bits-per-token (cross-entropy) under this model.

        Lower is better -- text that reads like the reference distribution.
        Perplexity is ``2 ** score`` if a human-readable number is wanted;
        the filter compares raw scores, so no exponentiation is needed there.
        """
        if not tokens:
            return float("inf")
        total_bits = 0.0
        history: tuple[str, ...] = ()
        for word in tokens:
            prob = self._prob(history, word)
            total_bits += -math.log2(max(prob, 1e-12))
            history = (history + (word,))[-(self.order - 1) :]
        return total_bits / len(tokens)

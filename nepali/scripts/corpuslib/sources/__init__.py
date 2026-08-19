"""Corpus source collectors, keyed by the ``collector`` field in sources.yaml."""

from __future__ import annotations

from .base import (
    FAILED,
    OK,
    PARTIAL,
    SKIPPED,
    Budget,
    CollectionResult,
    Collector,
    SourceRegistry,
    SourceSpec,
)
from .hfcorpora import IndicCorpCollector, SangrahaCollector
from .newscrawl import NewsCollector
from .ocrbooks import OcrBooksCollector
from .wikidump import WikipediaDumpCollector
from .wikimedia import MediaWikiCollector

#: Maps the ``collector`` key in ``configs/sources.yaml`` to its implementation.
#: Adding a source is a config change plus one entry here.
COLLECTORS: dict[str, type[Collector]] = {
    "sangraha": SangrahaCollector,
    "indiccorp": IndicCorpCollector,
    "mediawiki": MediaWikiCollector,
    "wikipedia_dump": WikipediaDumpCollector,
    "news": NewsCollector,
    "ocr_books": OcrBooksCollector,
}


def build_collector(spec: SourceSpec, **kwargs) -> Collector:
    """Instantiate the collector declared by ``spec``."""
    key = spec.params.get("collector")
    if key not in COLLECTORS:
        raise KeyError(
            f"{spec.source_id}: unknown collector {key!r}; "
            f"known collectors are {sorted(COLLECTORS)}"
        )
    return COLLECTORS[key](spec, **kwargs)


__all__ = [
    "COLLECTORS",
    "Budget",
    "CollectionResult",
    "Collector",
    "FAILED",
    "IndicCorpCollector",
    "MediaWikiCollector",
    "NewsCollector",
    "OK",
    "OcrBooksCollector",
    "PARTIAL",
    "SKIPPED",
    "SangrahaCollector",
    "SourceRegistry",
    "SourceSpec",
    "WikipediaDumpCollector",
    "build_collector",
]

"""
corpuslib -- the Phase 1 corpus construction library.

This package is vendored identically into ``hindi/scripts/`` and
``nepali/scripts/`` so that each language directory is self-contained, as the
assignment's repository-layout guidance requires. The two copies are kept
byte-identical by ``python hindi/scripts/run_pipeline.py sync`` (hindi is the
source of truth), and the QA gate hashes them to prove no drift has crept in.
All language-specific behaviour is expressed in ``<lang>/configs/*.yaml``,
never in code.

Module map
----------
``schema``      the document record contract; content-hash ids and lane tagging
``jsonlio``     append-proof, crash-safe shard I/O
``textnorm``    Unicode/NFC normalisation, PII redaction, script ratios
``langid``      fastText LID that self-tests and fails loudly
``filters``     heuristic quality filters with per-language postures
``dedup``       exact and MinHash/LSH deduplication, manual-priority resolution
``ledger``      the manual/downloaded token accounting and rebalancing rule
``nethttp``     polite, rate-limited, retrying HTTP
``runlog``      append-only ``log.md`` plus structured events
``config``      YAML loading
``sources``     one collector per corpus source
"""

from __future__ import annotations

__version__ = "2.0.0"

from . import (  # noqa: F401
    config,
    dedup,
    filters,
    jsonlio,
    langid,
    ledger,
    nethttp,
    runlog,
    schema,
    textnorm,
)

__all__ = [
    "config",
    "dedup",
    "filters",
    "jsonlio",
    "langid",
    "ledger",
    "nethttp",
    "runlog",
    "schema",
    "textnorm",
]

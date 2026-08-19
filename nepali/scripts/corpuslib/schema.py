"""
Canonical document record schema for the Phase 1 corpus.

Every document that enters the pipeline -- from any source, in any lane --
is represented by exactly this record shape. The schema is the contract that
makes the >=20% manual-token requirement auditable: `lane` and `source_id`
travel with the text from collection all the way to the final split files, so
the manual/downloaded ledger can be recomputed at any stage without guessing.

Design notes
------------
* ``doc_id`` is a content hash of the *normalized* text. Two documents with
  identical text therefore collide by construction, which makes exact-duplicate
  removal a set operation rather than a pipeline stage that can be skipped.
* ``lane`` is validated against the source registry at write time (see
  ``corpuslib.sources.base``). A source may belong to exactly one lane for the
  lifetime of the project, so no document can ever be counted on both sides of
  the manual/downloaded split.
* ``n_tokens`` is deliberately absent at collection time. Real token counts
  require the trained tokenizer, and estimating them from word counts is what
  produced the misleading budgets in the previous run. Word counts are recorded
  as ``n_words`` and are explicitly labelled an estimate everywhere they appear.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

# Lanes are a closed set. `manual` means this project performed the selection,
# extraction, cleaning and deduplication; `downloaded` means a third party did.
LANE_MANUAL = "manual"
LANE_DOWNLOADED = "downloaded"
VALID_LANES = frozenset({LANE_MANUAL, LANE_DOWNLOADED})

# How the bytes were obtained. Recorded separately from `lane` because the
# course clarification ties "manual" to who controls the pipeline, not to the
# transport -- an API pull we clean ourselves is manual, an HF dump is not.
VALID_METHODS = frozenset({"api", "scrape", "ocr", "hf_download", "http_download"})

_WORD_RE = re.compile(r"\S+")

# Schema version. Bump when the record shape changes so that stale interim
# files are rejected loudly instead of being silently misread.
SCHEMA_VERSION = 2


def content_hash(text: str) -> str:
    """Return the stable content identifier for ``text``.

    Whitespace is collapsed before hashing so that documents differing only in
    line wrapping or trailing spaces share an id. This is intentional: such
    pairs are exact duplicates for language-modelling purposes.
    """
    collapsed = " ".join(text.split())
    digest = hashlib.sha256(collapsed.encode("utf-8")).hexdigest()
    return digest[:24]


def count_words(text: str) -> int:
    """Whitespace-delimited token count -- an *estimate*, never a token count."""
    return len(_WORD_RE.findall(text))


@dataclass(slots=True)
class Document:
    """One corpus document with its full provenance chain."""

    text: str
    lang: str
    lane: str
    source_id: str
    source_name: str
    collection_method: str
    origin_url: str | None = None
    title: str | None = None
    collected_at: str = ""
    doc_id: str = ""
    n_chars: int = 0
    n_words: int = 0
    schema_version: int = SCHEMA_VERSION
    # Source-specific provenance: pageid/revid for wiki, domain/publish date for
    # news, book id and OCR confidence for scanned material, HF shard for dumps.
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id:
            self.doc_id = content_hash(self.text)
        if not self.collected_at:
            self.collected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not self.n_chars:
            self.n_chars = len(self.text)
        if not self.n_words:
            self.n_words = count_words(self.text)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Document":
        known = {f for f in cls.__slots__}
        return cls(**{k: v for k, v in payload.items() if k in known})


class SchemaError(ValueError):
    """Raised when a record violates the corpus contract."""


def validate(record: dict[str, Any], *, expect_lang: str | None = None) -> None:
    """Assert that ``record`` is a well-formed corpus document.

    Called by the JSONL writer on every record, so a malformed document can
    never reach disk. The cost is a few microseconds per document; the previous
    run demonstrated what unvalidated records cost instead.
    """
    for required in ("doc_id", "text", "lang", "lane", "source_id", "collection_method"):
        if required not in record:
            raise SchemaError(f"missing required field {required!r}")

    if not isinstance(record["text"], str) or not record["text"].strip():
        raise SchemaError(f"{record['doc_id']}: empty or non-string text")

    if record["lane"] not in VALID_LANES:
        raise SchemaError(f"{record['doc_id']}: unknown lane {record['lane']!r}")

    if record["collection_method"] not in VALID_METHODS:
        raise SchemaError(
            f"{record['doc_id']}: unknown collection_method {record['collection_method']!r}"
        )

    if expect_lang is not None and record["lang"] != expect_lang:
        raise SchemaError(
            f"{record['doc_id']}: language {record['lang']!r} written into the "
            f"{expect_lang!r} corpus -- cross-language contamination"
        )

    if record.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(
            f"{record['doc_id']}: schema_version {record.get('schema_version')!r} "
            f"!= {SCHEMA_VERSION}; regenerate this file"
        )

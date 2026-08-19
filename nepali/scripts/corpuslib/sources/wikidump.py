"""
Wikipedia collection via the static XML dump, not the live Action API.

Why this exists
----------------
The live-API path in :mod:`wikimedia` (``MediaWikiCollector``) is correct and
is exactly what the course clarification means by manual collection -- but
for Wikipedia specifically it is too slow to finish. Measured on 2026-08-17:
hi.wikipedia.org paged at roughly 262 articles/minute against the Action API
(20 pages/request, one request per host-throttle interval), which is 10+
hours to cover hi.wikipedia's ~163k articles. A 6-hour budget landed at 15%
of the wiki -- 13.2M of a 90M-word target.

``dumps.wikimedia.org`` publishes the same content -- every current-revision
article, as raw wikitext -- as a single compressed XML file, refreshed
regularly. Downloading and parsing it ourselves is still manual collection
under the same clarification that covers the API path: nothing about the
text arrives pre-extracted or pre-filtered. We stream the compressed dump
directly from the HTTP response (never buffering the whole file), decompress
it incrementally, and run every page through the *same* wikitext stripper,
translation detector and quality gate as the API collector -- extraction,
cleaning and filtering are still entirely our own code, just applied to text
fetched by the much cheaper route.

This module intentionally handles Wikipedia only. Wikisource's prose lives in
the ProofreadPage ``Page:`` namespace, which the standard ``pages-articles``
dump does not include (it covers only the namespaces treated as "content"),
and Wikibooks is small enough that the live API already reaches its target
inside its budget (1,189 docs / 3.2 minutes, observed). Converting those too
would add a second dump variant for a lane that is not the bottleneck.
"""

from __future__ import annotations

import bz2
import http.client
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterator

from .. import schema
from ..nethttp import DEFAULT_USER_AGENT
from ..textnorm import devanagari_ratio, normalize
from .base import Budget, Collector
from .wikimedia import looks_translated, strip_wikitext

DUMP_HOST = "dumps.wikimedia.org"

#: A dump is one long-lived streaming download (order of minutes to low hours
#: on a slow connection). Observed failure: the connection to dumps.wikimedia
#: org was cut off after 5.6 minutes / 3,814 documents with an EOFError from
#: the bz2 layer -- a truncated response, not a bad file. Retrying restarts
#: the download from byte 0 (there is no cheap resume point inside a
#: streaming bz2-over-HTTP read), but that is not wasted effort: documents
#: already yielded before the drop are already durably written, and the
#: ShardWriter's own doc_id dedup silently skips them on the way past, so a
#: retry only pays for bytes, not for re-processing.
MAX_DOWNLOAD_ATTEMPTS = 4

# Exceptions that mean "the connection to the dump host misbehaved," as
# opposed to a real data/parsing bug -- these are what a retry can fix.
_RETRYABLE_ERRORS = (
    EOFError,                    # bz2 stream truncated mid-read
    http.client.HTTPException,   # IncompleteRead and siblings
    urllib.error.URLError,
    TimeoutError,
    ConnectionError,
    OSError,
)

# MediaWiki project name -> dump `dbname` suffix. Wikipedia is the one
# irregular case (dbname suffix "wiki", not "wikipedia"); everything else
# is the project name unchanged. See https://meta.wikimedia.org/wiki/Data_dumps.
_PROJECT_DBNAME_SUFFIX = {
    "wikipedia": "wiki",
    "wikisource": "wikisource",
    "wikibooks": "wikibooks",
    "wiktionary": "wiktionary",
    "wikiquote": "wikiquote",
    "wikinews": "wikinews",
    "wikiversity": "wikiversity",
}

# The dump ships raw wikitext only -- no parsed category list the way the
# Action API's prop=categories does -- so translation detection reads
# category links straight out of the markup instead.
_CATEGORY_LINK_RE = re.compile(
    r"\[\[\s*(?:Category|श्रेणी|वर्ग)\s*:\s*([^\]\|]+)", re.IGNORECASE
)


def dbname_for_host(host: str) -> str:
    """``hi.wikipedia.org`` -> ``hiwiki``; ``ne.wikipedia.org`` -> ``newiki``."""
    parts = host.split(".")
    if len(parts) < 2:
        raise ValueError(f"not a MediaWiki project host: {host!r}")
    lang_prefix, project = parts[0], parts[1]
    suffix = _PROJECT_DBNAME_SUFFIX.get(project, project)
    return f"{lang_prefix}{suffix}"


def dump_url_for_host(host: str) -> str:
    dbname = dbname_for_host(host)
    return f"https://{DUMP_HOST}/{dbname}/latest/{dbname}-latest-pages-articles.xml.bz2"


def _extract_categories(raw_wikitext: str) -> list[str]:
    return [m.group(1).strip() for m in _CATEGORY_LINK_RE.finditer(raw_wikitext)]


class WikipediaDumpCollector(Collector):
    """Stream a Wikipedia XML dump and extract clean prose from it.

    One HTTP request, decompressed and XML-parsed as bytes arrive -- the
    dump is never written to disk or held in memory whole. ``budget`` is
    checked once per page, so the source stops promptly at its time or word
    ceiling even mid-download; whatever was already yielded stays collected.
    """

    def preflight(self) -> str | None:
        host = self.spec.params.get("host")
        if not host:
            return "no host configured"
        try:
            self._dump_url = dump_url_for_host(host)
        except ValueError as exc:
            return str(exc)
        if self.session is not None and not self.session.allowed(self._dump_url):
            return f"{self._dump_url} disallowed by robots.txt"
        return None

    def iter_documents(self, budget: Budget) -> Iterator[schema.Document]:
        """Retry the whole download up to :data:`MAX_DOWNLOAD_ATTEMPTS` times.

        A dropped connection restarts the dump from byte 0 -- there is no
        cheap mid-stream resume point -- but that only re-spends bandwidth,
        not correctness: documents already yielded on a prior attempt were
        already written, and the writer's own doc_id dedup drops the repeats
        silently on the way past.
        """
        last_error: Exception | None = None
        for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
            if budget.remaining_seconds <= 0:
                break
            try:
                yield from self._one_pass(budget)
                return  # the dump was consumed to completion (or budget ran out cleanly)
            except _RETRYABLE_ERRORS as exc:
                last_error = exc
                self.save_state(
                    {
                        "attempt": attempt,
                        "last_error": f"{type(exc).__name__}: {exc}",
                        "note": (
                            "connection to the dump host dropped; retrying the "
                            "download from the beginning -- documents already "
                            "collected are deduplicated by the writer, not lost "
                            "or double-counted"
                        ),
                    }
                )
                continue

        if last_error is not None:
            raise RuntimeError(
                f"{self.spec.source_id}: dump download failed {MAX_DOWNLOAD_ATTEMPTS} "
                f"times against {self._dump_url}; last error: "
                f"{type(last_error).__name__}: {last_error}"
            ) from last_error

    def _one_pass(self, budget: Budget) -> Iterator[schema.Document]:
        """One attempt at downloading and parsing the dump end to end."""
        host = self.spec.params["host"]
        min_chars = int(self.spec.params.get("min_chars", 400))
        min_deva = float(self.spec.params.get("min_devanagari_ratio", 0.55))
        user_agent = self.session.user_agent if self.session else DEFAULT_USER_AGENT

        request = urllib.request.Request(self._dump_url, headers={"User-Agent": user_agent})

        n_pages = 0
        n_redirect = 0
        n_translated = 0
        n_short = 0

        # `timeout` bounds each individual socket read/connect, not the
        # whole download -- a multi-hour transfer of many small reads is
        # fine as long as no single read stalls for two minutes.
        with urllib.request.urlopen(request, timeout=120) as response:
            with bz2.BZ2File(response) as decompressed:
                context = iter(ET.iterparse(decompressed, events=("start", "end")))
                _, root = next(context)  # the <mediawiki> root, from the first start event
                for event, elem in context:
                    if event != "end" or not elem.tag.endswith("}page"):
                        continue

                    if budget.remaining_seconds <= 0:
                        elem.clear()
                        root.clear()
                        break

                    n_pages += 1
                    document = self._extract_page(
                        elem, host=host, min_chars=min_chars, min_deva=min_deva
                    )
                    # Free this page's subtree (and every completed sibling)
                    # before yielding -- the dump has ~100k+ pages and the
                    # tree would otherwise grow without bound.
                    elem.clear()
                    root.clear()

                    if document is None:
                        continue
                    if document is _REDIRECT:
                        n_redirect += 1
                        continue
                    if document is _TRANSLATED:
                        n_translated += 1
                        continue
                    if document is _SHORT:
                        n_short += 1
                        continue

                    yield document

                    if n_pages % 2000 == 0:
                        self.save_state(
                            {
                                "pages_seen": n_pages,
                                "redirects_skipped": n_redirect,
                                "translated_skipped": n_translated,
                                "short_skipped": n_short,
                                "note": (
                                    "dump-based collection has no resume cursor; "
                                    "a re-run restarts from the beginning of the dump"
                                ),
                            }
                        )

        self.save_state(
            {
                "pages_seen": n_pages,
                "redirects_skipped": n_redirect,
                "translated_skipped": n_translated,
                "short_skipped": n_short,
                "completed": True,
            }
        )

    def _extract_page(
        self, elem: ET.Element, *, host: str, min_chars: int, min_deva: float
    ) -> schema.Document | None:
        """Return a Document, one of the sentinel skip markers, or None (unparseable)."""
        if elem.find("{*}redirect") is not None:
            return _REDIRECT
        ns_text = elem.findtext("{*}ns")
        if ns_text != "0":
            return None

        revision = elem.find("{*}revision")
        if revision is None:
            return None
        text_elem = revision.find("{*}text")
        raw = (text_elem.text or "") if text_elem is not None else ""
        if not raw:
            return None

        categories = _extract_categories(raw)
        if looks_translated(raw, categories):
            return _TRANSLATED

        title = elem.findtext("{*}title") or ""
        text = normalize(strip_wikitext(raw))
        if len(text) < min_chars or devanagari_ratio(text) < min_deva:
            return _SHORT

        pageid = elem.findtext("{*}id")
        revid = revision.findtext("{*}id")
        return self.make_document(
            text,
            title=title,
            origin_url=f"https://{host}/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
            meta={
                "host": host,
                "pageid": pageid,
                "revid": revid,
                "namespace": 0,
                "n_categories": len(categories),
                "collected_via": "xml_dump",
            },
        )


# Sentinels distinguishing "skipped for a known reason" from "not parseable
# at all" (None) without allocating an exception per page.
class _Skip:
    __slots__ = ()


_REDIRECT = _Skip()
_TRANSLATED = _Skip()
_SHORT = _Skip()

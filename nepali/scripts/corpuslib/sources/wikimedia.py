"""
MediaWiki Action API collectors -- the backbone of the manual lane.

Why this counts as manual collection
------------------------------------
The course clarification is explicit: manual means *you* perform selection,
extraction, cleaning and deduplication, as opposed to consuming a corpus
someone else already prepared. This collector queries the Action API for raw
wikitext, then applies its own markup parser, template stripper, section
filter and quality gate. Nothing about the text arrives pre-extracted.

The corollary, enforced by the source registry, is that the HuggingFace
``wikimedia/wikipedia`` dump is **not** collected at all. Both would be the same
articles; counting one as manual and the other as downloaded is the exact
double-count that invalidated the previous run's 20% claim.

Rate limiting
-------------
Probing returned HTTP 429 from ``*.wikipedia.org`` for a generic user agent.
Wikimedia's User-Agent policy requires a descriptive agent with contact
details, which :class:`~corpuslib.nethttp.PoliteSession` supplies, and the API
is queried at a sustainable rate with 20 pages per request (the anonymous
limit) rather than one page per request.
"""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Iterator
from typing import Any

from .. import schema
from ..textnorm import devanagari_ratio, normalize
from .base import Budget, Collector

# --- wikitext cleaning ------------------------------------------------------
# Order matters: nested structures are removed innermost-first, and templates
# must go before link syntax because templates frequently contain links.

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_REF_RE = re.compile(r"<ref[^>]*?/>|<ref[^>]*?>.*?</ref>", re.DOTALL | re.IGNORECASE)

# Wikisource ``Page:`` entries wrap running heads, folios and footers in
# <noinclude>; that is page furniture, not prose, so the whole block goes.
_NOINCLUDE_RE = re.compile(r"<noinclude>.*?</noinclude>", re.DOTALL | re.IGNORECASE)
# Main-namespace Wikisource pages are transclusion stubs: the text lives in the
# Page: namespace and <pages .../> pulls it in at render time. Left in place it
# would contribute nothing but markup.
_PAGES_TAG_RE = re.compile(r"<pages\b[^>]*/?>", re.DOTALL | re.IGNORECASE)
_SECTION_TAG_RE = re.compile(r"<section\b[^>]*/?>", re.IGNORECASE)
_TAG_BLOCK_RE = re.compile(
    r"<(gallery|table|math|score|timeline|imagemap|syntaxhighlight|source|pre|nowiki)"
    r"[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_HTML_TAG_RE = re.compile(r"<[^>]{1,200}>")
_TABLE_RE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
_HEADING_RE = re.compile(r"^\s*={2,6}\s*(.*?)\s*={2,6}\s*$", re.MULTILINE)
_LIST_RE = re.compile(r"^[\*#:;]+\s*", re.MULTILINE)
_BOLD_ITALIC_RE = re.compile(r"'{2,5}")
_HR_RE = re.compile(r"^-{4,}$", re.MULTILINE)

# File/image/category links carry no prose; the label inside them is caption
# text that reads as sentence fragments, so the whole construct is dropped.
_MEDIA_LINK_RE = re.compile(
    r"\[\[\s*(?:File|Image|Category|चित्र|श्रेणी|फाइल|तस्वीर|फाइल|सञ्चिका|"
    r"वर्ग|चित्र)\s*:[^\[\]]*(?:\[\[[^\[\]]*\]\][^\[\]]*)*\]\]",
    re.IGNORECASE,
)
# Interlanguage links: [[en:Foo]] -- always noise in a monolingual corpus.
_INTERWIKI_RE = re.compile(r"\[\[[a-z\-]{2,12}\s*:[^\]]*\]\]")
_INTERNAL_LINK_RE = re.compile(r"\[\[([^\]\|]+)\|([^\]]+)\]\]")
_SIMPLE_LINK_RE = re.compile(r"\[\[([^\]\|]+)\]\]")
_EXTERNAL_LINK_RE = re.compile(r"\[(?:https?|ftp)://[^\s\]]+\s*([^\]]*)\]")

# Sections that are reference apparatus rather than prose. Dropping them
# materially raises corpus quality: they are lists of citations and links.
_DROP_SECTIONS = {
    # Hindi
    "सन्दर्भ", "संदर्भ", "इन्हें भी देखें", "इसे भी देखें", "बाहरी कड़ियाँ",
    "बाहरी कडियाँ", "अग्रिम पठन", "टिप्पणियाँ", "ग्रन्थसूची", "सन्दर्भ ग्रन्थ",
    # Nepali
    "सन्दर्भ सामग्रीहरू", "सन्दर्भ सामग्री", "यो पनि हेर्नुहोस्", "बाह्य कडीहरू",
    "बाह्य लिङ्कहरू", "थप हेर्नुहोस्", "सन्दर्भहरू",
    # English headings that leak into Indic wikis
    "references", "external links", "see also", "further reading",
    "bibliography", "notes", "sources",
}

# Wikitext markers indicating the page was produced by the Content Translation
# tool or is otherwise a translation rather than natively written text.
_TRANSLATION_MARKERS = (
    "{{translated page", "{{translation", "{{machine translation",
    "{{अनूदित", "{{अनुवाद", "{{रूपान्तरण",
)

# ProofreadPage stamps every Wikisource `Page:` with a human-review level:
#   0 without text · 1 not proofread · 2 problematic · 3 proofread · 4 validated
# Levels 0-2 are raw uncorrected OCR and read as garbled Devanagari. Keeping
# only 3+ is the difference between a book corpus and an OCR-noise corpus, and
# it costs nothing to check because the marker travels in the same wikitext.
_PAGEQUALITY_RE = re.compile(r'<pagequality\s+level="(\d)"', re.IGNORECASE)


def page_quality(raw_wikitext: str) -> int | None:
    """Return the ProofreadPage review level, or ``None`` if unstamped."""
    match = _PAGEQUALITY_RE.search(raw_wikitext)
    return int(match.group(1)) if match else None


def strip_wikitext(raw: str) -> str:
    """Convert raw wikitext into plain prose.

    This is the extraction step that makes API collection *manual*: the HF
    Wikipedia dump ships the output of somebody else's version of this
    function.
    """
    text = _COMMENT_RE.sub("", raw)
    text = _NOINCLUDE_RE.sub("", text)
    text = _PAGES_TAG_RE.sub("", text)
    text = _SECTION_TAG_RE.sub("", text)
    text = _REF_RE.sub("", text)
    text = _TAG_BLOCK_RE.sub("", text)
    text = _TABLE_RE.sub("", text)
    text = _remove_templates(text)
    text = _MEDIA_LINK_RE.sub("", text)
    text = _INTERWIKI_RE.sub("", text)
    text = _INTERNAL_LINK_RE.sub(r"\2", text)
    text = _SIMPLE_LINK_RE.sub(r"\1", text)
    text = _EXTERNAL_LINK_RE.sub(r"\1", text)
    text = _drop_reference_sections(text)
    text = _HEADING_RE.sub(r"\n\1\n", text)
    text = _LIST_RE.sub("", text)
    text = _BOLD_ITALIC_RE.sub("", text)
    text = _HR_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    return text


def _remove_templates(text: str, max_depth: int = 12) -> str:
    """Remove ``{{...}}`` constructs, including nested ones.

    A regex cannot match balanced braces, so this scans depth-wise: repeatedly
    delete the innermost templates until none remain or the depth cap is hit.
    """
    pattern = re.compile(r"\{\{[^{}]*\}\}", re.DOTALL)
    for _ in range(max_depth):
        text, n = pattern.subn("", text)
        if n == 0:
            break
    # Any unbalanced remnants are noise.
    return text.replace("{{", "").replace("}}", "")


def _drop_reference_sections(text: str) -> str:
    """Remove citation/link sections, keeping only prose sections."""
    parts = re.split(r"^\s*(={2,6})\s*(.*?)\s*\1\s*$", text, flags=re.MULTILINE)
    if len(parts) == 1:
        return text
    out = [parts[0]]
    # re.split with 2 groups yields [pre, level, title, body, level, title, body...]
    for i in range(1, len(parts), 3):
        title = (parts[i + 1] or "").strip()
        body = parts[i + 2] if i + 2 < len(parts) else ""
        if title.lower() in _DROP_SECTIONS or title in _DROP_SECTIONS:
            continue
        out.append(f"\n{title}\n{body}")
    return "".join(out)


def looks_translated(raw_wikitext: str, categories: list[str]) -> bool:
    """Heuristic detector for machine-translated or imported articles.

    Translated articles are translationese by construction and degrade a
    monolingual corpus, so they are excluded at collection time rather than
    filtered later.
    """
    lowered = raw_wikitext[:4000].lower()
    if any(marker in lowered for marker in _TRANSLATION_MARKERS):
        return True
    for category in categories:
        low = category.lower()
        if "translat" in low or "अनुवाद" in category or "अनूदित" in category:
            return True
    return False


class MediaWikiCollector(Collector):
    """Walk a MediaWiki project's article namespace and extract clean prose.

    Works unchanged against Wikipedia, Wikisource, Wikibooks and Wikiquote --
    they share the Action API. The specific project comes from
    ``spec.params['host']``.
    """

    #: Pages requested per API call. 20 is the anonymous ceiling for
    #: ``prop=revisions`` with content; asking for more silently truncates.
    BATCH = 20

    #: Namespace to walk. Set by :meth:`preflight`, which resolves the
    #: ``auto_page`` sentinel against the wiki's own namespace table.
    _resolved_namespace: int = 0

    def preflight(self) -> str | None:
        host = self.spec.params.get("host")
        if not host:
            return "no host configured"
        try:
            payload = self._api(host, {"action": "query", "meta": "siteinfo",
                                       "siprop": "statistics|namespaces"})
        except Exception as exc:  # noqa: BLE001
            # A wiki that does not exist for this language 302-redirects to the
            # multilingual incubator and returns HTML, so JSON decoding fails.
            # Report that as a missing wiki rather than a transport error --
            # ne.wikisource.org is exactly this case.
            return (
                f"{host} did not return a MediaWiki API response "
                f"({type(exc).__name__}); the wiki may not exist for this language"
            )

        query = payload.get("query", {})
        articles = query.get("statistics", {}).get("articles", 0)
        min_articles = int(self.spec.params.get("min_articles", 50))

        # ProofreadPage's `Page:` namespace id differs per wiki -- it is 250 on
        # hi.wikisource, not the 104 many wikis use -- so resolve it by
        # canonical name rather than hardcoding a number.
        configured = self.spec.params.get("namespace", 0)
        if str(configured) == "auto_page":
            page_ns = [
                entry["id"]
                for entry in query.get("namespaces", {}).values()
                if entry.get("canonical") == "Page"
            ]
            if not page_ns:
                return f"{host} has no ProofreadPage 'Page' namespace"
            self._resolved_namespace = page_ns[0]
            # Article count is a main-namespace figure and understates a
            # Wikisource whose content lives in Page:; use total pages instead.
            articles = query.get("statistics", {}).get("pages", articles)
        else:
            self._resolved_namespace = int(configured)

        if articles < min_articles:
            return f"{host} has only {articles:,} pages (min {min_articles:,})"
        self._site_articles = articles
        return None

    def _api(self, host: str, params: dict[str, Any]) -> dict[str, Any]:
        params = {**params, "format": "json", "formatversion": "2"}
        url = f"https://{host}/w/api.php?" + urllib.parse.urlencode(params)
        return self.session.get_json(url, check_robots=False)

    def _iter_titles(self, host: str, namespace: int, start_from: str) -> Iterator[list[str]]:
        """Page through ``list=allpages`` yielding batches of article titles."""
        cursor = start_from
        while True:
            params = {
                "action": "query",
                "list": "allpages",
                "apnamespace": str(namespace),
                "aplimit": "500",
                "apfilterredir": "nonredirects",
            }
            if cursor:
                params["apcontinue"] = cursor
            payload = self._api(host, params)
            pages = payload.get("query", {}).get("allpages", [])
            if not pages:
                return
            titles = [p["title"] for p in pages]
            for i in range(0, len(titles), self.BATCH):
                yield titles[i : i + self.BATCH]
            cursor = payload.get("continue", {}).get("apcontinue")
            if not cursor:
                return
            self._cursor = cursor

    def _fetch_batch(self, host: str, titles: list[str]) -> list[dict[str, Any]]:
        """Fetch wikitext plus categories for up to :attr:`BATCH` titles."""
        payload = self._api(
            host,
            {
                "action": "query",
                "prop": "revisions|categories",
                "rvprop": "content|ids",
                "rvslots": "main",
                "cllimit": "max",
                "titles": "|".join(titles),
            },
        )
        return payload.get("query", {}).get("pages", [])

    @staticmethod
    def _book_key(title: str) -> str:
        """Group key for Wikisource ``Page:`` entries.

        ``पृष्ठ:Some Book.pdf/१७`` -> ``पृष्ठ:Some Book.pdf``. Individual scanned
        pages run 400-1800 characters and break mid-sentence, so emitting them
        as separate documents would give the tokenizer fragments and would make
        document-level deduplication meaningless. Consecutive pages of one work
        are therefore assembled into a single document.
        """
        return title.rsplit("/", 1)[0] if "/" in title else title

    def iter_documents(self, budget: Budget) -> Iterator[schema.Document]:
        host = self.spec.params["host"]
        # Resolved in preflight(); `auto_page` wikis only learn their Page
        # namespace id from siteinfo, so this must not be re-parsed here.
        namespace = self._resolved_namespace
        min_chars = int(self.spec.params.get("min_chars", 400))
        min_deva = float(self.spec.params.get("min_devanagari_ratio", 0.55))
        group_pages = bool(self.spec.params.get("group_pages", False))
        min_page_quality = int(self.spec.params.get("min_page_quality", 0))

        state = self.load_state() if self.resume else {}
        self._cursor = state.get("cursor", "")
        n_translated = 0
        n_short = 0
        n_unproofread = 0
        n_batches = 0

        # Accumulator for page-grouping mode.
        book_key: str | None = None
        book_parts: list[str] = []
        book_pages = 0

        def flush_book() -> schema.Document | None:
            nonlocal book_key, book_parts, book_pages
            if not book_key or not book_parts:
                book_key, book_parts, book_pages = None, [], 0
                return None
            text = normalize("\n".join(book_parts))
            key, pages_used = book_key, book_pages
            book_key, book_parts, book_pages = None, [], 0
            if len(text) < min_chars or devanagari_ratio(text) < min_deva:
                return None
            return self.make_document(
                text,
                title=key,
                origin_url=f"https://{host}/wiki/"
                + urllib.parse.quote(key.replace(" ", "_")),
                meta={
                    "host": host,
                    "namespace": namespace,
                    "pages_joined": pages_used,
                    "assembled_from": "proofread_page_namespace",
                },
            )

        for titles in self._iter_titles(host, namespace, self._cursor):
            if budget.remaining_seconds <= 0:
                break
            try:
                pages = self._fetch_batch(host, titles)
            except Exception:  # noqa: BLE001 - one bad batch must not end the walk
                continue

            for page in pages:
                if page.get("missing") or "revisions" not in page:
                    continue
                revisions = page["revisions"]
                if not revisions:
                    continue
                slot = revisions[0].get("slots", {}).get("main", {})
                raw = slot.get("content") or ""
                if not raw:
                    continue

                categories = [c.get("title", "") for c in page.get("categories", [])]
                if looks_translated(raw, categories):
                    n_translated += 1
                    continue

                # Reject un-proofread scans before they reach the corpus.
                if min_page_quality:
                    level = page_quality(raw)
                    if level is not None and level < min_page_quality:
                        n_unproofread += 1
                        continue

                text = normalize(strip_wikitext(raw))
                title = page.get("title", "")

                if group_pages:
                    key = self._book_key(title)
                    if book_key is not None and key != book_key:
                        document = flush_book()
                        if document is not None:
                            yield document
                        else:
                            n_short += 1
                    book_key = key
                    if text:
                        book_parts.append(text)
                        book_pages += 1
                    continue

                if len(text) < min_chars or devanagari_ratio(text) < min_deva:
                    n_short += 1
                    continue

                yield self.make_document(
                    text,
                    title=title,
                    origin_url=f"https://{host}/wiki/"
                    + urllib.parse.quote(title.replace(" ", "_")),
                    meta={
                        "host": host,
                        "pageid": page.get("pageid"),
                        "revid": revisions[0].get("revid"),
                        "namespace": namespace,
                        "n_categories": len(categories),
                    },
                )

            n_batches += 1
            if n_batches % 25 == 0:
                self.save_state(
                    {
                        "cursor": getattr(self, "_cursor", ""),
                        "translated_skipped": n_translated,
                        "short_skipped": n_short,
                    }
                )

        if group_pages:
            document = flush_book()
            if document is not None:
                yield document

        self.save_state(
            {
                "cursor": getattr(self, "_cursor", ""),
                "translated_skipped": n_translated,
                "short_skipped": n_short,
                "unproofread_skipped": n_unproofread,
                "completed": True,
            }
        )

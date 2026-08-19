"""
News collection: URL discovery plus self-written article extraction.

This is the source the previous run failed hardest at -- it produced 122 Hindi
and 128 Nepali articles in total, because it hit a single stale sitemap
containing ~100 URLs from 2014 and had no fallback. Probing the target domains
showed why a single strategy cannot work:

===========================  ==========================================
``www.bbc.com``              35 sitemaps advertised in robots.txt
``www.amarujala.com``        sitemap index -> thousands of URLs
``www.jagran.com``           news + all-news sitemaps
``ekantipur.com``            robots.txt lists no sitemap at all
``www.onlinekhabar.com``     no sitemap; RSS returns 55 items
``www.setopati.com``         sitemap with 6 entries; RSS returns 5
``ratopati.com``             ``Crawl-delay: 20`` -- 180 articles/hour ceiling
===========================  ==========================================

So discovery runs as a cascade, stopping when the URL frontier is large enough:

1. ``Sitemap:`` directives in robots.txt, recursing through sitemap indexes.
2. Conventional sitemap paths (``/sitemap_index.xml``, ``/wp-sitemap.xml``, ...).
3. RSS/Atom feeds -- current but small, so used as a supplement.
4. **Paginated section crawl** -- the fallback that actually carries the Nepali
   sites: walk configured category paths through ``/page/N`` and harvest links
   matching the host's article-URL shape.

Extraction is deliberately dependency-light: JSON-LD ``articleBody`` when the
site publishes it (most modern CMSs do), otherwise the paragraph set inside the
main article container, with navigation, related-story and share-widget text
removed by rule. That extraction and cleaning is what makes this lane manual.

WordPress REST API shortcut (``params.wp_rest_api: true``)
------------------------------------------------------------
One site (onlinekhabar.com, Nepali) does not need the cascade above at all:
its ``robots.txt`` is ``User-agent: *`` with zero ``Disallow`` lines (verified
live 2026-08-18), and it runs on WordPress with the default REST API open at
``/wp-json/wp/v2/posts`` -- paginated JSON that returns each post's full
rendered HTML body directly, no per-article page fetch or link discovery
needed. Confirmed live: 462,154 posts, ~526 words/post average. When this
param is set, ``iter_documents`` walks ``?per_page=100&page=N`` until the WP
API returns an empty page (genuine exhaustion, not a guess about frontier
size) instead of running the sitemap/feed/section cascade.
"""

from __future__ import annotations

import html
import json
import re
import time
import urllib.parse
from collections.abc import Iterator
from typing import Any

from .. import schema
from ..nethttp import FetchError
from ..textnorm import devanagari_ratio, normalize, strip_urls
from .base import Budget, Collector

_LOC_RE = re.compile(rb"<loc>\s*(.*?)\s*</loc>", re.DOTALL | re.IGNORECASE)
_SITEMAP_BLOCK_RE = re.compile(rb"<sitemap>", re.IGNORECASE)
_RSS_LINK_RE = re.compile(
    rb"<link[^>]*>\s*(?:<!\[CDATA\[)?\s*(https?://[^<\]\s]+)", re.IGNORECASE
)
_HREF_RE = re.compile(r"""href\s*=\s*["']([^"'#]+)""", re.IGNORECASE)

_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript|svg|iframe|form)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE
)
_JSONLD_RE = re.compile(
    r'<script[^>]+type\s*=\s*["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.DOTALL | re.IGNORECASE,
)
_PARA_RE = re.compile(r"<p\b[^>]*>(.*?)</p>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL | re.IGNORECASE)
_TIME_RE = re.compile(
    r'<time[^>]*datetime\s*=\s*["\']([^"\']+)', re.IGNORECASE
)

# Article containers, most specific first. Restricting paragraph extraction to
# these removes sidebars and "you may also like" blocks without a DOM parser.
_CONTAINER_PATTERNS = (
    r'<div[^>]*(?:class|id)\s*=\s*["\'][^"\']*(?:article-?(?:body|content|detail)|'
    r'story-?(?:body|content|detail)|entry-?content|post-?content|news-?detail|'
    r'main-?content|the-?content)[^"\']*["\'][^>]*>(.*?)</div>\s*(?:</div>|<footer|<aside)',
    r"<article[^>]*>(.*?)</article>",
)

# Boilerplate that survives paragraph extraction on Indic news sites.
_BOILERPLATE_RE = re.compile(
    r"(?:यह भी पढ़ें|और पढ़ें|अधिक पढ़ें|इसे भी पढ़ें|ताजा खबरें|"
    r"यो पनि पढ्नुहोस्|सम्बन्धित समाचार|प्रकाशित मिति|थप समाचार|"
    r"Also Read|Read More|Advertisement|Sponsored|Share this|Follow us|"
    r"Copyright|सर्वाधिकार सुरक्षित|प्रतिक्रिया दिनुहोस्)"
    r".{0,120}",
    re.IGNORECASE,
)
_CREDIT_RE = re.compile(
    r"^\s*(?:इमेज स्रोत|फोटो स्रोत|तस्वीर|Getty Images|Reuters|AFP|AP|PTI|ANI|"
    r"एजेन्सी|रासस|एजेंसी)\s*[,:।]?\s*",
    re.IGNORECASE | re.MULTILINE,
)


def _absolutise(base: str, href: str) -> str:
    return urllib.parse.urljoin(base, html.unescape(href.strip()))


def _same_site(url: str, host: str) -> bool:
    netloc = urllib.parse.urlsplit(url).netloc.lower()
    host = host.lower()
    return netloc == host or netloc.endswith("." + host.removeprefix("www."))


class NewsCollector(Collector):
    """Discover article URLs on one news domain, then extract and clean bodies.

    Configuration lives in ``spec.params``:

    ``host``
        Bare domain, e.g. ``ekantipur.com``.
    ``article_pattern``
        Regex an URL must match to be treated as an article. Keeps the frontier
        free of tag pages, author pages and section indexes.
    ``sections``
        Paths used by the paginated fallback crawl.
    ``page_template``
        How the host expresses pagination, e.g. ``"{section}/page/{n}"``.
    ``max_pages_per_section``
        Depth cap for the fallback crawl.
    """

    def preflight(self) -> str | None:
        host = self.spec.params.get("host")
        if not host:
            return "no host configured"
        try:
            self.session.get(f"https://{host}/", check_robots=False)
        except FetchError as exc:
            return f"{host} unreachable: {exc.reason}"
        return None

    # -- URL discovery -----------------------------------------------------

    def _article_re(self) -> re.Pattern[str]:
        pattern = self.spec.params.get("article_pattern")
        return re.compile(pattern) if pattern else re.compile(r".")

    def _from_sitemaps(self, host: str, want: int, deadline: float) -> list[str]:
        """Strategy 1+2: robots.txt directives, then conventional paths."""
        found: list[str] = []
        seen_sitemaps: set[str] = set()
        article_re = self._article_re()

        queue = list(self.session.sitemaps_for(f"https://{host}/"))
        queue += [
            f"https://{host}{path}"
            for path in (
                "/sitemap_index.xml",
                "/sitemap.xml",
                "/wp-sitemap.xml",
                "/news-sitemap.xml",
                "/sitemap-index.xml",
            )
        ]

        while queue and len(found) < want:
            if time.monotonic() > deadline:
                return found
            sitemap_url = queue.pop(0)
            if sitemap_url in seen_sitemaps:
                continue
            seen_sitemaps.add(sitemap_url)
            if len(seen_sitemaps) > 400:
                break
            try:
                payload = self.session.get(sitemap_url, check_robots=False)
            except FetchError:
                continue

            locs = [m.decode("utf-8", "replace") for m in _LOC_RE.findall(payload)]
            if not locs:
                continue
            is_index = bool(_SITEMAP_BLOCK_RE.search(payload))
            for loc in locs:
                # Stop mid-sitemap, not merely between sitemaps. Checking only
                # at the outer loop let one large sitemap overshoot badly:
                # `news_amarujala` accumulated 91,628 URLs when it needed 80,
                # and spent its entire budget in discovery without fetching a
                # single article.
                if len(found) >= want:
                    return found
                loc = html.unescape(loc.strip())
                if is_index or loc.endswith((".xml", ".xml.gz")):
                    if _same_site(loc, host):
                        queue.append(loc)
                elif _same_site(loc, host) and article_re.search(loc):
                    found.append(loc)
        return found

    def _from_feeds(self, host: str) -> list[str]:
        """Strategy 3: RSS/Atom. Small but always current."""
        article_re = self._article_re()
        found: list[str] = []
        for path in ("/feed", "/rss", "/index.xml", "/feed/", "/rss.xml", "/atom.xml"):
            try:
                payload = self.session.get(f"https://{host}{path}", check_robots=False)
            except FetchError:
                continue
            for match in _LOC_RE.findall(payload) + _RSS_LINK_RE.findall(payload):
                url = html.unescape(match.decode("utf-8", "replace").strip())
                if _same_site(url, host) and article_re.search(url):
                    found.append(url)
        return found

    def _from_sections(
        self, host: str, want: int, budget: Budget, deadline: float
    ) -> list[str]:
        """Strategy 4: paginated section crawl.

        This is what makes the Nepali outlets viable -- they publish no usable
        sitemap, but every one of them paginates its category pages.
        """
        article_re = self._article_re()
        sections = self.spec.params.get("sections") or ["/"]
        template = self.spec.params.get("page_template", "{section}?page={n}")
        max_pages = int(self.spec.params.get("max_pages_per_section", 60))
        found: list[str] = []
        seen: set[str] = set()

        for section in sections:
            if len(found) >= want or time.monotonic() > deadline:
                break
            empty_streak = 0
            for page_no in range(1, max_pages + 1):
                if len(found) >= want or time.monotonic() > deadline:
                    break
                section_path = section.rstrip("/")
                if page_no == 1:
                    page_url = f"https://{host}{section}"
                else:
                    suffix = template.format(section=section_path, n=page_no)
                    page_url = (
                        suffix if suffix.startswith("http") else f"https://{host}{suffix}"
                    )
                try:
                    markup = self.session.get(page_url, check_robots=False).decode(
                        "utf-8", "replace"
                    )
                except FetchError:
                    empty_streak += 1
                    if empty_streak >= 3:
                        break
                    continue

                new_here = 0
                for href in _HREF_RE.findall(markup):
                    url = _absolutise(page_url, href)
                    url = url.split("?")[0]
                    if url in seen or not _same_site(url, host):
                        continue
                    if not article_re.search(url):
                        continue
                    seen.add(url)
                    found.append(url)
                    new_here += 1

                empty_streak = 0 if new_here else empty_streak + 1
                if empty_streak >= 3:
                    break
        return found

    def discover(self, host: str, want: int, budget: Budget) -> list[str]:
        """Run the discovery cascade until ``want`` URLs are found.

        Discovery gets a hard slice of the source's budget, not the whole thing.
        The fetch-side circuit breaker cannot help here -- it only trips once
        fetching has started -- and without a cap a host with slow, deeply
        paginated section pages spends the entire budget enumerating URLs it
        never downloads. Observed on `news_setopati`: 17.6 minutes, zero
        documents, all of it in discovery.
        """
        deadline = time.monotonic() + budget.max_seconds * float(
            self.spec.params.get("discovery_budget_fraction", 0.30)
        )
        urls: list[str] = []
        seen: set[str] = set()

        def absorb(candidates: list[str]) -> None:
            for url in candidates:
                clean = url.split("#")[0]
                if clean not in seen:
                    seen.add(clean)
                    urls.append(clean)

        absorb(self._from_sitemaps(host, want, deadline))
        self.report(status="RUNNING", phase="discovery", urls=len(urls))

        if len(urls) < want and time.monotonic() < deadline:
            absorb(self._from_feeds(host))
            self.report(status="RUNNING", phase="discovery+feeds", urls=len(urls))

        if len(urls) < want and time.monotonic() < deadline:
            absorb(self._from_sections(host, want - len(urls), budget, deadline))
            self.report(status="RUNNING", phase="discovery+sections", urls=len(urls))

        if not urls:
            raise RuntimeError(
                f"{host}: discovery found no article URLs within its budget "
                f"({budget.max_seconds * 0.30:.0f}s). Check article_pattern and the "
                f"configured section paths."
            )
        return urls

    # -- article extraction ------------------------------------------------

    @staticmethod
    def _from_jsonld(markup: str) -> tuple[str | None, str | None, str | None]:
        """Return (body, headline, published) from JSON-LD, if present."""
        for block in _JSONLD_RE.findall(markup):
            try:
                payload = json.loads(html.unescape(block.strip()))
            except json.JSONDecodeError:
                continue
            candidates = payload if isinstance(payload, list) else [payload]
            if isinstance(payload, dict) and "@graph" in payload:
                candidates = payload["@graph"]
            for node in candidates:
                if not isinstance(node, dict):
                    continue
                body = node.get("articleBody")
                if body and isinstance(body, str) and len(body) > 200:
                    return (
                        body,
                        node.get("headline") or node.get("name"),
                        node.get("datePublished"),
                    )
        return None, None, None

    @staticmethod
    def _from_paragraphs(markup: str) -> str | None:
        """Extract paragraphs from the article container, not the whole page."""
        scope = markup
        for pattern in _CONTAINER_PATTERNS:
            match = re.search(pattern, markup, re.DOTALL | re.IGNORECASE)
            if match and len(match.group(1)) > 400:
                scope = match.group(1)
                break

        paragraphs = []
        for chunk in _PARA_RE.findall(scope):
            text = html.unescape(_TAG_RE.sub(" ", chunk))
            text = " ".join(text.split())
            # Very short paragraphs on news pages are captions, bylines and
            # share prompts rather than article prose.
            if len(text) >= 40:
                paragraphs.append(text)
        if not paragraphs:
            return None
        return "\n\n".join(paragraphs)

    def extract(self, url: str, markup: str) -> tuple[str, dict[str, Any]] | None:
        """Turn a fetched page into clean article text plus metadata."""
        markup = _SCRIPT_STYLE_RE.sub(" ", markup)

        body, headline, published = self._from_jsonld(markup)
        method = "jsonld"
        if not body:
            body = self._from_paragraphs(markup)
            method = "paragraphs"
        if not body:
            return None

        if not headline:
            title_match = _TITLE_RE.search(markup)
            headline = (
                html.unescape(_TAG_RE.sub("", title_match.group(1))).strip()
                if title_match
                else None
            )
        if not published:
            time_match = _TIME_RE.search(markup)
            published = time_match.group(1) if time_match else None

        text = html.unescape(_TAG_RE.sub(" ", body))
        text = _CREDIT_RE.sub("", text)
        text = _BOILERPLATE_RE.sub(" ", text)
        text = strip_urls(text)
        text = normalize(text)

        return text, {
            "extraction": method,
            "headline": headline,
            "published": published,
        }

    # -- WordPress REST API shortcut ----------------------------------------

    def _iter_wp_rest_api(self, host: str, budget: Budget) -> Iterator[schema.Document]:
        """Walk ``/wp-json/wp/v2/posts`` directly -- see module docstring.

        No discovery phase, no per-URL fetch, no circuit breaker: the API
        either returns posts or returns an empty list once ``page`` runs past
        the end, which is unambiguous exhaustion rather than a host refusing
        us, so there's nothing here for a breaker to protect against.
        """
        min_chars = int(self.spec.params.get("min_chars", 500))
        min_deva = float(self.spec.params.get("min_devanagari_ratio", 0.6))
        per_page = int(self.spec.params.get("wp_per_page", 100))

        state = self.load_state() if self.resume else {}
        page = int(state.get("wp_page", 1))

        while budget.remaining_seconds > 0:
            url = f"https://{host}/wp-json/wp/v2/posts?per_page={per_page}&page={page}"
            try:
                posts = self.session.get_json(url, check_robots=False)
            except FetchError:
                break
            if not isinstance(posts, list) or not posts:
                break  # past the last page -- genuine exhaustion

            for post in posts:
                body = post.get("content", {}).get("rendered", "")
                text = html.unescape(_TAG_RE.sub(" ", _SCRIPT_STYLE_RE.sub(" ", body)))
                text = _CREDIT_RE.sub("", text)
                text = _BOILERPLATE_RE.sub(" ", text)
                text = strip_urls(text)
                text = normalize(text)
                if len(text) < min_chars or devanagari_ratio(text) < min_deva:
                    continue

                title = html.unescape(
                    _TAG_RE.sub("", post.get("title", {}).get("rendered", ""))
                ).strip() or None
                yield self.make_document(
                    text,
                    title=title,
                    origin_url=post.get("link"),
                    meta={
                        "domain": host,
                        "extraction": "wp_rest_api",
                        "published": post.get("date"),
                    },
                )

            page += 1
            self.save_state({"wp_page": page})
            self.report(status="RUNNING", phase="wp_rest_api", page=page)

    # -- driver ------------------------------------------------------------

    def iter_documents(self, budget: Budget) -> Iterator[schema.Document]:
        host = self.spec.params["host"]

        if self.spec.params.get("wp_rest_api"):
            yield from self._iter_wp_rest_api(host, budget)
            return

        min_chars = int(self.spec.params.get("min_chars", 500))
        min_deva = float(self.spec.params.get("min_devanagari_ratio", 0.6))

        state = self.load_state() if self.resume else {}
        frontier: list[str] = state.get("frontier", [])
        done: set[str] = set(state.get("done", []))

        if not frontier:
            if self.limit:
                # Smoke mode: the run will stop after `limit` documents, so
                # discovering a full-run frontier would make --smoke as slow as
                # the real thing and defeat the point of a five-minute check.
                want = self.limit * 4
            else:
                # Size the frontier from the word target: Indic news articles run
                # roughly 250-450 words, so 1 URL ~= 300 usable words, and a
                # generous multiplier absorbs extraction failures.
                want = max(
                    2_000, int(budget.target_words / 200) if budget.target_words else 5_000
                )
            frontier = self.discover(host, want, budget)
            self.save_state({"frontier": frontier, "done": []})

        self.report(status="RUNNING", phase="fetching", frontier=len(frontier))

        n_failed = 0
        n_short = 0
        processed = 0
        n_yielded = 0

        # Circuit breaker. Measured on a smoke run, `news_jagran` discovered
        # 2,500 valid article URLs and then failed 2,487 of 2,500 fetches,
        # spending 27.7 minutes to produce nothing. Overnight that is hours of
        # budget burned on a host that is refusing us. Once enough attempts have
        # been made to be confident, a source failing at this rate gives up and
        # says so, rather than grinding through its whole frontier.
        breaker_min_attempts = int(self.spec.params.get("breaker_min_attempts", 60))
        breaker_max_failure_rate = float(
            self.spec.params.get("breaker_max_failure_rate", 0.90)
        )

        for url in frontier:
            if budget.remaining_seconds <= 0:
                break
            if url in done:
                continue
            done.add(url)
            processed += 1

            if (
                processed >= breaker_min_attempts
                and n_yielded == 0
                and n_failed / processed >= breaker_max_failure_rate
            ):
                self.report(
                    status="RUNNING",
                    phase="circuit_breaker_tripped",
                    failed=n_failed,
                    attempted=processed,
                )
                raise RuntimeError(
                    f"{host}: {n_failed}/{processed} fetches failed and no document was "
                    f"extracted; abandoning this source rather than consuming its full "
                    f"time budget. Check the article_pattern and whether the host is "
                    f"blocking automated requests."
                )

            try:
                markup = self.session.get(url).decode("utf-8", "replace")
            except FetchError:
                n_failed += 1
                continue

            extracted = self.extract(url, markup)
            if extracted is None:
                n_failed += 1
                continue
            text, meta = extracted

            if len(text) < min_chars or devanagari_ratio(text) < min_deva:
                n_short += 1
                continue

            n_yielded += 1
            yield self.make_document(
                text,
                title=meta.get("headline"),
                origin_url=url,
                meta={
                    "domain": host,
                    "extraction": meta.get("extraction"),
                    "published": meta.get("published"),
                },
            )

            if processed % 200 == 0:
                self.save_state({"frontier": frontier, "done": sorted(done)})

        self.save_state(
            {
                "frontier": frontier,
                "done": sorted(done),
                "failed": n_failed,
                "short": n_short,
            }
        )

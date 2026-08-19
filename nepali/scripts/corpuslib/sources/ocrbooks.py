"""
OCR collection from scanned books -- the manual source the assignment names first.

The Phase 1 brief lists "OCR from books/PDFs" as its leading example of manual
collection, and it is the only source here whose text has never existed in
machine-readable form on the web. That makes it the most defensible 20%
contribution and the least likely to collide with Sangraha or IndicCorp during
cross-lane deduplication.

Pipeline per book: locate a public-domain scan on archive.org, download the
PDF, rasterise pages with PyMuPDF (no poppler dependency), run Tesseract with
the Devanagari model, then repair the text. That repair step is substantive
work, not a formality -- raw Devanagari OCR arrives with hyphenated line
breaks, page furniture, and pages that failed outright:

* pages whose mean per-word confidence falls below a floor are discarded rather
  than allowed to poison the corpus;
* running headers and page numbers are detected by cross-page repetition;
* soft line breaks inside paragraphs are rejoined so the tokenizer sees
  sentences instead of fixed-width fragments.

The collector degrades cleanly: if Tesseract or its language pack is absent, it
reports ``SKIPPED`` with the exact install command instead of failing the run.
"""

from __future__ import annotations

import collections
import io
import re
import shutil
import subprocess
import urllib.parse
from collections.abc import Iterator
from typing import Any

from .. import schema
from ..nethttp import FetchError
from ..textnorm import devanagari_ratio, normalize
from .base import Budget, Collector

ARCHIVE_SEARCH = "https://archive.org/advancedsearch.php"
ARCHIVE_META = "https://archive.org/metadata/"

# A trailing hyphen or Devanagari hyphen before a newline is a soft break
# introduced by typesetting, not a real hyphen.
_SOFT_BREAK_RE = re.compile(r"[-‐‑­]\s*\n\s*")
# A line ending without terminal punctuation continues the same paragraph.
_SOFT_WRAP_RE = re.compile(r"(?<![।?!:.\n])\n(?!\n)")
_PAGE_NUMBER_RE = re.compile(r"^\s*[\[\(]?\s*[०-९0-9ivxlcIVXLC]{1,5}\s*[\]\)]?\s*$")
# OCR noise: runs of isolated punctuation or Latin letters amid Devanagari.
_NOISE_LINE_RE = re.compile(r"^[^ऀ-ॿ]{0,4}$")


class OcrBooksCollector(Collector):
    """Download and OCR public-domain scanned books for one language."""

    def preflight(self) -> str | None:
        if shutil.which("tesseract") is None:
            return "tesseract binary not found (install: brew install tesseract)"
        lang_pack = self.spec.params.get("tesseract_lang", "hin")
        try:
            available = subprocess.run(
                ["tesseract", "--list-langs"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            ).stdout.split()
        except Exception as exc:  # noqa: BLE001
            return f"cannot query tesseract: {type(exc).__name__}: {exc}"
        if lang_pack not in available:
            return (
                f"tesseract language pack {lang_pack!r} missing "
                f"(install: brew install tesseract-lang)"
            )
        try:
            import pytesseract  # noqa: F401
        except ImportError:
            return "pytesseract not installed (pip install pytesseract)"
        try:
            import pymupdf  # noqa: F401
        except ImportError:
            try:
                import fitz  # noqa: F401
            except ImportError:
                return "PyMuPDF not installed (pip install pymupdf)"
        return None

    # -- discovery ---------------------------------------------------------

    def _search_books(self, want: int) -> list[dict[str, Any]]:
        """Find candidate scans, tolerating archive.org's search being down.

        Two endpoints are tried, then a curated seed list. As of 2026-08-17 both
        search endpoints return ``request_error`` with null items -- their
        Elasticsearch backend is erroring server-side, not rejecting our query.
        A source whose *discovery* is unavailable must still be able to collect,
        so ``seed_identifiers`` in the config provides a deterministic fallback
        that needs only the per-item metadata endpoint, which does work.
        """
        query = self.spec.params.get("archive_query", "language:Hindi AND mediatype:texts")
        found: list[dict[str, Any]] = []

        # 1. The scrape API (current-generation endpoint).
        try:
            url = "https://archive.org/services/search/v1/scrape?" + urllib.parse.urlencode(
                {"q": query, "fields": "identifier,title,year", "count": str(min(want, 100))}
            )
            payload = self.session.get_json(url, check_robots=False)
            items = [item for item in (payload.get("items") or []) if isinstance(item, dict)]
            found.extend(items)
        except Exception:  # noqa: BLE001 - fall through to the next strategy
            pass

        # 2. The legacy advanced-search endpoint.
        if not found:
            try:
                url = ARCHIVE_SEARCH + "?" + urllib.parse.urlencode(
                    {
                        "q": query,
                        "fl[]": ["identifier", "title", "year"],
                        "rows": str(min(want, 200)),
                        "page": "1",
                        "output": "json",
                    },
                    doseq=True,
                )
                payload = self.session.get_json(url, check_robots=False)
                docs = payload.get("response", {}).get("docs") or []
                found.extend(d for d in docs if isinstance(d, dict))
            except Exception:  # noqa: BLE001
                pass

        # 3. Curated identifiers from the config.
        if not found:
            seeds = self.spec.params.get("seed_identifiers") or []
            found = [{"identifier": s, "title": None, "year": None} for s in seeds]
            if found:
                self.report(status="RUNNING", phase="search_unavailable_using_seeds",
                            candidates=len(found))

        return found

    def _pdf_url(self, identifier: str) -> str | None:
        """Find the PDF inside an archive.org item, verifying its language first.

        The language check is not redundant with the search query: seed
        identifiers bypass search entirely, and archive.org collections such as
        the Digital Library of India mix languages freely under similar
        identifiers. Reading `metadata.language` costs nothing here and prevents
        a Bengali or English scan from being OCR'd with the Devanagari model and
        silently entering a monolingual corpus.
        """
        try:
            meta = self.session.get_json(ARCHIVE_META + identifier, check_robots=False)
        except FetchError:
            return None

        expected = self.spec.params.get("archive_language_codes")
        if expected:
            language = meta.get("metadata", {}).get("language")
            languages = language if isinstance(language, list) else [language]
            languages = [str(x).lower() for x in languages if x]
            if languages and not any(x in expected for x in languages):
                return None

        server = meta.get("server")
        directory = meta.get("dir")
        if not server or not directory:
            return None
        for entry in meta.get("files", []):
            name = entry.get("name", "")
            if name.lower().endswith(".pdf") and entry.get("format") not in {
                "Grayscale LuraTech PDF"
            }:
                return f"https://{server}{directory}/{name}"
        return None

    # -- OCR ---------------------------------------------------------------

    def _ocr_pdf(self, pdf_bytes: bytes, budget: Budget) -> tuple[list[str], float]:
        """Rasterise and OCR each page; return per-page text and mean confidence."""
        import pytesseract
        from PIL import Image

        try:
            import pymupdf as mupdf
        except ImportError:
            import fitz as mupdf

        lang_pack = self.spec.params.get("tesseract_lang", "hin")
        max_pages = int(self.spec.params.get("max_pages_per_book", 60))
        dpi = int(self.spec.params.get("dpi", 200))
        min_conf = float(self.spec.params.get("min_page_confidence", 60.0))

        document = mupdf.open(stream=pdf_bytes, filetype="pdf")
        pages: list[str] = []
        confidences: list[float] = []
        zoom = dpi / 72.0
        matrix = mupdf.Matrix(zoom, zoom)

        try:
            for page_no in range(min(document.page_count, max_pages)):
                if budget.remaining_seconds <= 0:
                    break
                pixmap = document.load_page(page_no).get_pixmap(matrix=matrix)
                image = Image.open(io.BytesIO(pixmap.tobytes("png")))

                data = pytesseract.image_to_data(
                    image,
                    lang=lang_pack,
                    output_type=pytesseract.Output.DICT,
                )
                words = [
                    (text, float(conf))
                    for text, conf in zip(data["text"], data["conf"])
                    if text.strip() and str(conf) not in {"-1", "-1.0"}
                ]
                if not words:
                    continue
                page_conf = sum(c for _, c in words) / len(words)
                if page_conf < min_conf:
                    # A low-confidence page is scanner failure, not language
                    # data. Keeping it would inject noise the tokenizer learns.
                    continue
                confidences.append(page_conf)
                pages.append(pytesseract.image_to_string(image, lang=lang_pack))
        finally:
            document.close()

        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return pages, mean_conf

    @staticmethod
    def _strip_running_headers(pages: list[str]) -> list[str]:
        """Remove lines that repeat across many pages -- headers and footers."""
        if len(pages) < 4:
            return pages
        first_lines: collections.Counter[str] = collections.Counter()
        last_lines: collections.Counter[str] = collections.Counter()
        for page in pages:
            lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
            if lines:
                first_lines[lines[0]] += 1
                last_lines[lines[-1]] += 1
        threshold = max(3, len(pages) // 3)
        repeated = {
            line
            for counter in (first_lines, last_lines)
            for line, count in counter.items()
            if count >= threshold
        }

        cleaned = []
        for page in pages:
            kept = [
                ln
                for ln in page.splitlines()
                if ln.strip() not in repeated
                and not _PAGE_NUMBER_RE.match(ln)
                and not _NOISE_LINE_RE.match(ln.strip())
            ]
            cleaned.append("\n".join(kept))
        return cleaned

    @staticmethod
    def _reflow(text: str) -> str:
        """Rejoin soft-wrapped lines so paragraphs survive as paragraphs."""
        text = _SOFT_BREAK_RE.sub("", text)      # hyphenated word split
        text = _SOFT_WRAP_RE.sub(" ", text)      # mid-sentence wrap
        return text

    # -- driver ------------------------------------------------------------

    def iter_documents(self, budget: Budget) -> Iterator[schema.Document]:
        min_chars = int(self.spec.params.get("min_chars", 1200))
        min_deva = float(self.spec.params.get("min_devanagari_ratio", 0.7))
        max_books = int(self.spec.params.get("max_books", 60))
        if self.limit:
            # Smoke mode: OCR costs minutes per book, so cap it hard. Two books
            # is enough to prove the download -> rasterise -> Tesseract ->
            # repair chain works end to end.
            max_books = min(max_books, 2)

        state = self.load_state() if self.resume else {}
        done: set[str] = set(state.get("done", []))

        books = self._search_books(max_books * 3)
        self.report(status="RUNNING", phase="search", candidates=len(books))

        processed = 0
        for book in books:
            if budget.remaining_seconds <= 0 or processed >= max_books:
                break
            identifier = book.get("identifier")
            if not identifier or identifier in done:
                continue
            done.add(identifier)

            pdf_url = self._pdf_url(identifier)
            if not pdf_url:
                continue
            try:
                pdf_bytes = self.session.get(pdf_url, check_robots=False)
            except FetchError:
                continue
            if len(pdf_bytes) < 50_000:
                continue

            try:
                pages, mean_conf = self._ocr_pdf(pdf_bytes, budget)
            except Exception:  # noqa: BLE001 - a corrupt scan must not end the run
                continue
            if not pages:
                continue

            pages = self._strip_running_headers(pages)
            text = normalize(self._reflow("\n".join(pages)))
            processed += 1
            self.report(
                status="RUNNING",
                phase="ocr",
                books=processed,
                last=identifier,
                conf=round(mean_conf, 1),
            )

            if len(text) < min_chars or devanagari_ratio(text) < min_deva:
                continue

            yield self.make_document(
                text,
                title=book.get("title"),
                origin_url=f"https://archive.org/details/{identifier}",
                meta={
                    "archive_id": identifier,
                    "year": book.get("year"),
                    "pages_kept": len(pages),
                    "mean_ocr_confidence": round(mean_conf, 2),
                    "tesseract_lang": self.spec.params.get("tesseract_lang"),
                },
            )
            self.save_state({"done": sorted(done)})

        self.save_state({"done": sorted(done), "books_processed": processed})

"""
Unicode normalisation and PII redaction for Devanagari corpora.

Two rules govern everything here:

1. **Normalise, do not transliterate.** Devanagari has multiple encodings for
   the same grapheme (nukta composition, the Devanagari danda vs. the ASCII
   period, Arabic-Indic vs. ASCII digits). Collapsing those is safe and makes
   deduplication and tokenizer training behave. Anything that would change the
   *reading* of the text is out of scope.

2. **Preserve length information.** Filters downstream key off character counts
   and symbol ratios, so normalisation must not silently delete large spans.
   ``normalize`` reports what it removed via :func:`normalization_report`.

Nepali and Hindi share the script but differ in convention -- notably Nepali
uses the danda far more heavily and permits the ``‍`` ZWJ inside
conjuncts such as ``पुर्‍याउने``. Blanket ZWJ stripping corrupts Nepali, so
joiner handling is language-aware and driven by config.
"""

from __future__ import annotations

import re
import unicodedata

# Zero-width and directional controls that carry no linguistic content in
# Devanagari text and routinely leak in from HTML and PDF extraction.
_ALWAYS_STRIP = (
    "​"  # ZERO WIDTH SPACE
    "‎"  # LEFT-TO-RIGHT MARK
    "‏"  # RIGHT-TO-LEFT MARK
    "﻿"  # BOM / ZERO WIDTH NO-BREAK SPACE
    "­"  # SOFT HYPHEN
    "⁠"  # WORD JOINER
)
_ALWAYS_STRIP_RE = re.compile(f"[{_ALWAYS_STRIP}]")

# ZWJ/ZWNJ are linguistically meaningful in Devanagari conjunct control.
# Stripped only when a language config opts in.
_JOINERS_RE = re.compile("[‌‍]")

# Control characters except tab/newline; these are always corruption.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Whitespace hygiene.
_SPACES_RE = re.compile(r"[ \t  -   　]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+(?=\n)")

# Devanagari digits -> ASCII. Optional: keeps number tokens from fragmenting the
# vocabulary across two digit systems. Applied only when configured.
_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Web-crawled corpora routinely carry newlines that were escaped once too often,
# arriving as the two characters backslash + 'n' rather than U+000A. Measured in
# Sangraha `unverified`: 3 of 50 sampled documents, 99 occurrences. Left alone
# these become literal "\n" pieces in the BPE vocabulary, wasting merges on an
# artefact. Only the whitespace escapes are undone -- a bare backslash is left
# untouched, since it can be legitimate content.
_LITERAL_ESCAPE_RE = re.compile(r"\\+([nrt])")


def _unescape_whitespace(match: re.Match) -> str:
    return {"n": "\n", "r": "\n", "t": " "}[match.group(1)]

# PII patterns. Deliberately conservative -- a false positive destroys text,
# and this corpus is public-web material where PII density is low.
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")
# Indian and Nepali mobile numbers, with or without country code, allowing
# space/dash separators but requiring a word boundary so years and IDs survive.
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?9[17][\s-]?)?[6-9]\d{9}(?!\d)"
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+")

EMAIL_TOKEN = "<EMAIL>"
PHONE_TOKEN = "<PHONE>"


def normalize(
    text: str,
    *,
    form: str = "NFC",
    strip_joiners: bool = False,
    normalize_digits: bool = False,
    collapse_whitespace: bool = True,
    unescape_whitespace: bool = True,
) -> str:
    """Return ``text`` in canonical form for this corpus.

    Parameters
    ----------
    form:
        Unicode normalisation form. NFC is correct for Devanagari: it composes
        nukta forms so that ``क़`` has one representation rather than two.
    strip_joiners:
        Remove ZWJ/ZWNJ. Safe for Hindi web text, **harmful for Nepali**, where
        ZWJ appears inside legitimate conjuncts. Language configs set this.
    normalize_digits:
        Map Devanagari digits to ASCII so numerals occupy one vocabulary region.
    """
    if not text:
        return ""

    if unescape_whitespace:
        text = _LITERAL_ESCAPE_RE.sub(_unescape_whitespace, text)

    text = unicodedata.normalize(form, text)
    text = _CONTROL_RE.sub("", text)
    text = _ALWAYS_STRIP_RE.sub("", text)

    if strip_joiners:
        text = _JOINERS_RE.sub("", text)
    if normalize_digits:
        text = text.translate(_DEVA_DIGITS)

    if collapse_whitespace:
        text = _SPACES_RE.sub(" ", text)
        text = _TRAILING_WS_RE.sub("", text)
        text = _BLANKLINES_RE.sub("\n\n", text)

    return text.strip()


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    """Replace emails and phone numbers with placeholders.

    Returns the redacted text and a per-category count, so the report can state
    how much PII was found rather than merely claiming a filter ran.
    """
    counts = {"email": 0, "phone": 0}

    def _email(match: re.Match) -> str:
        counts["email"] += 1
        return EMAIL_TOKEN

    def _phone(match: re.Match) -> str:
        counts["phone"] += 1
        return PHONE_TOKEN

    text = _EMAIL_RE.sub(_email, text)
    text = _PHONE_RE.sub(_phone, text)
    return text, counts


def strip_urls(text: str) -> str:
    """Remove bare URLs. Used for scraped HTML where links survive extraction."""
    return _URL_RE.sub(" ", text)


def devanagari_ratio(text: str) -> float:
    """Fraction of non-space characters lying in the Devanagari block.

    The cheapest and most reliable script gate for this project: no model to
    load, no failure mode, and it catches the English/Latin boilerplate that
    dominates scraped Indic pages before the LID model ever sees it.
    """
    considered = 0
    devanagari = 0
    for ch in text:
        if ch.isspace():
            continue
        considered += 1
        # Devanagari (0900-097F) plus Devanagari Extended (A8E0-A8FF).
        if "ऀ" <= ch <= "ॿ" or "꣠" <= ch <= "ꣿ":
            devanagari += 1
    if considered == 0:
        return 0.0
    return devanagari / considered


def symbol_ratio(text: str) -> float:
    """Fraction of non-space characters that carry no linguistic content.

    ``str.isalnum()`` looks like the obvious test here and is wrong for
    Devanagari: combining vowel signs (matras, e.g. the ``ा``/``ि``/``े`` in
    ``भारत``/``विशाल``/``देश``) are Unicode category Mc/Mn, and Python's
    ``isalnum`` returns False for every mark character. Under that test,
    ordinary Devanagari prose -- which is mark characters on nearly every
    syllable -- scores as 30-40% "symbols", making ``max_symbol_ratio`` reject
    normal text rather than the markup remnants and emoji spam it exists to
    catch. Letters (L*), marks (M*) and digits (N*) all count as linguistic
    content here; only punctuation/symbol/other categories count as symbols.
    """
    considered = 0
    symbols = 0
    for ch in text:
        if ch.isspace():
            continue
        considered += 1
        category = unicodedata.category(ch)
        if category[0] not in ("L", "M", "N"):
            symbols += 1
    if considered == 0:
        return 1.0
    return symbols / considered


def normalization_report(before: str, after: str) -> dict[str, int]:
    """Character-level delta, for the per-stage retention table in the report."""
    return {
        "chars_before": len(before),
        "chars_after": len(after),
        "chars_removed": max(0, len(before) - len(after)),
    }


def lid_ready(text: str, max_chars: int = 1200) -> str:
    """Flatten ``text`` into the single-line form fastText scores best.

    Newlines measurably degrade fastText LID confidence (a multi-line Nepali
    sample scored 0.58 versus 0.91 for the same text on one line), and very
    short strings are unreliable regardless -- ``नमस्ते`` alone is classified
    Marathi with 0.83 confidence. Callers gate on length before scoring.
    """
    return " ".join(text.split())[:max_chars]

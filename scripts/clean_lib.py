"""Shared cleaning/normalization/dedup logic for the Hindi and Bhojpuri pipelines.

Shared code only - each language's raw/interim/cleaned/splits data stays in its own
directory tree and is never mixed here. Implements the pipeline described in
report/phase1/phase1_strategy.md section 6:

  raw text -> NFC normalize + strip zero-width/control chars -> language-ID filter
  -> boilerplate/short-doc/malformed-Unicode removal -> exact + near-duplicate removal
"""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

try:
    from langdetect import DetectorFactory, LangDetectException, detect_langs

    DetectorFactory.seed = 0
except ImportError:
    detect_langs = None
    LangDetectException = Exception

# Zero-width and control characters that are NOT meaningful in Devanagari text:
# LRM/RLM (‎/‏, directional marks, unused in Devanagari), BOM
# (﻿), and C0/DEL control codes. Deliberately excludes ZWNJ (‌) and
# ZWJ (‍) - those control conjunct ligature formation in real Devanagari
# orthography (e.g. half-form vs. full-conjunct rendering) and stripping them
# can change a word's correct written form, not just its whitespace.
_ZERO_WIDTH_AND_CONTROL = re.compile(
    "[‎‏﻿\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)
_INLINE_WHITESPACE_RUN = re.compile(r"[ \t]+")
_BLANK_LINE_RUN = re.compile(r"\n{3,}")
_NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE_RUN = re.compile(r"\s+")
_REPLACEMENT_CHAR = "�"
# A token counts as "non-language" only if it is PURE Latin script (letters
# plus internal hyphen/apostrophe and trailing punctuation) with no digits and
# no Devanagari - e.g. "IRCTC", "Delhi-NCR", "Musk's". Deliberately narrow:
# digits stay per the course requirement, and mixed tokens (e.g. "5G",
# "iPhone14") are left alone since they're not pure foreign-script noise.
_PURE_LATIN_TOKEN = re.compile(r"^[A-Za-z]+(?:[.\-'][A-Za-z]+)*[.,!?;:\"'()\[\]]*$")

DEVANAGARI_BLOCK = (0x0900, 0x097F)


def normalize_text(text: str) -> str:
    """NFC-normalize and strip zero-width/control characters only (preserves
    Devanagari and all other real text, per phase1_strategy.md section 6)."""
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH_AND_CONTROL.sub("", text)
    text = _INLINE_WHITESPACE_RUN.sub(" ", text)
    text = _BLANK_LINE_RUN.sub("\n\n", text)
    return text.strip()


def devanagari_ratio(text: str) -> float:
    """Fraction of non-whitespace characters in the Devanagari Unicode block."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    dev = sum(1 for c in chars if DEVANAGARI_BLOCK[0] <= ord(c) <= DEVANAGARI_BLOCK[1])
    return dev / len(chars)


def malformed_ratio(text: str) -> float:
    """Fraction of characters that are the Unicode replacement char (decoding
    failures / mojibake)."""
    if not text:
        return 1.0
    return text.count(_REPLACEMENT_CHAR) / len(text)


def is_too_short(text: str, min_chars: int) -> bool:
    return len(text) < min_chars


def langdetect_verdict(text: str, expected_iso: str, sample_chars: int = 3000, min_prob: float = 0.85):
    """Returns True/False if langdetect can render a confident verdict, or None
    if it can't (too little/ambiguous text, or - notably - if langdetect has no
    label for the expected language at all, e.g. Bhojpuri ("bho") isn't in
    langdetect's supported-language list; callers must not trust a None as
    either pass or fail)."""
    if detect_langs is None:
        return None
    sample = text[:sample_chars]
    try:
        results = detect_langs(sample)
    except LangDetectException:
        return None
    if not results:
        return None
    top = results[0]
    return top.lang == expected_iso and top.prob >= min_prob


def strip_latin_tokens(text: str) -> tuple[str, int]:
    """Remove whitespace-delimited tokens that are purely Latin-script (course
    requirement: strip non-language words, digits stay). Devanagari tokens,
    digit tokens, and mixed-script/alphanumeric tokens are all left untouched.
    Line structure is preserved; a line that was entirely a removed token
    becomes an empty line (harmless - normalize_text's blank-line collapsing
    handles runs of these). Returns (cleaned_text, tokens_removed)."""
    removed = 0
    out_lines = []
    for line in text.split("\n"):
        kept = []
        for tok in line.split(" "):
            if tok and _PURE_LATIN_TOKEN.match(tok):
                removed += 1
                continue
            kept.append(tok)
        out_lines.append(" ".join(kept))
    return "\n".join(out_lines), removed


def dedup_key(text: str) -> str:
    """Hash of whitespace-collapsed, punctuation-stripped text. This is what
    phase1_strategy.md section 6 means by 'near-duplicate removal (normalized-
    text hash)' - it catches exact dupes and formatting/whitespace variants of
    the same document, not fuzzy/paraphrase similarity."""
    key = _WHITESPACE_RUN.sub(" ", text).strip()
    key = _NON_WORD.sub("", key).lower()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def read_metadata(meta_path: Path):
    if not meta_path.exists():
        return
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def reset_dir(out_dir: Path) -> None:
    """Clear a stage output directory (text/ + metadata.jsonl) so pipeline
    stages can be re-run from scratch rather than silently appending to stale
    output."""
    text_dir = out_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    for old in text_dir.glob("*.txt"):
        old.unlink()
    meta_path = out_dir / "metadata.jsonl"
    if meta_path.exists():
        meta_path.unlink()


def write_doc(out_dir: Path, doc_id: str, text: str, meta: dict) -> None:
    (out_dir / "text").mkdir(parents=True, exist_ok=True)
    (out_dir / "text" / f"{doc_id}.txt").write_text(text, encoding="utf-8")
    with (out_dir / "metadata.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

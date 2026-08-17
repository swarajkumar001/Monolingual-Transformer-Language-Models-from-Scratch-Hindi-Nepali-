"""Ingest manually-collected Bhojpuri text into data/raw/manual/, with metadata.

Three subcommands, matching the three manual-collection methods the project
requires at least 20% of tokens to come from:

  typed   - fold in text files you already typed/transcribed yourself.
  ocr     - run OCR over PDFs/images (requires the Tesseract binary installed
            separately - `pip install pytesseract` alone is NOT enough, it's just
            a wrapper). Tesseract has no dedicated Bhojpuri model; since Bhojpuri
            uses the same Devanagari script as Hindi, the `hin` traineddata is
            used by default (--tesseract-lang lets you override this).
  scrape  - fetch a list of URLs you provide and strip boilerplate HTML.

Every ingested document gets a metadata record identical in shape to the
downloaded-source records, so Phase 1 statistics can report the manual/downloaded
split honestly. Nothing here fabricates text - if a source produces no usable
text (e.g. OCR fails, a URL 404s), it is skipped and logged, not padded.
"""

import argparse
import hashlib
import json
from pathlib import Path

LANG = "bhojpuri"
RAW_MANUAL = Path(__file__).resolve().parents[1] / "raw" / "manual"


def document_id(method: str, key: str) -> str:
    digest = hashlib.sha1(f"{method}:{key}".encode("utf-8")).hexdigest()[:10]
    return f"bho_manual_{method}_{digest}"


def write_document(doc_id: str, text: str, meta: dict) -> None:
    (RAW_MANUAL / "text").mkdir(parents=True, exist_ok=True)
    (RAW_MANUAL / "text" / f"{doc_id}.txt").write_text(text, encoding="utf-8")
    meta_path = RAW_MANUAL / "metadata.jsonl"
    with meta_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")


def cmd_typed(args):
    """Register plain-text files you already typed or transcribed yourself."""
    src_dir = Path(args.input_dir)
    files = sorted(src_dir.glob("*.txt"))
    if not files:
        print(f"No .txt files found in {src_dir}")
        return
    count = 0
    for fp in files:
        text = fp.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        doc_id = document_id("typed", fp.name)
        meta = {
            "document_id": doc_id,
            "language": LANG,
            "source_type": "manual",
            "source_name": "typed_transcribed",
            "source_description": args.description or f"Typed/transcribed text: {fp.name}",
            "collection_method": "typed",
            "license_or_usage_note": args.license or "author-provided",
            "url": "",
            "characters": len(text),
            "split": None,
            "cleaning_status": "raw",
        }
        write_document(doc_id, text, meta)
        count += 1
    print(f"[{LANG}] typed: ingested {count} documents from {src_dir}")


def cmd_ocr(args):
    """OCR a folder of PDFs/images. Requires the Tesseract binary (no bho model - uses hin)."""
    try:
        import pytesseract
        from PIL import Image
        from pdf2image import convert_from_path
    except ImportError as e:
        print(f"Missing OCR dependency: {e}. Run: pip install pytesseract pdf2image Pillow")
        return

    src_dir = Path(args.input_dir)
    inputs = sorted(list(src_dir.glob("*.pdf")) + list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpg")))
    if not inputs:
        print(f"No PDFs/images found in {src_dir}")
        return

    count = 0
    for fp in inputs:
        try:
            if fp.suffix.lower() == ".pdf":
                pages = convert_from_path(str(fp))
            else:
                pages = [Image.open(fp)]
            text = "\n".join(pytesseract.image_to_string(p, lang=args.tesseract_lang) for p in pages).strip()
        except Exception as e:
            print(f"  OCR failed for {fp.name}: {e}")
            continue
        if not text:
            print(f"  OCR produced no text for {fp.name}, skipping")
            continue
        doc_id = document_id("ocr", fp.name)
        meta = {
            "document_id": doc_id,
            "language": LANG,
            "source_type": "manual",
            "source_name": "ocr",
            "source_description": args.description or f"OCR from {fp.name}",
            "collection_method": "ocr",
            "license_or_usage_note": args.license or "author-provided",
            "url": "",
            "characters": len(text),
            "split": None,
            "cleaning_status": "raw",
        }
        write_document(doc_id, text, meta)
        count += 1
        print(f"  OCR'd {fp.name}: {len(text)} chars")
    print(f"[{LANG}] ocr: ingested {count} documents from {src_dir}")


def cmd_pdf(args):
    """Extract text directly from PDFs that already carry a text layer (e.g.
    most modern government/textbook PDFs) - no OCR, no Tesseract needed. If a
    PDF turns out to be scanned images with no embedded text, pypdf returns
    little/nothing for it; that file is skipped and logged (use the `ocr`
    subcommand for that case instead, not this one)."""
    try:
        from pypdf import PdfReader
    except ImportError as e:
        print(f"Missing dependency: {e}. Run: pip install pypdf")
        return

    src_dir = Path(args.input_dir)
    files = sorted(src_dir.glob("*.pdf"))
    if not files:
        print(f"No PDFs found in {src_dir}")
        return
    count = 0
    for fp in files:
        try:
            reader = PdfReader(str(fp))
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        except Exception as e:
            print(f"  PDF read failed for {fp.name}: {e}")
            continue
        if len(text) < 200:
            print(f"  {fp.name}: only {len(text)} chars extracted - likely a scanned "
                  f"PDF with no text layer, needs the `ocr` subcommand instead, skipping")
            continue
        doc_id = document_id("pdf", fp.name)
        meta = {
            "document_id": doc_id,
            "language": LANG,
            "source_type": "manual",
            "source_name": "pdf_text_extraction",
            "source_description": args.description or f"PDF text extraction: {fp.name}",
            "collection_method": "pdf_text_extraction",
            "license_or_usage_note": args.license or "author-provided",
            "url": "",
            "characters": len(text),
            "split": None,
            "cleaning_status": "raw",
        }
        write_document(doc_id, text, meta)
        count += 1
        print(f"  {fp.name}: {len(text)} chars extracted")
    print(f"[{LANG}] pdf: ingested {count} documents from {src_dir}")


def cmd_scrape(args):
    """Fetch a list of URLs (one per line in a file) and strip boilerplate HTML."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as e:
        print(f"Missing scrape dependency: {e}. Run: pip install requests beautifulsoup4")
        return

    urls = [u.strip() for u in Path(args.url_file).read_text(encoding="utf-8").splitlines() if u.strip()]
    count = 0
    for url in urls:
        try:
            resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 (research collection)"})
            resp.raise_for_status()
        except Exception as e:
            print(f"  Failed to fetch {url}: {e}")
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
        if not text:
            print(f"  No text extracted from {url}, skipping")
            continue
        doc_id = document_id("scrape", url)
        meta = {
            "document_id": doc_id,
            "language": LANG,
            "source_type": "manual",
            "source_name": "scrape",
            "source_description": args.description or f"Scraped: {url}",
            "collection_method": "scrape",
            "license_or_usage_note": args.license or "verify site terms before use",
            "url": url,
            "characters": len(text),
            "split": None,
            "cleaning_status": "raw",
        }
        write_document(doc_id, text, meta)
        count += 1
        print(f"  Scraped {url}: {len(text)} chars")
    print(f"[{LANG}] scrape: ingested {count} documents from {args.url_file}")


def main():
    parser = argparse.ArgumentParser(description="Ingest manually-collected Bhojpuri text")
    sub = parser.add_subparsers(dest="command", required=True)

    p_typed = sub.add_parser("typed", help="Ingest already-typed/transcribed .txt files")
    p_typed.add_argument("--input-dir", required=True)
    p_typed.add_argument("--description", default=None)
    p_typed.add_argument("--license", default=None)
    p_typed.set_defaults(func=cmd_typed)

    p_ocr = sub.add_parser("ocr", help="OCR PDFs/images")
    p_ocr.add_argument("--input-dir", required=True)
    p_ocr.add_argument("--tesseract-lang", default="hin")
    p_ocr.add_argument("--description", default=None)
    p_ocr.add_argument("--license", default=None)
    p_ocr.set_defaults(func=cmd_ocr)

    p_pdf = sub.add_parser("pdf", help="Extract text directly from PDFs with a text layer (no OCR)")
    p_pdf.add_argument("--input-dir", required=True)
    p_pdf.add_argument("--description", default=None)
    p_pdf.add_argument("--license", default=None)
    p_pdf.set_defaults(func=cmd_pdf)

    p_scrape = sub.add_parser("scrape", help="Scrape a list of URLs")
    p_scrape.add_argument("--url-file", required=True)
    p_scrape.add_argument("--description", default=None)
    p_scrape.add_argument("--license", default=None)
    p_scrape.set_defaults(func=cmd_scrape)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

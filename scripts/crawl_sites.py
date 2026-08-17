"""Crawl WordPress-REST and sitemap-based sites into {lang}/data/raw/manual/.

Three discovery modes, chosen per-site based on what was actually confirmed
live before writing this script (not assumed):
  wordpress  - paginate the site's `/wp-json/wp/v2/posts` REST API directly.
               Returns each post's rendered HTML content without needing to
               crawl links. Confirmed working on khabarbhojpuri.com,
               jogira.com, anjoria.com, sirijan.com, bhojpuriplanet.com.
  sitemap    - parse sitemap.xml (or a sitemap index of sitemaps) for page
               URLs, then fetch + strip boilerplate from each page.
               NOTE: hindisamay.com's sitemap.xml is corrupted at the source
               (URLs contain truncated multi-byte UTF-8 fragments, every one
               tested 400s) - this mode is kept for sites where the sitemap
               is actually well-formed, hindisamay.com is not currently
               configured to use it.
  mediawiki  - enumerate pages via `action=query&list=allpages`, then render
               each with `action=parse&prop=text` and strip boilerplate same
               as the other modes. Confirmed working on kavitakosh.org
               (api at /kk/api.php, not /w/api.php) and gadyakosh.org
               (api at /gk/api.php). TextExtracts is not installed on either
               wiki (prop=extracts returns empty), hence the HTML-render
               route instead of asking for plaintext directly.

Site list and per-site license_status/notes come from --config, a JSON file
(see scripts/manual_sites.json). License notes are carried through into each
document's metadata verbatim - this script does not adjudicate copyright, it
records what was told to it plus the collection method, so the report can be
honest about provenance.
"""

import argparse
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANG_ISO = {"hindi": "hi", "bhojpuri": "bho"}
HEADERS = {"User-Agent": "Mozilla/5.0 (research collection; coursework LM training corpus)"}
REQUEST_DELAY = 0.5
MIN_CHARS = 80


_BOILERPLATE_CLASS_RE = ["btn", "dropdown", "toolbar", "navbar", "menu", "kkrachna-home"]


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    for tag in soup.find_all(class_=True):
        if not tag.attrs:
            continue
        classes = " ".join(tag.attrs.get("class") or [])
        if any(marker in classes for marker in _BOILERPLATE_CLASS_RE):
            tag.decompose()
    return "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())


def crawl_wordpress(site: dict, max_docs: int):
    base = site["base_url"].rstrip("/")
    page = 1
    docs_written = 0
    while docs_written < max_docs:
        per_page = min(50, max_docs - docs_written)
        url = f"{base}/wp-json/wp/v2/posts?per_page={per_page}&page={page}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
        except Exception as e:
            print(f"    page {page} request failed: {e}")
            break
        if r.status_code != 200:
            break
        try:
            posts = r.json()
        except ValueError:
            break
        if not posts:
            break
        for post in posts:
            title = post.get("title", {}).get("rendered", "")
            html = post.get("content", {}).get("rendered", "")
            text = html_to_text(f"{title}\n\n{html}")
            if len(text) < MIN_CHARS:
                continue
            yield {
                "text": text,
                "url": post.get("link", base),
                "title": title,
                "published": post.get("date", ""),
            }
            docs_written += 1
            if docs_written >= max_docs:
                break
        page += 1
        time.sleep(REQUEST_DELAY)


def _iter_sitemap_urls(sitemap_url: str, seen_sitemaps: set, depth: int = 0):
    if sitemap_url in seen_sitemaps or depth > 2:
        return
    seen_sitemaps.add(sitemap_url)
    try:
        r = requests.get(sitemap_url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception as e:
        print(f"    failed to parse sitemap {sitemap_url}: {e}")
        return
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    # sitemap index -> nested sitemaps
    for sm in root.findall(".//sm:sitemap/sm:loc", ns):
        yield from _iter_sitemap_urls(sm.text.strip(), seen_sitemaps, depth + 1)
    # urlset -> page urls
    for loc in root.findall(".//sm:url/sm:loc", ns):
        yield loc.text.strip()


def crawl_sitemap(site: dict, max_docs: int):
    base = site["base_url"].rstrip("/")
    sitemap_url = site.get("sitemap_url", urljoin(base + "/", "sitemap.xml"))
    docs_written = 0
    for page_url in _iter_sitemap_urls(sitemap_url, set()):
        if docs_written >= max_docs:
            break
        try:
            r = requests.get(page_url, headers=HEADERS, timeout=30)
            r.raise_for_status()
        except Exception:
            continue
        text = html_to_text(r.text)
        if len(text) < MIN_CHARS:
            continue
        yield {"text": text, "url": page_url, "title": "", "published": ""}
        docs_written += 1
        time.sleep(REQUEST_DELAY)


def crawl_mediawiki(site: dict, max_docs: int):
    api_url = site["api_url"]
    docs_written = 0
    apcontinue = None
    while docs_written < max_docs:
        params = {
            "action": "query", "list": "allpages", "apnamespace": 0,
            "aplimit": min(50, max_docs - docs_written), "format": "json",
        }
        if apcontinue:
            params["apcontinue"] = apcontinue
        try:
            r = requests.get(api_url, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"    allpages request failed: {e}")
            break
        pages = data.get("query", {}).get("allpages", [])
        if not pages:
            break
        for page in pages:
            title = page["title"]
            try:
                pr = requests.get(
                    api_url,
                    params={"action": "parse", "page": title, "format": "json", "prop": "text"},
                    headers=HEADERS, timeout=30,
                )
                pr.raise_for_status()
                html = pr.json().get("parse", {}).get("text", {}).get("*", "")
            except Exception:
                continue
            text = html_to_text(html)
            if len(text) < MIN_CHARS:
                continue
            yield {"text": text, "url": f"{api_url}?action=parse&page={title}", "title": title, "published": ""}
            docs_written += 1
            if docs_written >= max_docs:
                break
            time.sleep(REQUEST_DELAY)
        apcontinue = data.get("continue", {}).get("apcontinue")
        if not apcontinue:
            break


CRAWLERS = {"wordpress": crawl_wordpress, "sitemap": crawl_sitemap, "mediawiki": crawl_mediawiki}


def main():
    parser = argparse.ArgumentParser(description="Crawl configured sites into raw/manual/")
    parser.add_argument("--config", required=True, help="Path to a manual_sites.json config")
    parser.add_argument("--site-id", default=None, help="Crawl only this site_id (default: all in config)")
    parser.add_argument("--max-docs-per-site", type=int, default=500)
    args = parser.parse_args()

    sites = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.site_id:
        sites = [s for s in sites if s["site_id"] == args.site_id]

    for site in sites:
        lang = site["language"]
        lang_iso = LANG_ISO[lang]
        mode = site["mode"]
        crawler = CRAWLERS.get(mode)
        if crawler is None:
            print(f"[{site['site_id']}] unknown mode '{mode}', skipping")
            continue

        out_dir = PROJECT_ROOT / lang / "data" / "raw" / "manual"
        (out_dir / "text").mkdir(parents=True, exist_ok=True)

        print(f"[{site['site_id']}] crawling {site['base_url']} (mode={mode}, lang={lang})...")
        count = 0
        for doc in crawler(site, args.max_docs_per_site):
            digest = hashlib.sha1(doc["url"].encode("utf-8")).hexdigest()[:10]
            doc_id = f"{lang_iso}_manual_scrape_{digest}"
            record = {
                "document_id": doc_id,
                "language": lang,
                "source_type": "manual",
                "source_name": site["site_id"],
                "source_description": doc.get("title") or site.get("source_name", site["site_id"]),
                "collection_method": "web_scrape",
                "license_or_usage_note": site.get("license_notes", "unverified"),
                "url": doc["url"],
                "characters": len(doc["text"]),
                "split": None,
                "cleaning_status": "raw",
            }
            (out_dir / "text" / f"{doc_id}.txt").write_text(doc["text"], encoding="utf-8")
            with (out_dir / "metadata.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
        print(f"[{site['site_id']}] wrote {count} documents")


if __name__ == "__main__":
    main()

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
import os
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANG_ISO = {"hindi": "hi", "bhojpuri": "bho"}
HEADERS = {"User-Agent": "Mozilla/5.0 (research collection; coursework LM training corpus)"}
REQUEST_DELAY = 0.5
MIN_CHARS = 80
STALE_LOCK_SECONDS = 15 * 60  # a lock this old means its owner crashed/was killed, not that it's slow


def append_metadata_locked(meta_path: Path, record: dict, retries: int = 200, wait: float = 0.05):
    """Append one JSON line to metadata.jsonl safely even when multiple
    crawler processes for *different* site_ids share the same output file
    (e.g. gadya_kosh, kavita_kosh, patrika_hindi all write to
    hindi/data/raw/manual/metadata.jsonl concurrently) - site_lock() above
    only serializes same-site reruns, it does not protect this cross-site
    case. A bare open(...,'a').write() is not atomic across processes on
    Windows for a real JSON line; interleaved writes corrupted lines here in
    practice. Uses a short-lived companion lock file (atomic exclusive
    create) held only for the duration of one append, so sites can still
    crawl in parallel - this is not the crawl-duration site_lock."""
    lock_path = meta_path.with_suffix(meta_path.suffix + ".writelock")
    line = json.dumps(record, ensure_ascii=False) + "\n"
    for _ in range(retries):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except (FileExistsError, PermissionError):
            # Windows can raise PermissionError instead of FileExistsError
            # when O_CREAT|O_EXCL races against another process's create/
            # delete of the same lock file under contention (observed in
            # practice with 3+ crawlers hammering one shared lock) - both
            # mean "try again shortly", neither should crash the crawl.
            time.sleep(wait)
    else:
        raise TimeoutError(f"could not acquire write lock for {meta_path} after {retries} attempts")
    try:
        with meta_path.open("a", encoding="utf-8") as f:
            f.write(line)
    finally:
        lock_path.unlink(missing_ok=True)


@contextmanager
def site_lock(site_id: str, out_dir: Path):
    """Prevent two crawler processes from appending to the same site's
    metadata.jsonl concurrently. Two unlocked processes interleaving
    f.write() calls on the same file corrupted lines mid-write in practice
    (see report/phase1 notes) - this turns that race into a clean skip
    instead of silent data corruption."""
    lock_path = out_dir / f".{site_id}.lock"
    if lock_path.exists():
        age = time.time() - lock_path.stat().st_mtime
        if age < STALE_LOCK_SECONDS:
            raise SystemExit(
                f"[{site_id}] lock held by another process (age {age:.0f}s) - "
                f"skipping to avoid duplicate concurrent writers. If that process "
                f"is dead, delete {lock_path} and rerun."
            )
        print(f"[{site_id}] stale lock (age {age:.0f}s) from a crashed run, reclaiming")
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


_BOILERPLATE_CLASS_RE = ["btn", "dropdown", "toolbar", "navbar", "menu", "kkrachna-home"]


def get_with_retry(url: str, retries: int = 5, **kwargs):
    """requests.get with backoff. Without this, a transient blip (DNS hiccup,
    momentary connection drop) kills the whole crawl instead of just costing a
    few seconds - this happened in practice (gadyakosh.org and kavitakosh.org
    both failed on the same DNS resolution error at the same moment, while
    fetch_archive_org.py's retry loop rode through the identical blip)."""
    last_exc = None
    for attempt in range(retries):
        try:
            return requests.get(url, headers=HEADERS, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(min(30, 5 * (attempt + 1)))
    raise last_exc


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


def crawl_wordpress(site: dict, max_docs: int, existing_ids: set, lang_iso: str):
    base = site["base_url"].rstrip("/")
    page = 1
    docs_written = 0
    while docs_written < max_docs:
        per_page = min(50, max_docs - docs_written)
        url = f"{base}/wp-json/wp/v2/posts?per_page={per_page}&page={page}"
        try:
            r = get_with_retry(url, timeout=30)
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
            url = post.get("link", base)
            # The listing response already contains the full post content, so
            # there's no expensive fetch to skip here - but docs_written must
            # only count genuinely-new posts, or pagination can stop early
            # once a page of already-collected posts is hit.
            digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
            if f"{lang_iso}_manual_scrape_{digest}" in existing_ids:
                continue
            title = post.get("title", {}).get("rendered", "")
            html = post.get("content", {}).get("rendered", "")
            text = html_to_text(f"{title}\n\n{html}")
            if len(text) < MIN_CHARS:
                continue
            yield {
                "text": text,
                "url": url,
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
        r = get_with_retry(sitemap_url, timeout=30)
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


def crawl_sitemap(site: dict, max_docs: int, existing_ids: set, lang_iso: str):
    base = site["base_url"].rstrip("/")
    sitemap_url = site.get("sitemap_url", urljoin(base + "/", "sitemap.xml"))
    docs_written = 0
    for page_url in _iter_sitemap_urls(sitemap_url, set()):
        if docs_written >= max_docs:
            break
        # Skip the expensive fetch entirely for URLs already collected in a
        # prior run - url is known from the sitemap listing alone, so the
        # doc_id (a hash of it) can be checked before spending a request on
        # it. Without this, a rerun re-fetches every already-collected page
        # before it can reach any new ones.
        digest = hashlib.sha1(page_url.encode("utf-8")).hexdigest()[:10]
        if f"{lang_iso}_manual_scrape_{digest}" in existing_ids:
            continue
        try:
            r = get_with_retry(page_url, timeout=30)
            r.raise_for_status()
        except Exception:
            continue
        text = html_to_text(r.text)
        if len(text) < MIN_CHARS:
            continue
        yield {"text": text, "url": page_url, "title": "", "published": ""}
        docs_written += 1
        time.sleep(REQUEST_DELAY)


def crawl_mediawiki(site: dict, max_docs: int, existing_ids: set, lang_iso: str):
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
            r = get_with_retry(api_url, params=params, timeout=30)
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
            # Skip the expensive action=parse render entirely for pages
            # already collected - the url (and thus doc_id) is derivable from
            # the title alone via the cheap allpages listing, before paying
            # for the per-page fetch. Without this, a rerun re-fetches every
            # already-collected page (0.5s+ each) before reaching new ones -
            # for gadyakosh.org's 11k+ already-collected pages that's hours
            # of wasted requests before any new document gets written.
            candidate_url = f"{api_url}?action=parse&page={title}"
            digest = hashlib.sha1(candidate_url.encode("utf-8")).hexdigest()[:10]
            if f"{lang_iso}_manual_scrape_{digest}" in existing_ids:
                continue
            try:
                pr = get_with_retry(
                    api_url,
                    params={"action": "parse", "page": title, "format": "json", "prop": "text"},
                    timeout=30,
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

        try:
            with site_lock(site["site_id"], out_dir):
                _crawl_one_site(site, lang, lang_iso, mode, crawler, out_dir, args.max_docs_per_site)
        except SystemExit as e:
            print(e)
            continue


def _crawl_one_site(site, lang, lang_iso, mode, crawler, out_dir, max_docs_per_site):
        # Rerunning a crawl (e.g. after a restart) re-walks the same site from
        # the start, so without this check every already-collected doc would
        # be re-appended as a duplicate metadata line each time. Skipping
        # known doc_ids makes reruns incremental instead of duplicating.
        meta_path = out_dir / "metadata.jsonl"
        existing_ids = set()
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        existing_ids.add(json.loads(line)["document_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue

        print(f"[{site['site_id']}] crawling {site['base_url']} (mode={mode}, lang={lang}); "
              f"{len(existing_ids)} docs already collected from this site's out dir...")
        count = skipped_existing = 0
        for doc in crawler(site, max_docs_per_site, existing_ids, lang_iso):
            digest = hashlib.sha1(doc["url"].encode("utf-8")).hexdigest()[:10]
            doc_id = f"{lang_iso}_manual_scrape_{digest}"
            if doc_id in existing_ids:
                skipped_existing += 1
                continue
            if count >= max_docs_per_site:
                break
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
            append_metadata_locked(out_dir / "metadata.jsonl", record)
            count += 1
        print(f"[{site['site_id']}] wrote {count} new documents (skipped {skipped_existing} already collected)")


if __name__ == "__main__":
    main()

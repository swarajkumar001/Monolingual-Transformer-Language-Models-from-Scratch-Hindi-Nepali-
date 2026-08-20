"""
Polite, resilient HTTP for unattended overnight collection.

The collection run is expected to make on the order of 10^5 requests across a
dozen hosts while nobody is watching. Three things follow from that:

* **Politeness is not optional.** Probing revealed ``ratopati.com`` publishes
  ``Crawl-delay: 20``, and that Wikimedia hosts return HTTP 429 to generic user
  agents -- enough of them, fast enough, to make a working wiki look unreachable.
  Every request here passes through a per-host token bucket seeded from
  ``robots.txt``, behind a descriptive User-Agent as Wikimedia policy requires.
* **A single host must not be able to stall the run.** Every request has a
  timeout, every retry has a bounded backoff, and every source has a wall-clock
  budget enforced by the caller.
* **Transient failure is normal, and must not be fatal.** 429/5xx/timeouts are
  retried with jitter; 4xx other than 429 fail fast. Truncated responses
  (``IncompleteRead`` and its ``http.client.HTTPException`` siblings) are
  retried too -- they do *not* inherit from ``OSError``, so a naive except
  clause lets them escape and kill an entire source, which is exactly what one
  truncated read did to nagariknews on a smoke run.

The user agent identifies the project and a contact address, as the Wikimedia
User-Agent policy requires.
"""

from __future__ import annotations

import gzip
import http.client
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
import zlib
from dataclasses import dataclass, field

DEFAULT_USER_AGENT = (
    "LMA-Phase1-CorpusBot/1.0 "
    "(IIIT coursework; monolingual LM corpus construction; "
    "contact: swarajkumar25102002@gmail.com)"
)

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class FetchError(RuntimeError):
    """A request failed permanently after exhausting retries."""

    def __init__(self, url: str, reason: str, status: int | None = None) -> None:
        super().__init__(f"{url}: {reason}")
        self.url = url
        self.reason = reason
        self.status = status


class RobotsDisallowed(FetchError):
    """The host's robots.txt forbids this path for our user agent."""


@dataclass
class HostPolicy:
    """Rate-limit state for one host, derived from its robots.txt."""

    host: str
    min_interval: float = 0.34          # seconds between requests (~3/s default)
    last_request: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)
    robots: urllib.robotparser.RobotFileParser | None = None
    sitemaps: list[str] = field(default_factory=list)
    consecutive_429: int = 0

    def wait(self) -> None:
        """Block until this host is allowed another request."""
        with self.lock:
            now = time.monotonic()
            gap = self.min_interval - (now - self.last_request)
            if gap > 0:
                time.sleep(gap)
            self.last_request = time.monotonic()

    def penalise(self) -> None:
        """Back off after a 429: double the interval, capped."""
        with self.lock:
            self.consecutive_429 += 1
            self.min_interval = min(self.min_interval * 2, 30.0)

    def relax(self) -> None:
        self.consecutive_429 = 0


class PoliteSession:
    """Thread-safe HTTP client with per-host rate limiting and robots support.

    A single instance is shared across all collection threads. Rate limits are
    tracked per host, so parallel work on different domains proceeds at full
    speed while a slow host throttles only itself.
    """

    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        default_interval: float = 0.34,
        timeout: float = 30.0,
        max_retries: int = 4,
        respect_robots: bool = True,
    ) -> None:
        self.user_agent = user_agent
        self.default_interval = default_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self.respect_robots = respect_robots
        self._hosts: dict[str, HostPolicy] = {}
        self._hosts_lock = threading.Lock()
        self.n_requests = 0
        self.n_retries = 0
        self.n_failures = 0
        self._counter_lock = threading.Lock()

    # -- host policy -------------------------------------------------------

    def policy_for(self, url: str) -> HostPolicy:
        host = urllib.parse.urlsplit(url).netloc
        with self._hosts_lock:
            policy = self._hosts.get(host)
            if policy is None:
                policy = HostPolicy(host=host, min_interval=self.default_interval)
                self._hosts[host] = policy
                needs_robots = True
            else:
                needs_robots = False
        if needs_robots and self.respect_robots:
            self._load_robots(policy, url)
        return policy

    def _load_robots(self, policy: HostPolicy, sample_url: str) -> None:
        """Fetch robots.txt, honouring Crawl-delay and recording Sitemap hints."""
        parts = urllib.parse.urlsplit(sample_url)
        robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)
        try:
            raw = self._raw_get(robots_url, honour_policy=False)
            text = raw.decode("utf-8", errors="replace")
            parser.parse(text.splitlines())
            policy.robots = parser
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.lower().startswith("sitemap:"):
                    policy.sitemaps.append(stripped.split(":", 1)[1].strip())
            delay = parser.crawl_delay(self.user_agent)
            if delay:
                # Honour the published delay exactly; ratopati.com asks for 20s.
                policy.min_interval = max(policy.min_interval, float(delay))
        except Exception:  # noqa: BLE001 - absent robots.txt means "allowed"
            policy.robots = None

    def allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        policy = self.policy_for(url)
        if policy.robots is None:
            return True
        try:
            return policy.robots.can_fetch(self.user_agent, url)
        except Exception:  # noqa: BLE001
            return True

    def sitemaps_for(self, url: str) -> list[str]:
        """Sitemap URLs advertised in the host's robots.txt."""
        return list(self.policy_for(url).sitemaps)

    # -- fetching ----------------------------------------------------------

    def _raw_get(self, url: str, *, honour_policy: bool = True) -> bytes:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "*/*",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = response.read()
            encoding = (response.headers.get("Content-Encoding") or "").lower()
        if encoding == "gzip" or payload[:2] == b"\x1f\x8b":
            try:
                payload = gzip.decompress(payload)
            except Exception:  # noqa: BLE001 - some hosts mislabel encoding
                pass
        elif encoding == "deflate":
            try:
                payload = zlib.decompress(payload)
            except Exception:  # noqa: BLE001
                try:
                    payload = zlib.decompress(payload, -zlib.MAX_WBITS)
                except Exception:  # noqa: BLE001
                    pass
        return payload

    def get(self, url: str, *, check_robots: bool = True) -> bytes:
        """Fetch ``url``, respecting rate limits and retrying transient errors."""
        if check_robots and not self.allowed(url):
            raise RobotsDisallowed(url, "disallowed by robots.txt")

        policy = self.policy_for(url)
        last_error: str = "unknown"
        last_status: int | None = None

        for attempt in range(self.max_retries):
            policy.wait()
            try:
                with self._counter_lock:
                    self.n_requests += 1
                payload = self._raw_get(url)
                policy.relax()
                return payload
            except urllib.error.HTTPError as exc:
                last_status = exc.code
                last_error = f"HTTP {exc.code}"
                if exc.code == 429:
                    policy.penalise()
                if exc.code not in RETRYABLE_STATUS:
                    break
            except urllib.error.URLError as exc:
                last_error = f"URLError: {exc.reason}"
            except http.client.HTTPException as exc:
                # IncompleteRead, ChunkedEncodingError and friends inherit from
                # HTTPException, NOT from OSError, so without this clause a
                # truncated response escapes the retry loop and kills the whole
                # source. Observed on nagariknews: one IncompleteRead ended a
                # source that had collected nothing yet.
                last_error = f"{type(exc).__name__}: {exc}"
            except (TimeoutError, OSError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            with self._counter_lock:
                self.n_retries += 1
            # Exponential backoff with jitter; bounded so one bad host cannot
            # consume the overnight budget.
            time.sleep(min(2.0 ** attempt + random.uniform(0, 0.75), 45.0))

        with self._counter_lock:
            self.n_failures += 1
        raise FetchError(url, last_error, last_status)

    def get_text(self, url: str, *, check_robots: bool = True) -> str:
        return self.get(url, check_robots=check_robots).decode("utf-8", errors="replace")

    def get_json(self, url: str, *, check_robots: bool = True) -> dict:
        import json

        return json.loads(self.get(url, check_robots=check_robots))

    def stats(self) -> dict[str, int]:
        return {
            "requests": self.n_requests,
            "retries": self.n_retries,
            "failures": self.n_failures,
            "hosts": len(self._hosts),
        }

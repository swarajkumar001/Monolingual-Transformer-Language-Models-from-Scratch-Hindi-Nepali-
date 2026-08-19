#!/usr/bin/env python3
"""
Step 01 -- collect every configured source for this language.

Run directly to collect one language:

    python hindi/scripts/s01_collect.py                 # full run
    python hindi/scripts/s01_collect.py --smoke         # 200 docs/source, ~2 min
    python hindi/scripts/s01_collect.py --only wikipedia_api,news_bbc_hindi
    python hindi/scripts/s01_collect.py --lane manual
    python hindi/scripts/s01_collect.py --resume

Or let ``collect.py`` (in this same ``scripts/`` directory) drive S00 + S01
together, always resuming so re-running it never re-visits an already
satisfied source; run the same command from the other language's
``scripts/collect.py`` to collect both, in two terminals.

Design
------
Sources run concurrently in a thread pool. The work is network-bound, and the
:class:`~corpuslib.nethttp.PoliteSession` rate-limits per *host*, so parallel
sources on different domains proceed at full speed while a slow host throttles
only itself. Wikimedia hosts share a limiter automatically because they share
a netloc key per subdomain.

Nothing here can abort the run. Each source is wrapped by
:meth:`Collector.run`, which converts any exception into a ``FAILED`` result,
so one dead domain costs its own budget and nothing else. That property is the
entire point of an unattended overnight run.

Progress is written to ``data/collection_status.json`` every few seconds so the
run can be inspected from another terminal while it is still going.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpuslib import runlog, schema  # noqa: E402
from corpuslib.config import load_language_config, load_sources  # noqa: E402
from corpuslib.jsonlio import atomic_write_json, human_bytes  # noqa: E402
from corpuslib.ledger import Ledger  # noqa: E402
from corpuslib.nethttp import PoliteSession  # noqa: E402
from corpuslib.sources import FAILED, OK, PARTIAL, SKIPPED, build_collector  # noqa: E402

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class ProgressBoard:
    """Thread-safe live view of every source's state, persisted to disk.

    A background heartbeat rewrites the status file every 30 seconds even when
    no source has reported. Without it, a source in a long discovery phase
    leaves the file frozen, and from the outside a healthy run that is patiently
    walking sitemaps is indistinguishable from a hung one -- which is precisely
    the question ``data/collection_status.json`` (read from another terminal)
    exists to answer at 3am.
    """

    HEARTBEAT_SECONDS = 30.0

    def __init__(self, status_path: Path, lang: str) -> None:
        self.status_path = status_path
        self.lang = lang
        self.started = time.time()
        self._state: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._last_flush = 0.0
        self._stop = threading.Event()
        self._heartbeat = threading.Thread(target=self._beat, daemon=True)
        self._heartbeat.start()

    def _beat(self) -> None:
        while not self._stop.wait(self.HEARTBEAT_SECONDS):
            try:
                self.flush()
            except Exception:  # noqa: BLE001 - a status write must never kill a run
                pass

    def stop(self) -> None:
        self._stop.set()

    def update(self, source_id: str, fields: dict) -> None:
        with self._lock:
            entry = self._state.setdefault(source_id, {})
            entry.update(fields)
            entry["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            now = time.time()
            should_flush = now - self._last_flush > 5.0
            if should_flush:
                self._last_flush = now
                snapshot = json.loads(json.dumps(self._state, default=str))
        if should_flush:
            atomic_write_json(
                self.status_path,
                {
                    "lang": self.lang,
                    "started_at": datetime.fromtimestamp(
                        self.started, timezone.utc
                    ).isoformat(timespec="seconds"),
                    "elapsed_seconds": round(time.time() - self.started, 1),
                    "sources": snapshot,
                },
            )

    def flush(self) -> None:
        with self._lock:
            snapshot = json.loads(json.dumps(self._state, default=str))
        atomic_write_json(
            self.status_path,
            {
                "lang": self.lang,
                "started_at": datetime.fromtimestamp(
                    self.started, timezone.utc
                ).isoformat(timespec="seconds"),
                "elapsed_seconds": round(time.time() - self.started, 1),
                "sources": snapshot,
            },
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect all configured corpus sources for this language.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated source_ids to run (default: every enabled source)",
    )
    parser.add_argument(
        "--lane",
        choices=["manual", "downloaded", "both"],
        default="both",
        help="restrict collection to one lane",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="collect at most --smoke-docs per source; use this before a full run",
    )
    parser.add_argument("--smoke-docs", type=int, default=200)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep existing shards and continue from each source's checkpoint",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="concurrent sources; raise only if the network is the bottleneck",
    )
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=0,
        help="override every source's individual time budget",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    config = load_language_config()
    config.ensure_dirs()
    registry = load_sources()
    log = runlog.StepLogger(config.lang, config.events_path)

    selected = registry.enabled()
    if args.lane != "both":
        selected = [s for s in selected if s.lane == args.lane]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        unknown = wanted - {s.source_id for s in registry.all()}
        if unknown:
            print(f"unknown source_id(s): {sorted(unknown)}", file=sys.stderr)
            return 2
        selected = [s for s in selected if s.source_id in wanted]

    if not selected:
        print("no sources selected", file=sys.stderr)
        return 2

    headroom = float(config.targets.get("collection_headroom", 1.0))
    if args.smoke:
        headroom = 1.0

    banner = (
        f"{config.display_name} (Model {config.model_role}) — collecting "
        f"{len(selected)} source(s), {args.workers} workers"
        + (f", SMOKE {args.smoke_docs} docs/source" if args.smoke else "")
    )
    print("=" * 78)
    print(banner)
    print("=" * 78, flush=True)

    log(
        "S01",
        f"Collection start ({config.display_name})",
        runlog.SUCCESS,
        f"Starting collection of {len(selected)} sources "
        f"({'smoke' if args.smoke else 'full'} mode, headroom {headroom}x).",
        metrics={
            "sources": len(selected),
            "manual_sources": sum(1 for s in selected if s.lane == schema.LANE_MANUAL),
            "downloaded_sources": sum(
                1 for s in selected if s.lane == schema.LANE_DOWNLOADED
            ),
        },
    )

    board = ProgressBoard(config.status_path, config.lang)
    session = PoliteSession()
    started = time.time()
    results = []

    def run_one(spec):
        if args.max_seconds:
            spec = dataclasses.replace(spec, max_seconds=args.max_seconds)
        collector = build_collector(
            spec,
            lang=config.lang,
            lang_code=config.lang_code,
            out_root=config.raw_dir,
            session=session,
            resume=args.resume,
            headroom=headroom,
            progress=board.update,
            limit=args.smoke_docs if args.smoke else 0,
        )
        return collector.run()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, spec): spec for spec in selected}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - never let one source kill the run
                from corpuslib.sources import CollectionResult

                result = CollectionResult(
                    source_id=spec.source_id,
                    lane=spec.lane,
                    status=FAILED,
                    message=f"{type(exc).__name__}: {exc}",
                    target_words=spec.target_words,
                )
            results.append(result)

            icon = {OK: "OK  ", PARTIAL: "PART", SKIPPED: "SKIP", FAILED: "FAIL"}[result.status]
            print(
                f"[{icon}] {result.source_id:<24} {result.lane:<11} "
                f"{result.docs:>8,} docs  {result.words:>13,} words  "
                f"{human_bytes(result.bytes):>9}  {result.seconds/60:>6.1f}m",
                flush=True,
            )
            if result.message:
                print(f"       └─ {result.message}", flush=True)

            log(
                "S01",
                f"{spec.source_name}",
                {
                    OK: runlog.SUCCESS,
                    PARTIAL: runlog.WARNING,
                    SKIPPED: runlog.WARNING,
                    FAILED: runlog.BLOCKED,
                }[result.status],
                result.message or f"Collected {result.docs:,} documents.",
                metrics={
                    "lane": result.lane,
                    "docs": result.docs,
                    "words_estimate": result.words,
                    "bytes": result.bytes,
                    "seconds": round(result.seconds, 1),
                    "target_fraction": round(result.target_fraction, 3),
                },
            )
            board.flush()

    board.stop()
    board.flush()

    # -- collection ledger -------------------------------------------------
    ledger = Ledger.from_manifests(config.lang, config.raw_dir, stage="post_collection")
    summary = ledger.write(config.interim_dir / "collection_ledger.json")

    elapsed = time.time() - started
    print()
    print(ledger.render_table("words"))
    print()

    floor = config.manual_floor
    ratio = ledger.manual_ratio("words")
    meets = ratio >= floor
    shortfall = ledger.shortfall_words(floor)
    downloaded_tokens_cap = int(
        config.targets.get(
            "downloaded_tokens_cap", round(config.targets.get("final_tokens", 500_000_000) * (1 - floor))
        )
    )

    print(f"manual share (words, pre-cleaning): {ratio:.2%}  floor {floor:.0%}")
    if meets:
        print("  → floor met at collection time.")
    else:
        print(f"  → short by {shortfall:,} manual words at the current downloaded size.")
        print(
            f"  → s10_balance_lanes caps the downloaded lane at a fixed "
            f"{downloaded_tokens_cap:,} tokens regardless -- it will NOT force this ratio by "
            f"cutting downloaded further. Collect more manual: `s01_collect.py --lane manual "
            f"--resume`, or add sources in configs/sources.yaml."
        )

    log(
        "S01",
        f"Collection summary ({config.display_name})",
        runlog.SUCCESS if meets else runlog.WARNING,
        f"Collected {ledger.total('words'):,} estimated words across "
        f"{len(summary['sources'])} sources in {elapsed/3600:.2f}h. "
        f"Manual share {ratio:.2%} (floor {floor:.0%}).",
        metrics={
            "total_words_estimate": ledger.total("words"),
            "manual_words": summary["manual"],
            "downloaded_words": summary["downloaded"],
            "manual_ratio": round(ratio, 4),
            "hours": round(elapsed / 3600, 2),
            "http_requests": session.stats()["requests"],
            "http_failures": session.stats()["failures"],
        },
        decision=(
            "manual floor met at collection"
            if meets
            else f"downloaded lane is capped at a fixed {downloaded_tokens_cap:,} tokens in s10, "
                 f"regardless of manual size -- collect more manual instead"
        ),
    )

    failed = [r for r in results if r.status == FAILED]
    print(
        f"\n{len(results)} sources: "
        f"{sum(1 for r in results if r.status == OK)} OK, "
        f"{sum(1 for r in results if r.status == PARTIAL)} partial, "
        f"{sum(1 for r in results if r.status == SKIPPED)} skipped, "
        f"{len(failed)} failed  ·  {elapsed/3600:.2f}h elapsed"
    )
    # A failed source is reported but does not fail the step: the balance stage
    # can still produce a compliant corpus from what did arrive.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Structured run logging.

Two audiences, two formats, one call site:

* ``log.md`` at the repository root stays append-only and human-readable. It is
  the file to open after an overnight run to see what happened and why, in the
  format the execution plan specifies.
* ``<lang>/data/interim/run_events.jsonl`` receives the same events as machine
  records, so the report generator can build retention tables without parsing
  prose.

Every step reports one of four statuses. ``SUCCESS`` means the step met its own
exit check -- not merely that it did not crash. That distinction is what the
previous run got wrong: a stage that deleted 94% of the corpus logged
``SUCCESS`` because no exception escaped.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

SUCCESS = "SUCCESS"
WARNING = "WARNING"
FALLBACK = "FALLBACK_USED"
BLOCKED = "BLOCKED"

_LOCK = threading.Lock()


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists() or (parent / "LMA_Individual_Project_v1.pdf").exists():
            return parent
    return here.parents[3]


REPO_ROOT = _repo_root()
LOG_PATH = REPO_ROOT / "log.md"


def _fmt_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def log_step(
    step_id: str,
    name: str,
    status: str,
    what_happened: str,
    *,
    metrics: dict[str, Any] | None = None,
    decision: str | None = None,
    fallback: str | None = None,
    lang: str | None = None,
    events_path: Path | None = None,
) -> None:
    """Append one step record to ``log.md`` and the structured event log.

    Writes are serialised with a lock and an ``O_APPEND`` open so that the
    parallel collectors in the overnight run cannot interleave partial lines.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"### [{timestamp}] {step_id} — {name}",
        f"- Status: {status}",
        f"- What happened: {what_happened}",
    ]
    if metrics:
        rendered = ", ".join(f"{k}={_fmt_metric(v)}" for k, v in metrics.items())
        lines.append(f"- Metrics: {rendered}")
    if decision:
        lines.append(f"- Decision made: {decision}")
    if fallback:
        lines.append(f"- Fallback triggered: {fallback}")
    block = "\n".join(lines) + "\n\n\n"

    with _LOCK:
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())

        if events_path is not None:
            events_path.parent.mkdir(parents=True, exist_ok=True)
            event = {
                "ts": timestamp,
                "step": step_id,
                "name": name,
                "status": status,
                "lang": lang,
                "message": what_happened,
                "metrics": metrics or {},
                "decision": decision,
                "fallback": fallback,
            }
            with open(events_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


class StepLogger:
    """Convenience binder so a script does not repeat its language and paths."""

    def __init__(self, lang: str, events_path: Path) -> None:
        self.lang = lang
        self.events_path = events_path

    def __call__(
        self,
        step_id: str,
        name: str,
        status: str,
        what_happened: str,
        **kwargs: Any,
    ) -> None:
        log_step(
            step_id,
            name,
            status,
            what_happened,
            lang=self.lang,
            events_path=self.events_path,
            **kwargs,
        )


def read_events(events_path: Path) -> list[dict[str, Any]]:
    if not Path(events_path).exists():
        return []
    out = []
    with Path(events_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out

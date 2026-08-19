"""
One-time disk cleanup: wipe regenerable pipeline output, and trim the raw
DOWNLOADED lane down to the same token budget ``s10_balance_lanes`` enforces
later, so collected-but-unusable excess doesn't sit on disk.

The raw MANUAL lane is never touched by anything in this module -- it is the
scarce, hours-to-recollect resource; deletion and trimming only ever apply to
what a re-run of the pipeline (interim/final/tokenizer) or a fresh download
(the excess beyond the downloaded cap) can reproduce.

Trimming works at whole-shard-file granularity, not per-document: each
downloaded source's raw shards (``part-NNNNN.jsonl``) are kept in order up to
that source's proportional share of the budget, and the remaining shard
files are deleted outright. This is coarser than the exact, seeded,
per-document selection ``s10_balance_lanes`` does later on the *cleaned*
corpus -- that step remains the precise, authoritative cap. This one just
keeps raw disk usage roughly in line with it, cheaply, without rewriting
JSONL contents.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import LanguageConfig, tokens_per_word
from .jsonlio import atomic_write_json

REGENERABLE_DIRS = ("interim", "final")
REGENERABLE_ROOTS = ("tokenizer",)  # relative to the language root, not data/


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def human(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


@dataclass
class SourceTrimPlan:
    source_id: str
    source_dir: Path
    action: str  # "keep_all" | "trim"
    words_before: int
    words_after: int
    docs_after: int
    chars_after: int
    keep_shards: list[Path] = field(default_factory=list)
    drop_shards: list[Path] = field(default_factory=list)


@dataclass
class CleanPlan:
    interim_bytes: dict[str, int]
    final_bytes: dict[str, int]
    tokenizer_bytes: dict[str, int]
    downloaded_cap_words: int
    downloaded_words_before: int
    downloaded_words_after: int
    sources: list[SourceTrimPlan]

    def render(self, config: LanguageConfig) -> str:
        lines = [f"Cleanup plan for {config.display_name}:", ""]
        lines.append("DELETE (regenerable by re-running the pipeline):")
        for name in REGENERABLE_DIRS:
            lines.append(f"  {config.lang}/data/{name}/  ({human(self.interim_bytes.get(name, 0) if name == 'interim' else self.final_bytes.get(name, 0))})")
        lines.append(f"  {config.lang}/tokenizer/  ({human(sum(self.tokenizer_bytes.values()))})")
        lines.append("")
        lines.append(
            f"TRIM raw DOWNLOADED lane: {self.downloaded_words_before:,} -> "
            f"{self.downloaded_words_after:,} words (cap {self.downloaded_cap_words:,} words)"
        )
        for s in self.sources:
            if s.action == "trim":
                lines.append(
                    f"  {s.source_id}: {s.words_before:,} -> {s.words_after:,} words, "
                    f"dropping {len(s.drop_shards)} of {len(s.keep_shards) + len(s.drop_shards)} shard files"
                )
            else:
                lines.append(f"  {s.source_id}: {s.words_before:,} words, kept as-is (already under its share)")
        lines.append("")
        lines.append(f"KEEP UNTOUCHED: {config.lang}/data/raw/manual/  (100%, never trimmed or deleted)")
        return "\n".join(lines)


def plan(config: LanguageConfig) -> CleanPlan:
    interim_bytes = {"interim": dir_size(config.interim_dir)}
    final_bytes = {"final": dir_size(config.final_dir)}
    tokenizer_bytes = {"tokenizer": dir_size(config.tokenizer_dir)}

    downloaded_root = config.raw_dir / "downloaded"
    cap_tokens = int(
        config.targets.get(
            "downloaded_tokens_cap",
            round(config.targets.get("final_tokens", 500_000_000) * (1 - config.manual_floor)),
        )
    )
    cap_words = int(cap_tokens / tokens_per_word(config))

    raw_sources: list[dict[str, Any]] = []
    total_words = 0
    if downloaded_root.exists():
        for source_dir in sorted(p for p in downloaded_root.iterdir() if p.is_dir()):
            manifest_path = source_dir / "_manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text("utf-8"))
            words = int(manifest.get("words_estimate", 0))
            raw_sources.append({"dir": source_dir, "manifest": manifest, "words": words})
            total_words += words

    source_plans: list[SourceTrimPlan] = []
    words_after_total = 0
    for entry in raw_sources:
        source_dir: Path = entry["dir"]
        manifest = entry["manifest"]
        source_id = manifest.get("source_id", source_dir.name)
        words_before = entry["words"]
        budget = int(words_before / total_words * cap_words) if total_words else 0

        shards = sorted(source_dir.glob("part-*.jsonl"))
        if total_words <= cap_words:
            # Whole lane already fits -- nothing to trim anywhere.
            source_plans.append(SourceTrimPlan(
                source_id, source_dir, "keep_all", words_before, words_before,
                manifest.get("docs", 0), manifest.get("chars", 0), keep_shards=shards,
            ))
            words_after_total += words_before
            continue

        keep, drop = [], []
        running_words, running_docs, running_chars = 0, 0, 0
        for shard in shards:
            if running_words >= budget and keep:
                drop.append(shard)
                continue
            shard_words = shard_docs = shard_chars = 0
            with shard.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    shard_words += rec.get("n_words", 0)
                    shard_docs += 1
                    shard_chars += rec.get("n_chars", 0)
            if running_words + shard_words > budget and keep:
                drop.append(shard)
                continue
            keep.append(shard)
            running_words += shard_words
            running_docs += shard_docs
            running_chars += shard_chars

        action = "trim" if drop else "keep_all"
        source_plans.append(SourceTrimPlan(
            source_id, source_dir, action, words_before, running_words,
            running_docs, running_chars, keep_shards=keep, drop_shards=drop,
        ))
        words_after_total += running_words

    return CleanPlan(
        interim_bytes=interim_bytes, final_bytes=final_bytes, tokenizer_bytes=tokenizer_bytes,
        downloaded_cap_words=cap_words, downloaded_words_before=total_words,
        downloaded_words_after=words_after_total, sources=source_plans,
    )


def apply(config: LanguageConfig, clean_plan: CleanPlan) -> None:
    """Execute a plan from :func:`plan`. Never touches raw/manual."""
    for directory in (config.interim_dir, config.final_dir, config.tokenizer_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    for s in clean_plan.sources:
        if s.action != "trim":
            continue
        for shard in s.drop_shards:
            shard.unlink(missing_ok=True)
        manifest_path = s.source_dir / "_manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}
        manifest.update({
            "docs": s.docs_after,
            "words_estimate": s.words_after,
            "chars": s.chars_after,
            "bytes": sum(p.stat().st_size for p in s.keep_shards),
            "shards": len(s.keep_shards),
            "stop_reason": "trimmed_to_downloaded_cap",
        })
        atomic_write_json(manifest_path, manifest)

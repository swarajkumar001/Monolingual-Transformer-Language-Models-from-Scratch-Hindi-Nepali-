"""
Configuration loading.

All language-specific behaviour lives in ``<lang>/configs/*.yaml`` and nothing
else. The pipeline code in ``corpuslib`` is byte-identical between the Hindi and
Nepali trees -- a property the QA gate verifies by hashing both copies -- so the
two pipelines remain genuinely independent (separate data, separate configs,
separate outputs) without the code drift that the previous per-language script
copies developed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .sources import SourceRegistry, SourceSpec


def _language_root(start: Path | None = None) -> Path:
    """Locate the ``<lang>/`` directory containing this pipeline copy."""
    here = (start or Path(__file__)).resolve()
    for parent in here.parents:
        if (parent / "configs").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError(f"could not locate the language root above {here}")


@dataclass(slots=True)
class LanguageConfig:
    """Everything one language's pipeline needs to know about itself."""

    lang: str
    lang_code: str
    model_role: str                 # "H" (higher-resource) or "L" (lower-resource)
    display_name: str
    root: Path
    raw: dict[str, Any] = field(default_factory=dict)

    # -- derived paths -----------------------------------------------------

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def final_dir(self) -> Path:
        return self.data_dir / "final"

    @property
    def tokenizer_dir(self) -> Path:
        return self.root / "tokenizer"

    @property
    def events_path(self) -> Path:
        return self.interim_dir / "run_events.jsonl"

    @property
    def status_path(self) -> Path:
        return self.data_dir / "collection_status.json"

    # -- convenience accessors --------------------------------------------

    def section(self, name: str) -> dict[str, Any]:
        return self.raw.get(name, {}) or {}

    @property
    def targets(self) -> dict[str, Any]:
        return self.section("targets")

    @property
    def normalization(self) -> dict[str, Any]:
        return self.section("normalization")

    @property
    def langid(self) -> dict[str, Any]:
        return self.section("langid")

    @property
    def manual_tokens_target(self) -> int:
        """Minimum final manual-lane tokens this corpus must reach.

        An absolute floor, not a ratio -- collection and reporting are both
        judged against this number directly, not against what share of the
        (variable) total it happens to work out to.
        """
        return int(self.targets.get("manual_tokens_target", 100_000_000))

    @property
    def downloaded_tokens_target(self) -> int:
        """Downloaded-lane tokens this corpus should hold -- s10 trims down to
        approximately this many, and rawtrim keeps raw disk usage in line with
        it; manual is never touched by either."""
        return int(self.targets.get("downloaded_tokens_target", 400_000_000))

    def ensure_dirs(self) -> None:
        for directory in (
            self.raw_dir / "manual",
            self.raw_dir / "downloaded",
            self.interim_dir,
            self.final_dir,
            self.tokenizer_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def load_language_config(root: Path | None = None) -> LanguageConfig:
    """Read ``configs/language.yaml`` for the language this copy belongs to."""
    root = Path(root) if root else _language_root()
    payload = yaml.safe_load((root / "configs" / "language.yaml").read_text("utf-8"))
    return LanguageConfig(
        lang=payload["lang"],
        lang_code=payload["lang_code"],
        model_role=payload.get("model_role", "H"),
        display_name=payload.get("display_name", payload["lang"].title()),
        root=root,
        raw=payload,
    )


def load_sources(root: Path | None = None) -> SourceRegistry:
    """Read ``configs/sources.yaml`` into a validated :class:`SourceRegistry`.

    Construction fails loudly if two enabled sources claim the same content
    family in different lanes -- the defect that made the previous run's manual
    percentage meaningless.
    """
    root = Path(root) if root else _language_root()
    payload = yaml.safe_load((root / "configs" / "sources.yaml").read_text("utf-8"))
    defaults = payload.get("defaults", {}) or {}

    specs: list[SourceSpec] = []
    for entry in payload.get("sources", []):
        params = {**(defaults.get("params", {}) or {}), **(entry.get("params", {}) or {})}
        specs.append(
            SourceSpec(
                source_id=entry["source_id"],
                source_name=entry["source_name"],
                lane=entry["lane"],
                collection_method=entry["collection_method"],
                enabled=bool(entry.get("enabled", True)),
                target_words=int(entry.get("target_words", 0)),
                max_seconds=int(entry.get("max_seconds", defaults.get("max_seconds", 3600))),
                priority=int(entry.get("priority", 50)),
                params=params,
            )
        )
    return SourceRegistry(specs)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text("utf-8")) or {}


def tokens_per_word(config: LanguageConfig) -> float:
    """Prefer S02's measured factor (``configs/token_calibration.json``);
    fall back to the config placeholder (``targets.assumed_tokens_per_word``)
    if S02 has not run yet. Shared by every step that converts a token
    budget into a word budget before a tokenizer exists (s10_balance_lanes,
    rawtrim) -- one place, so they cannot disagree.
    """
    calib_path = config.configs_dir / "token_calibration.json"
    if calib_path.exists():
        try:
            measured = json.loads(calib_path.read_text("utf-8")).get("tokens_per_word")
            if measured:
                return float(measured)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return float(config.targets.get("assumed_tokens_per_word", 1.75))

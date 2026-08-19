#!/usr/bin/env python3
"""
Step 00 -- environment and source self-test. Run this before anything expensive.

    python hindi/scripts/s00_doctor.py

This step exists because of one specific failure. The ``fasttext`` PyPI wheel
calls ``np.array(probs, copy=False)``, which raises on NumPy >= 2. The previous
pipeline caught that exception with ``except Exception: dropped += 1`` and
deleted 2,275,943 Hindi and 1,710,844 Nepali documents while logging
``SUCCESS``. The defect was a one-line environment incompatibility that took
four hours of pipeline time to surface and was then misread as a data problem.

Every check below is cheap, and each one maps to a way the run can silently
produce a wrong corpus. A check that fails prints the exact command that fixes
it. Checks are grouped so that a failure in an optional capability (OCR) is
reported as a degraded capability rather than a blocker.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpuslib import runlog  # noqa: E402
from corpuslib.config import load_language_config, load_sources  # noqa: E402
from corpuslib.jsonlio import atomic_write_json, free_disk_gb  # noqa: E402

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""
    critical: bool = True


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name, status, detail, fix="", critical=True) -> None:
        self.checks.append(Check(name, status, detail, fix, critical))

    @property
    def blockers(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL and c.critical]

    @property
    def degraded(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL and not c.critical]

    def to_dict(self) -> dict:
        return {
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "detail": c.detail,
                    "fix": c.fix,
                    "critical": c.critical,
                }
                for c in self.checks
            ],
            "blockers": len(self.blockers),
            "degraded": len(self.degraded),
        }


# --- individual checks ------------------------------------------------------


def check_python_packages(report: Report) -> None:
    required = {
        "datasets": "pip install datasets",
        "huggingface_hub": "pip install huggingface_hub",
        "sentencepiece": "pip install sentencepiece",
        "fasttext": "pip install fasttext-wheel",
        "yaml": "pip install pyyaml",
        "numpy": "pip install numpy",
        "matplotlib": "pip install matplotlib",
    }
    optional = {
        "pytesseract": "pip install pytesseract",
        "pymupdf": "pip install pymupdf",
        "PIL": "pip install pillow",
    }
    for module, fix in required.items():
        found = importlib.util.find_spec(module) is not None
        report.add(
            f"python:{module}",
            PASS if found else FAIL,
            "importable" if found else "missing",
            fix,
            critical=True,
        )
    for module, fix in optional.items():
        found = importlib.util.find_spec(module) is not None
        report.add(
            f"python:{module}",
            PASS if found else FAIL,
            "importable" if found else "missing (OCR source will be skipped)",
            fix,
            critical=False,
        )


def check_language_id(report: Report, config) -> None:
    """The check that would have prevented the previous run's corpus loss."""
    from corpuslib.langid import LanguageIdentifier, LidUnavailable

    target = config.langid.get("target_label", config.lang_code)
    try:
        started = time.monotonic()
        identifier = LanguageIdentifier(
            target,
            min_confidence=config.langid.get("min_confidence", 0.7),
            min_margin=config.langid.get("min_margin", 0.2),
        )
        elapsed = time.monotonic() - started
    except LidUnavailable as exc:
        report.add(
            "langid:self_test",
            FAIL,
            str(exc),
            "check lid.176.bin exists at the repo root and fasttext imports; "
            "the NumPy>=2 wrapper bug is worked around in corpuslib.langid",
            critical=True,
        )
        return

    report.add(
        "langid:self_test",
        PASS,
        f"fastText lid.176 loaded and classified hi/ne/en correctly in {elapsed:.1f}s",
        critical=True,
    )

    # Verify the target language is actually detected on real text, and that
    # the neighbouring-Devanagari margin behaves.
    probes = {
        "hi": "भारत एक विशाल देश है जहाँ अनेक भाषाएँ बोली जाती हैं और संस्कृति विविध है। "
              "यहाँ की परंपराएँ सदियों पुरानी हैं और हर क्षेत्र की अपनी पहचान है।",
        "ne": "नेपाल एक सुन्दर हिमाली देश हो जहाँ सगरमाथा अवस्थित छ र यहाँको संस्कृति धनी छ। "
              "काठमाडौं उपत्यकामा धेरै ऐतिहासिक सम्पदाहरू रहेका छन्।",
    }
    verdict = identifier.classify(probes[target])
    report.add(
        f"langid:accepts_{target}",
        PASS if verdict.accepted else FAIL,
        f"conf={verdict.confidence:.3f} margin={verdict.margin:.3f} "
        f"runner_up={verdict.runner_up or 'none'} reason={verdict.reason}",
        "loosen langid.min_confidence / min_margin in configs/language.yaml",
        critical=True,
    )

    other = "ne" if target == "hi" else "hi"
    rejected = identifier.classify(probes[other])
    report.add(
        f"langid:rejects_{other}",
        PASS if not rejected.accepted else WARN,
        f"{other} text -> accepted={rejected.accepted} reason={rejected.reason}",
        "tighten langid.min_margin if the neighbouring language is being accepted",
        critical=False,
    )


def check_tokenizer(report: Report) -> None:
    """Train a throwaway model so a SentencePiece problem surfaces now, not on day 3."""
    try:
        import sentencepiece as spm
    except ImportError:
        report.add("sentencepiece:train", FAIL, "not installed", "pip install sentencepiece")
        return

    import tempfile

    sample = (
        "भारत एक विशाल देश है। यहाँ अनेक भाषाएँ बोली जाती हैं।\n"
        "नेपाल हिमालको काखमा अवस्थित छ। यहाँको प्राकृतिक सौन्दर्य अतुलनीय छ।\n"
    ) * 200
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "sample.txt"
        corpus.write_text(sample, encoding="utf-8")
        try:
            spm.SentencePieceTrainer.train(
                input=str(corpus),
                model_prefix=str(Path(tmp) / "probe"),
                vocab_size=200,
                model_type="bpe",
                character_coverage=0.9995,
                minloglevel=2,
            )
            processor = spm.SentencePieceProcessor(model_file=str(Path(tmp) / "probe.model"))
            pieces = processor.encode("भारत एक विशाल देश है।", out_type=str)
            report.add(
                "sentencepiece:train",
                PASS,
                f"trained a 200-token probe model; sample encode -> {pieces[:6]}",
            )
        except Exception as exc:  # noqa: BLE001
            report.add(
                "sentencepiece:train", FAIL, f"{type(exc).__name__}: {exc}", "reinstall sentencepiece"
            )


def check_ocr(report: Report, config) -> None:
    binary = shutil.which("tesseract")
    if not binary:
        report.add(
            "ocr:tesseract",
            FAIL,
            "tesseract not on PATH",
            "brew install tesseract tesseract-lang",
            critical=False,
        )
        return
    try:
        langs = subprocess.run(
            ["tesseract", "--list-langs"], capture_output=True, text=True, timeout=30, check=False
        ).stdout.split()
    except Exception as exc:  # noqa: BLE001
        report.add("ocr:tesseract", FAIL, f"{type(exc).__name__}: {exc}", "", critical=False)
        return

    wanted = "hin" if config.lang == "hindi" else "nep"
    have = wanted in langs
    report.add(
        f"ocr:lang_{wanted}",
        PASS if have else FAIL,
        f"tesseract at {binary}, {len(langs)} language packs, {wanted}={'yes' if have else 'no'}",
        "brew install tesseract-lang",
        critical=False,
    )


def check_sources(report: Report, session, registry) -> None:
    """Confirm each enabled source is actually reachable before committing a night to it."""
    from corpuslib.sources import build_collector

    for spec in registry.enabled():
        collector = build_collector(
            spec,
            lang="probe",
            lang_code="xx",
            out_root=Path("/tmp/doctor-probe"),
            session=session,
        )
        started = time.monotonic()
        try:
            reason = collector.preflight()
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - started
        report.add(
            f"source:{spec.source_id}",
            PASS if reason is None else FAIL,
            (f"reachable ({elapsed:.1f}s)" if reason is None else reason)
            + f" · lane={spec.lane} target={spec.target_words:,}w",
            "disable in configs/sources.yaml if permanently unavailable",
            critical=False,
        )


def check_disk(report: Report, config) -> None:
    free = free_disk_gb(config.root)
    needed = 45.0
    report.add(
        "disk:free",
        PASS if free >= needed else FAIL,
        f"{free:.1f}GB free (need ~{needed:.0f}GB for both languages)",
        "free disk space before starting the overnight run",
        critical=free < needed,
    )


def check_lane_invariant(report: Report, registry) -> None:
    """The registry validates on construction; report it explicitly for the log."""
    manual = registry.by_lane("manual")
    downloaded = registry.by_lane("downloaded")
    families_manual = {s.params.get("content_family") for s in manual if s.enabled}
    families_downloaded = {s.params.get("content_family") for s in downloaded if s.enabled}
    overlap = families_manual & families_downloaded
    report.add(
        "invariant:lane_disjoint",
        PASS if not overlap else FAIL,
        (
            f"{len(manual)} manual / {len(downloaded)} downloaded sources; "
            f"no content family spans both lanes"
        )
        if not overlap
        else f"content families in BOTH lanes: {sorted(overlap)}",
        "a content family must belong to exactly one lane in configs/sources.yaml",
        critical=True,
    )

    manual_target = sum(s.target_words for s in manual if s.enabled)
    downloaded_target = sum(s.target_words for s in downloaded if s.enabled)
    total = manual_target + downloaded_target
    planned_ratio = manual_target / total if total else 0.0
    report.add(
        "invariant:planned_manual_ratio",
        PASS if planned_ratio >= 0.20 else WARN,
        f"planned manual share {planned_ratio:.1%} "
        f"({manual_target:,} of {total:,} target words)",
        "raise manual target_words, or add/deepen manual sources -- "
        "s10_balance_lanes caps the downloaded lane at a fixed token budget "
        "regardless, it will not force this ratio by discarding more downloaded text",
        critical=False,
    )


# --- driver -----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Environment and source self-test.")
    parser.add_argument(
        "--skip-network", action="store_true", help="skip source reachability probes"
    )
    args = parser.parse_args(argv)

    config = load_language_config()
    config.ensure_dirs()
    registry = load_sources()
    report = Report()

    print("=" * 78)
    print(f"DOCTOR — {config.display_name} (Model {config.model_role})")
    print("=" * 78, flush=True)

    check_python_packages(report)
    check_disk(report, config)
    check_lane_invariant(report, registry)
    check_language_id(report, config)
    check_tokenizer(report)
    check_ocr(report, config)

    if not args.skip_network:
        from corpuslib.nethttp import PoliteSession

        check_sources(report, PoliteSession(), registry)

    width = max(len(c.name) for c in report.checks) + 2
    for check in report.checks:
        marker = {PASS: "  ok ", WARN: " warn", FAIL: " FAIL"}[check.status]
        print(f"[{marker}] {check.name:<{width}} {check.detail}")
        if check.status == FAIL and check.fix:
            print(f"{'':<{width + 9}}fix: {check.fix}")

    atomic_write_json(config.interim_dir / "doctor_report.json", report.to_dict())

    blockers, degraded = report.blockers, report.degraded
    print()
    if blockers:
        print(f"BLOCKED — {len(blockers)} critical check(s) failed:")
        for check in blockers:
            print(f"  · {check.name}: {check.detail}")
        status, message = runlog.BLOCKED, f"{len(blockers)} critical checks failed"
    elif degraded:
        print(f"READY (degraded) — {len(degraded)} optional capability unavailable:")
        for check in degraded:
            print(f"  · {check.name}: {check.detail}")
        status, message = runlog.WARNING, f"{len(degraded)} optional capabilities unavailable"
    else:
        print("READY — all checks passed.")
        status, message = runlog.SUCCESS, "all checks passed"

    runlog.log_step(
        "S00",
        f"Doctor ({config.display_name})",
        status,
        f"Environment self-test: {message}.",
        metrics={
            "checks": len(report.checks),
            "passed": sum(1 for c in report.checks if c.status == PASS),
            "failed": sum(1 for c in report.checks if c.status == FAIL),
            "blockers": len(blockers),
        },
        lang=config.lang,
        events_path=config.events_path,
    )
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())

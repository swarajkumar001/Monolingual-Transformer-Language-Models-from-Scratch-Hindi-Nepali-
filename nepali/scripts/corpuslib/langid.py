"""
Language identification, with the failure that destroyed the previous run
made structurally impossible.

What went wrong before
----------------------
The ``fasttext`` PyPI wheel calls ``np.array(probs, copy=False)``, which raises
``ValueError`` on NumPy >= 2. Every ``model.predict()`` call therefore threw,
and the caller swallowed it with ``except Exception: dropped += 1``. The result:
2,275,943 Hindi and 1,710,844 Nepali documents were deleted by a library
incompatibility, the stage logged ``SUCCESS``, and the surviving 5.85% was
exactly the one source that had been configured to skip LID entirely.

Three defences are built in here
--------------------------------
1. :class:`LanguageIdentifier` calls ``model.f.predict`` -- the raw binding --
   and assembles the result itself, bypassing the broken NumPy wrapper.
2. :meth:`LanguageIdentifier.self_test` runs known Hindi, Nepali and English
   strings through the model at construction time and raises if the labels come
   back wrong. A broken model cannot reach the corpus.
3. :meth:`classify_batch` counts errors separately from rejections and raises
   ``LidErrorRate`` once the error rate exceeds a threshold. Failures are never
   accounted as "this document is not in our language".

Devanagari neighbours
---------------------
Hindi, Nepali, Maithili, Bhojpuri, Doteli and Sanskrit share the script, and
fastText confuses them. Measured on the probe strings: a clean Nepali sentence
scores ``ne 0.914 / hi 0.078``, but a Nepali municipal sentence scores
``ne 0.891 / dty 0.064``. The margin between the top label and the best
competing Devanagari label is therefore a better signal than raw confidence,
and that is what :class:`LanguageIdentifier` reports.
"""

from __future__ import annotations

import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .textnorm import devanagari_ratio, lid_ready

LID_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
LID_MODEL_NAME = "lid.176.bin"

# Languages written in Devanagari that fastText can emit and that we must treat
# as *plausible confusions* rather than as unrelated languages.
DEVANAGARI_NEIGHBOURS = frozenset(
    {"hi", "ne", "mai", "bho", "mr", "sa", "dty", "new", "gom", "awa", "mag", "hne", "bh"}
)

# Reference strings with unambiguous ground truth, used for the startup self-test.
SELF_TEST_CASES = (
    ("hi", "भारत एक विशाल देश है जहाँ अनेक भाषाएँ बोली जाती हैं और संस्कृति विविध है।"),
    ("ne", "नेपाल एक सुन्दर हिमाली देश हो जहाँ सगरमाथा अवस्थित छ र यहाँको संस्कृति धनी छ।"),
    ("en", "The quick brown fox jumps over the lazy dog and runs far away today."),
)


class LidUnavailable(RuntimeError):
    """The LID model could not be loaded. Callers must halt, not degrade."""


class LidErrorRate(RuntimeError):
    """Too many documents failed to score -- the model, not the data, is wrong."""


@dataclass(slots=True)
class LidVerdict:
    """One document's language decision, with the evidence behind it."""

    label: str
    confidence: float
    margin: float          # top confidence minus best competing Devanagari label
    runner_up: str
    accepted: bool
    reason: str

    def as_meta(self) -> dict[str, float | str]:
        return {
            "lid_label": self.label,
            "lid_confidence": round(self.confidence, 4),
            "lid_margin": round(self.margin, 4),
            "lid_runner_up": self.runner_up,
            "lid_reason": self.reason,
        }


class LanguageIdentifier:
    """fastText ``lid.176`` wrapper that fails loudly instead of silently.

    Parameters
    ----------
    target_label:
        The ISO code this corpus accepts, e.g. ``"hi"`` or ``"ne"``.
    min_confidence:
        Documents scoring the target label above this are accepted outright.
    min_margin:
        Minimum gap between the target label's score and the best competing
        Devanagari-script label. Guards against Hindi/Nepali/Maithili bleed,
        which raw confidence alone does not catch.
    min_chars:
        Documents shorter than this are never scored -- fastText is unreliable
        on fragments (a bare ``नमस्ते`` classifies as Marathi at 0.83).
    min_devanagari_ratio:
        Script gate applied before the model. Cheap, deterministic, and it
        removes Latin-script boilerplate that would otherwise consume LID time.
    """

    def __init__(
        self,
        target_label: str,
        *,
        model_path: Path | None = None,
        min_confidence: float = 0.65,
        min_margin: float = 0.15,
        min_chars: int = 120,
        min_devanagari_ratio: float = 0.55,
        max_error_rate: float = 0.01,
    ) -> None:
        self.target_label = target_label
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.min_chars = min_chars
        self.min_devanagari_ratio = min_devanagari_ratio
        self.max_error_rate = max_error_rate

        self._model = self._load(model_path)
        self.n_scored = 0
        self.n_errors = 0
        self.self_test()

    # -- construction ------------------------------------------------------

    @staticmethod
    def _resolve_model_path(model_path: Path | None) -> Path:
        if model_path is not None:
            return Path(model_path)
        # Walk up from this file looking for the model at the repository root.
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / LID_MODEL_NAME
            if candidate.exists():
                return candidate
        return here.parents[3] / LID_MODEL_NAME

    def _load(self, model_path: Path | None):
        path = self._resolve_model_path(model_path)
        if not path.exists():
            print(f"[langid] downloading {LID_MODEL_URL} -> {path}", file=sys.stderr)
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(LID_MODEL_URL, str(path))
            except Exception as exc:  # noqa: BLE001 - re-raised as a typed error
                raise LidUnavailable(f"could not download lid.176.bin: {exc}") from exc
        try:
            import fasttext
        except ImportError as exc:
            raise LidUnavailable("fasttext is not installed") from exc
        try:
            return fasttext.load_model(str(path))
        except Exception as exc:  # noqa: BLE001
            raise LidUnavailable(f"could not load {path}: {exc}") from exc

    def self_test(self) -> None:
        """Verify the model returns correct labels on known text.

        This is the check that would have caught the NumPy incompatibility in
        under a second instead of after a four-hour pipeline run.
        """
        for expected, sample in SELF_TEST_CASES:
            try:
                predictions = self._raw_predict(lid_ready(sample), k=3)
            except Exception as exc:  # noqa: BLE001
                raise LidUnavailable(
                    f"LID self-test raised on a known {expected!r} sample: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not predictions:
                raise LidUnavailable(f"LID self-test returned nothing for {expected!r}")
            top_label = predictions[0][0]
            if top_label != expected:
                raise LidUnavailable(
                    f"LID self-test failed: expected {expected!r}, got {top_label!r} "
                    f"({predictions[:3]}). The model file is wrong or corrupt."
                )

    # -- scoring -----------------------------------------------------------

    def _raw_predict(self, line: str, k: int = 4) -> list[tuple[str, float]]:
        """Call the fastText C++ binding directly.

        ``model.predict`` wraps the result in ``np.array(..., copy=False)``,
        which NumPy 2 rejects. ``model.f.predict`` returns
        ``[(probability, "__label__xx"), ...]`` and has no NumPy dependency.
        """
        raw = self._model.f.predict(line, k, 0.0, "strict")
        return [(label.replace("__label__", ""), float(prob)) for prob, label in raw]

    def classify(self, text: str) -> LidVerdict:
        """Return the language verdict for one document."""
        if len(text) < self.min_chars:
            return LidVerdict("", 0.0, 0.0, "", False, "too_short")

        ratio = devanagari_ratio(text)
        if ratio < self.min_devanagari_ratio:
            return LidVerdict("", 0.0, 0.0, "", False, f"script_ratio_{ratio:.2f}")

        line = lid_ready(text)
        try:
            predictions = self._raw_predict(line, k=4)
            self.n_scored += 1
        except Exception as exc:  # noqa: BLE001
            self.n_errors += 1
            self._check_error_rate()
            return LidVerdict("", 0.0, 0.0, "", False, f"lid_error:{type(exc).__name__}")

        if not predictions:
            self.n_errors += 1
            self._check_error_rate()
            return LidVerdict("", 0.0, 0.0, "", False, "lid_empty")

        scores = dict(predictions)
        top_label, top_conf = predictions[0]
        target_conf = scores.get(self.target_label, 0.0)

        # Best competing label that is also a Devanagari language: this is the
        # confusion that actually matters, not the score of some Latin language.
        competitors = [
            (lab, conf)
            for lab, conf in predictions
            if lab != self.target_label and lab in DEVANAGARI_NEIGHBOURS
        ]
        runner_up, runner_conf = competitors[0] if competitors else ("", 0.0)
        margin = target_conf - runner_conf

        if top_label != self.target_label:
            return LidVerdict(
                top_label, top_conf, margin, runner_up, False, f"wrong_language:{top_label}"
            )
        if target_conf < self.min_confidence:
            return LidVerdict(
                top_label, target_conf, margin, runner_up, False,
                f"low_confidence:{target_conf:.2f}",
            )
        if margin < self.min_margin:
            return LidVerdict(
                top_label, target_conf, margin, runner_up, False,
                f"narrow_margin_vs_{runner_up}:{margin:.2f}",
            )
        return LidVerdict(top_label, target_conf, margin, runner_up, True, "accepted")

    def _check_error_rate(self) -> None:
        total = self.n_scored + self.n_errors
        if total < 200:
            return
        rate = self.n_errors / total
        if rate > self.max_error_rate:
            raise LidErrorRate(
                f"LID error rate {rate:.1%} exceeds {self.max_error_rate:.1%} "
                f"({self.n_errors} errors in {total} documents). This indicates a "
                f"broken model or environment, not bad data. Halting rather than "
                f"deleting the corpus."
            )

    @property
    def error_rate(self) -> float:
        total = self.n_scored + self.n_errors
        return self.n_errors / total if total else 0.0

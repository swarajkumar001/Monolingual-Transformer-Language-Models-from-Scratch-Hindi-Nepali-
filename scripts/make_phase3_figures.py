"""Regenerate every Phase 3 attention figure locally, with Devanagari labels.

The Kaggle kernels produced these figures with numeric tick labels, because no
Devanagari font is installed in the Kaggle image and the apt package the
notebook tried to install does not exist. A heatmap whose axes read
"0, 2, 4, ..." is far less useful than one whose axes read the actual tokens of
the reasoning question, and the spec's attention section is explicitly about
being able to say *which* tokens a head attends to.

This machine has Nirmala UI, so the figures are redrawn here from the same
checkpoints and the same prompts. Nothing else changes: the attention values,
entropies and distances are recomputed by the same code and should match the
committed JSON exactly.

Usage
-----
    python scripts/make_phase3_figures.py \\
        --hindi-pretrained  <path/to/hindi/ckpt_best.pt> \\
        --hindi-finetuned   <path/to/hindi_reasoning/ckpt_best.pt> \\
        --nepali-pretrained <path/to/nepali/ckpt_best.pt> \\
        --nepali-finetuned  <path/to/nepali_reasoning/ckpt_best.pt>
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPORT_DIR = REPO_ROOT / "report/phase3"
FIG_DIR = REPORT_DIR / "figures"

LANGS = ("hindi", "nepali")


def regenerate(lang: str, pretrained: Path, finetuned: Path) -> dict:
    """Redraw one language's attention figures and recompute its summary.

    Args:
        lang: Language name.
        pretrained: Path to the Phase 2 checkpoint.
        finetuned: Path to the Phase 3 checkpoint.

    Returns:
        The comparison report.

    Raises:
        FileNotFoundError: If either checkpoint is missing.
    """
    for path in (pretrained, finetuned):
        if not path.is_file():
            raise FileNotFoundError(f"[{lang}] checkpoint not found: {path}")
    compare = importlib.import_module(f"{lang}.reasoning.attention_compare")
    return compare.compare(
        pretrained, finetuned, split="test", batch_size=16,
        device="cpu", fig_dir=FIG_DIR,
    )


def main() -> None:
    """Regenerate both models' Phase 3 attention figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    for lang in LANGS:
        parser.add_argument(f"--{lang}-pretrained", required=True)
        parser.add_argument(f"--{lang}-finetuned", required=True)
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    viz = importlib.import_module("hindi.eval.viz")
    font = viz.find_devanagari_font()
    print("Devanagari font:", font)
    if font is None:
        print("  WARNING: no Devanagari font found; labels will be positions "
              "again, which is the thing this script exists to fix")

    for lang in LANGS:
        print(f"[{lang}]")
        report = regenerate(
            lang,
            Path(getattr(args, f"{lang}_pretrained")),
            Path(getattr(args, f"{lang}_finetuned")),
        )
        out = REPORT_DIR / f"{lang}_attention_comparison.json"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        for name, block in report["models"].items():
            print(f"  {name:11s} mean entropy {block['mean_entropy']:.4f}   "
                  f"mean distance {block['mean_distance']:.3f}")
        print(f"  prompt: {report['example_prompts'][0]}")
        print(f"  wrote {out.relative_to(REPO_ROOT)} and "
              f"{len(report['figures'])} figures")


if __name__ == "__main__":
    main()

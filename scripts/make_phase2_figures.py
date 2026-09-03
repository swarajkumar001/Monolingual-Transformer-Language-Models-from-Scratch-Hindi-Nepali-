"""Regenerate every Phase 2 figure locally, for both models together.

The Kaggle runs each produced their own figures, but in separate kernels and
with an older revision of the plotting code. Regenerating here does three
things those could not:

* draws Model H and Model L on the **same axes**, which is what the spec's
  "report every metric for both models side by side" asks for;
* renders Devanagari tick labels, because a Devanagari font is available on
  this machine and was not installed in the Kaggle image - on Kaggle the
  heatmap labels fell back to numeric positions;
* uses one consistent revision of the plotting code for both models.

Usage
-----
    python scripts/make_phase2_figures.py \\
        --hindi-checkpoint <path/to/hindi/ckpt_best.pt> \\
        --nepali-checkpoint <path/to/nepali/ckpt_best.pt>
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

REPORT_DIR = REPO_ROOT / "report/phase2"
FIG_DIR = REPORT_DIR / "figures"

LANGS = ("hindi", "nepali")


def load_eval(lang: str) -> dict | None:
    """Read one language's evaluation JSON if it exists.

    Args:
        lang: Language name.

    Returns:
        The parsed report, or ``None``.
    """
    path = REPORT_DIR / f"{lang}_eval.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def combined_loss_curves(logs: dict[str, Path]) -> None:
    """Plot both models' training and validation loss on shared axes.

    Args:
        logs: Mapping from language to its ``training_log.jsonl``.
    """
    viz = importlib.import_module("hindi.eval.viz")
    present = {k: v for k, v in logs.items() if v.is_file()}
    if not present:
        print("  no training logs found")
        return
    out = viz.plot_loss_curves(present, FIG_DIR / "loss_curves_both.png",
                               x_axis="tokens")
    print("  wrote", out.relative_to(REPO_ROOT))
    out = viz.plot_loss_curves(present, FIG_DIR / "loss_curves_both_by_step.png",
                               x_axis="step")
    print("  wrote", out.relative_to(REPO_ROOT))


def attention_figures(lang: str, checkpoint: Path) -> dict | None:
    """Draw one model's attention heatmaps from its checkpoint.

    Args:
        lang: Language name.
        checkpoint: Path to that language's checkpoint.

    Returns:
        The attention summary dict, or ``None`` if the checkpoint is missing.
    """
    if not checkpoint.is_file():
        print(f"  [{lang}] checkpoint not found: {checkpoint}")
        return None
    ev_mod = importlib.import_module(f"{lang}.eval.evaluate")
    attn_mod = importlib.import_module(f"{lang}.eval.attention")
    viz = importlib.import_module(f"{lang}.eval.viz")

    ev = ev_mod.Evaluator(str(checkpoint),
                          data_dir=REPO_ROOT / f"{lang}/data/packed",
                          device="cpu")
    x, _ = ev.data.fixed_batches("test", 4, 1, seed=11)[0]
    attentions = attn_mod.collect_attention(ev.model, x)
    tokens = [ev.decode([int(t)]) for t in x[0].cpu().numpy()]

    for layer in (0, len(attentions) - 1):
        out = viz.plot_attention_heatmap(
            attentions[layer][0].numpy(), tokens, layer, [0, 1, 2, 3],
            FIG_DIR / f"{lang}_attention_layer{layer}.png", language=lang)
        print("  wrote", out.relative_to(REPO_ROOT))

    summary = attn_mod.summarise_model_attention(ev.model, x)
    summary["uniform_baseline_entropy"] = attn_mod.uniform_baseline_entropy(
        summary["sequence_length"])
    summary["example_sentences"] = [ev.decode(r) for r in x[:2].cpu().numpy()]
    summary["example_tokens"] = tokens[:48]
    (REPORT_DIR / f"{lang}_attention.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    """Regenerate all Phase 2 figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hindi-checkpoint", required=True)
    parser.add_argument("--nepali-checkpoint", required=True)
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    viz = importlib.import_module("hindi.eval.viz")
    print("Devanagari font:", viz.find_devanagari_font())

    print("loss curves:")
    combined_loss_curves({
        lang: REPORT_DIR / f"{lang}_training_log.jsonl" for lang in LANGS
    })

    summaries = {}
    checkpoints = {"hindi": Path(args.hindi_checkpoint),
                   "nepali": Path(args.nepali_checkpoint)}
    for lang in LANGS:
        print(f"attention [{lang}]:")
        summary = attention_figures(lang, checkpoints[lang])
        if summary:
            summaries[lang] = summary

    if summaries:
        out = viz.plot_attention_summary(
            summaries, FIG_DIR / "attention_summary_both.png")
        print("  wrote", out.relative_to(REPO_ROOT))

    reports = {l: r["generation"] for l in LANGS
               if (r := load_eval(l)) is not None}
    if reports:
        out = viz.plot_generation_metrics(
            reports, FIG_DIR / "generation_metrics_both.png")
        print("  wrote", out.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()

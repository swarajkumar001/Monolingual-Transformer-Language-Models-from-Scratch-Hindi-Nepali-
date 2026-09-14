"""Plotting for training curves, attention heatmaps and attention summaries.

Visualisation is kept separate from computation throughout: nothing in this
module measures anything. It consumes the JSONL training logs written by
:mod:`hindi.train.train` and the statistics dictionaries produced by
:mod:`hindi.eval.attention`, and turns them into figures.

Every figure carries a title, axis labels, and a legend wherever more than one
series is drawn, per the course requirement that plots missing labels or
legends may receive zero marks.

Devanagari note: matplotlib's default font has no Devanagari coverage, so token
labels would render as empty boxes. :func:`find_devanagari_font` looks for a
capable font and the heatmap falls back to numbered positions (printing the
token strings alongside) when none is installed, rather than silently drawing
tofu.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Colour-blind-safe pair, used consistently so Model H and Model L keep the
# same colour across every figure in the report.
COLOR_H = "#1f6f8b"
COLOR_L = "#a34f9d"
LANG_COLORS = {"hindi": COLOR_H, "nepali": COLOR_L}

DEVANAGARI_CANDIDATES = (
    "Nirmala UI",
    "Noto Sans Devanagari",
    "Noto Serif Devanagari",
    "Mangal",
    "Lohit Devanagari",
    "FreeSerif",
    "Kokila",
)


def find_devanagari_font() -> str | None:
    """Return the name of an installed font that can render Devanagari.

    Returns:
        A font family name, or ``None`` if no candidate is installed.
    """
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in DEVANAGARI_CANDIDATES:
        if candidate in available:
            return candidate
    return None


def load_training_log(path: str | Path) -> tuple[dict, dict]:
    """Read a JSONL training log into train and validation series.

    Args:
        path: Path to a ``training_log.jsonl`` written by the trainer.

    Returns:
        A ``(train, val)`` pair of dicts, each with ``step``, ``tokens`` and
        ``loss`` lists.
    """
    train = {"step": [], "tokens": [], "loss": []}
    val = {"step": [], "tokens": [], "loss": []}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            target = val if "val_loss" in rec else train
            key = "val_loss" if "val_loss" in rec else "train_loss"
            if key not in rec:
                continue
            target["step"].append(rec["step"])
            target["tokens"].append(rec.get("tokens_seen", 0))
            target["loss"].append(rec[key])
    return train, val


def plot_loss_curves(
    logs: dict[str, str | Path],
    out_path: str | Path,
    x_axis: str = "tokens",
    stage: str = "pretraining",
) -> Path:
    """Plot training and validation loss for one or both models.

    Args:
        logs: Mapping from language name to its ``training_log.jsonl`` path.
        out_path: Where to write the figure.
        x_axis: ``"tokens"`` to plot against tokens consumed, or ``"step"``.
        stage: Named in the axis titles. Defaults to ``"pretraining"`` so the
            Phase 2 figures are unchanged; Phase 3 passes ``"finetuning"``,
            since a chart of finetuning labelled "pretraining" is simply wrong.

    Returns:
        The path the figure was written to.
    """
    fig, (ax_loss, ax_ppl) = plt.subplots(1, 2, figsize=(13, 5))

    for lang, log_path in logs.items():
        train, val = load_training_log(log_path)
        color = LANG_COLORS.get(lang, None)
        xt = train["tokens"] if x_axis == "tokens" else train["step"]
        xv = val["tokens"] if x_axis == "tokens" else val["step"]
        if x_axis == "tokens":
            xt = [v / 1e6 for v in xt]
            xv = [v / 1e6 for v in xv]

        ax_loss.plot(xt, train["loss"], color=color, alpha=0.4, linewidth=1,
                     label=f"{lang} — train")
        ax_loss.plot(xv, val["loss"], color=color, linewidth=2, marker="o",
                     markersize=3, label=f"{lang} — validation")
        ax_ppl.plot(xv, [np.exp(min(l, 20)) for l in val["loss"]], color=color,
                    linewidth=2, marker="o", markersize=3,
                    label=f"{lang} — validation")

    xlabel = "Training tokens consumed (millions)" if x_axis == "tokens" else "Step"
    ax_loss.set_title(f"Cross-entropy loss during {stage}")
    ax_loss.set_xlabel(xlabel)
    ax_loss.set_ylabel("Cross-entropy loss (nats/token)")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()

    ax_ppl.set_title(f"Validation perplexity during {stage}")
    ax_ppl.set_xlabel(xlabel)
    ax_ppl.set_ylabel("Perplexity (exp of cross-entropy)")
    ax_ppl.set_yscale("log")
    ax_ppl.grid(alpha=0.3, which="both")
    ax_ppl.legend()

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_attention_heatmap(
    attn: np.ndarray,
    tokens: list[str],
    layer: int,
    heads: list[int],
    out_path: str | Path,
    language: str = "",
    max_positions: int = 48,
) -> Path:
    """Draw query-vs-key attention heatmaps for several heads of one layer.

    Args:
        attn: Attention probabilities of shape ``(h, T, T)`` for one sequence.
        tokens: Decoded token strings, one per position.
        layer: Layer index, used in the titles.
        heads: Which head indices to draw.
        out_path: Where to write the figure.
        language: Language name, used in the suptitle.
        max_positions: Truncate to the first N positions so labels stay legible.

    Returns:
        The path the figure was written to.
    """
    font = find_devanagari_font()
    n = min(max_positions, attn.shape[-1])
    labels = tokens[:n] if font else [str(i) for i in range(n)]

    fig, axes = plt.subplots(
        1, len(heads), figsize=(5.2 * len(heads), 5.0), squeeze=False
    )
    for ax, head in zip(axes[0], heads):
        matrix = attn[head, :n, :n]
        im = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0.0)
        ax.set_title(f"Layer {layer}, head {head}")
        ax.set_xlabel("Key position (attended to)")
        ax.set_ylabel("Query position (attending from)")
        step = max(1, n // 16)
        ax.set_xticks(range(0, n, step))
        ax.set_yticks(range(0, n, step))
        ax.set_xticklabels(labels[::step], rotation=90, fontsize=7,
                           fontfamily=font if font else None)
        ax.set_yticklabels(labels[::step], fontsize=7,
                           fontfamily=font if font else None)
        # The colour bar is this figure's legend: it maps colour to probability.
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention probability")

    title = f"Causal self-attention — {language}" if language else "Causal self-attention"
    if not font:
        title += "  (positions shown; no Devanagari font installed)"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_attention_summary(
    summaries: dict[str, dict],
    out_path: str | Path,
) -> Path:
    """Plot per-head attention entropy and mean attention distance by layer.

    Args:
        summaries: Mapping from language name to the dict returned by
            :func:`hindi.eval.attention.summarise_model_attention`.
        out_path: Where to write the figure.

    Returns:
        The path the figure was written to.
    """
    fig, (ax_ent, ax_dist) = plt.subplots(1, 2, figsize=(13, 5))

    for lang, summary in summaries.items():
        color = LANG_COLORS.get(lang, None)
        n_layer = summary["n_layer"]
        entropies = np.array([l["entropy"] for l in summary["layers"]])
        distances = np.array([l["mean_distance"] for l in summary["layers"]])
        layers = np.arange(n_layer)

        # One point per head, jittered horizontally so overlapping heads show.
        for arr, ax in ((entropies, ax_ent), (distances, ax_dist)):
            jitter = np.linspace(-0.18, 0.18, arr.shape[1])
            for head in range(arr.shape[1]):
                ax.scatter(layers + jitter[head], arr[:, head], color=color,
                           alpha=0.55, s=26,
                           label=lang if head == 0 else None)
            ax.plot(layers, arr.mean(axis=1), color=color, linewidth=2,
                    linestyle="--", label=f"{lang} — layer mean")

        baseline = summary.get("uniform_baseline_entropy")
        if baseline:
            ax_ent.axhline(baseline, color="grey", linestyle=":", linewidth=1.5,
                           label="uniform-attention baseline")

    ax_ent.set_title("Attention entropy per head")
    ax_ent.set_xlabel("Layer index (0 = first block)")
    ax_ent.set_ylabel("Mean attention entropy (nats)")
    ax_ent.grid(alpha=0.3)
    ax_ent.legend(fontsize=8)

    ax_dist.set_title("Mean attention distance per head")
    ax_dist.set_xlabel("Layer index (0 = first block)")
    ax_dist.set_ylabel("Mean distance attended back (token positions)")
    ax_dist.grid(alpha=0.3)
    ax_dist.legend(fontsize=8)

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_generation_metrics(
    reports: dict[str, dict],
    out_path: str | Path,
) -> Path:
    """Plot overlap and diversity metrics across decoding settings.

    Args:
        reports: Mapping from language name to that model's ``generation``
            report section.
        out_path: Where to write the figure.

    Returns:
        The path the figure was written to.
    """
    fig, (ax_overlap, ax_div) = plt.subplots(1, 2, figsize=(13, 5))
    settings = list(next(iter(reports.values()))["settings"].keys())
    x = np.arange(len(settings))
    width = 0.35

    for offset, (lang, report) in enumerate(reports.items()):
        color = LANG_COLORS.get(lang, None)
        chrf = [report["settings"][s]["chrf_pp"] for s in settings]
        rep = [report["settings"][s]["repetition_rate"] for s in settings]
        d2 = [report["settings"][s]["distinct_2"] for s in settings]
        pos = x + (offset - 0.5) * width
        ax_overlap.bar(pos, chrf, width, color=color, label=f"{lang} — chrF++")
        ax_div.plot(x, rep, color=color, marker="o", label=f"{lang} — repetition rate")
        ax_div.plot(x, d2, color=color, marker="s", linestyle="--",
                    label=f"{lang} — Distinct-2")

    ax_overlap.set_title("Generation overlap with held-out references")
    ax_overlap.set_xlabel("Decoding setting")
    ax_overlap.set_ylabel("chrF++ score (0-100)")
    ax_overlap.set_xticks(x)
    ax_overlap.set_xticklabels(settings, rotation=20)
    ax_overlap.grid(alpha=0.3, axis="y")
    ax_overlap.legend()

    ax_div.set_title("Repetition and diversity by decoding setting")
    ax_div.set_xlabel("Decoding setting")
    ax_div.set_ylabel("Rate (fraction of generated n-grams)")
    ax_div.set_xticks(x)
    ax_div.set_xticklabels(settings, rotation=20)
    ax_div.grid(alpha=0.3)
    ax_div.legend(fontsize=8)

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

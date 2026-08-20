"""Generate the Nepali (Model L) Phase 1 figures.

Unlike report/phase1/make_plots.py (hindi/bhojpuri), this does NOT load any
SentencePiece model or re-measure anything against the ~9.2GB of final
train/val/test JSONL. Nepali's pipeline was executed on a separate machine
and only its output artifacts were transferred into this repo, so every
number plotted here is a direct read of one of those already-computed files:

  - nepali/data/final/corpus_stats.json (retention funnel, dedup,
    decontamination, lane balancing, final token totals)
  - nepali/data/final/split_stats.json (split sizes, leakage check)
  - nepali/tokenizer/selection_report.json (BPE vocab sweep: fertility,
    chars/token, unk_rate per candidate vocab size)
  - nepali/tokenizer/evaluation_report.json (test-split fertility, and the
    top-50 token frequency list used for figure 8 - NOT the full
    piece-frequency distribution, since that would require re-encoding the
    full validation split; labelled accordingly)
  - log.md's S04/S05 entries (language-ID reject reasons, heuristic-filter
    drop reasons) - transcribed as recorded, not re-derived

Run from the repo root:
    .venv/Scripts/python.exe report/phase1/make_plots_nepali.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
NEPALI_ROOT = REPO_ROOT / "nepali"
FIGURES_DIR = Path(__file__).resolve().parent / "figures" / "nepali"
METRICS_OUT = Path(__file__).resolve().parent / "measured_tokenizer_metrics_nepali.json"

COLOR = "#7a4fa3"
COLOR2 = "#c7562b"

# S04 language-ID reject reasons and S05 heuristic-filter drop reasons, as
# recorded in log.md's [2026-08-19 11:36:58] S04 and [2026-08-19 11:59:10] S05
# entries (script-count reasons only; single-digit long-tail script_ratio_*
# buckets folded into "other script-ratio (<=5 docs each)" for legibility).
S04_REJECT_REASONS = {
    "wrong_language": 1599, "low_confidence": 1189, "narrow_margin_vs_hi": 600,
    "too_short": 251, "other script-ratio (<=5 docs each)": 121,
}
S05_DROP_REASONS = {
    "repeated_lines": 1858, "too_few_sentences": 1226, "too_short": 382,
    "link_farm": 481, "low_lexical_diversity": 5, "too_few_words": 8,
    "mean_word_too_long": 1,
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_corpus_stats() -> dict:
    return json.loads((NEPALI_ROOT / "data" / "final" / "corpus_stats.json").read_text("utf-8"))


def load_split_stats() -> dict:
    return json.loads((NEPALI_ROOT / "data" / "final" / "split_stats.json").read_text("utf-8"))


def load_selection_report() -> dict:
    return json.loads((NEPALI_ROOT / "tokenizer" / "selection_report.json").read_text("utf-8"))


def load_evaluation_report() -> dict:
    return json.loads((NEPALI_ROOT / "tokenizer" / "evaluation_report.json").read_text("utf-8"))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save(fig, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_retention_funnel(stats: dict, name: str) -> Path:
    """Figure 1: documents remaining after each pipeline stage, S03 (raw
    normalised) through S10 (lane-balanced train pool). S09's docs_out is
    derived (s08 docs_out minus decontamination.contaminated_removed) since
    corpus_stats.json's own funnel entry for it carries null docs_in/out."""
    funnel = {row["stage"]: row for row in stats["retention_funnel"]}
    s08_out = funnel["s08_dedup"]["docs_out"]
    s09_out = s08_out - stats["decontamination"]["contaminated_removed"]
    order = [
        ("S03\nnormalise", funnel["s03_normalized"]["docs_out"]),
        ("S04\nlang ID", funnel["s04_langid"]["docs_out"]),
        ("S07\ntrain pool", funnel["s07_train_pool"]["docs_out"]),
        ("S05\nheuristics", funnel["s05_heuristics"]["docs_out"]),
        ("S06\nperplexity", funnel["s06_perplexity"]["docs_out"]),
        ("S08\ndedup", s08_out),
        ("S09\ndecontam", s09_out),
        ("S10\nbalance", funnel["s10_balanced"]["docs_out"]),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    labels, counts = zip(*order)
    ax.bar(labels, counts, color=COLOR)
    ax.set_xlabel("Pipeline stage")
    ax.set_ylabel("Documents remaining")
    ax.set_title("Nepali: document retention through the pipeline")
    return _save(fig, name)


def plot_manual_share(stats: dict, name: str) -> Path:
    """Figure 2: final manual vs. downloaded token share against the 20% floor."""
    manual = stats["overall_manual_tokens"]
    total = stats["overall_tokens"]
    manual_pct = 100 * manual / total
    downloaded_pct = 100 - manual_pct
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar([0], [manual_pct], label="Manual", color="#2e8b57")
    ax.bar([0], [downloaded_pct], bottom=[manual_pct], label="Downloaded", color="#9aa5ad")
    ax.axhline(20.0, color="#c0392b", linestyle="--", linewidth=1.5, label="20% floor")
    ax.set_xticks([0])
    ax.set_xticklabels(["Nepali (Model L)"])
    ax.set_ylabel("Share of final corpus tokens (%)")
    ax.set_title("Manual vs. downloaded share vs. the 20% floor")
    ax.set_ylim(0, 100)
    ax.legend()
    return _save(fig, name)


def plot_drop_reasons(name: str) -> Path:
    """Figure 3: S04 language-ID rejects and S05 heuristic-filter drops,
    side by side. Values transcribed from log.md's S04/S05 entries."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, title, reasons in (
        (axes[0], "S04 language-ID rejects", S04_REJECT_REASONS),
        (axes[1], "S05 heuristic-filter drops", S05_DROP_REASONS),
    ):
        items = sorted(reasons.items(), key=lambda kv: kv[1])
        labels, counts = zip(*items)
        ax.barh(labels, counts, color=COLOR)
        ax.set_title(title)
        ax.set_xlabel("Documents dropped")
    fig.suptitle("Nepali: drop reasons at S04 and S05")
    fig.tight_layout()
    return _save(fig, name)


def plot_split_sizes(split_stats: dict, name: str) -> Path:
    """Figure 4: train/validation/test document counts (log scale)."""
    splits = split_stats["splits"]
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["train", "val", "test"]
    vals = [splits[s]["docs"] for s in labels]
    ax.bar(labels, vals, color=COLOR)
    ax.set_yscale("log")
    ax.set_ylabel("Documents (log scale)")
    ax.set_title("Nepali: train/validation/test split sizes")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    return _save(fig, name)


def plot_vocab_metric(candidates: list[dict], metric_key: str, ylabel: str, title: str, name: str) -> Path:
    """Shared by figures 5-7 (chars/token, fertility, unk_rate vs vocab size)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    xs = [c["vocab_size"] for c in candidates]
    ys = [c[metric_key] for c in candidates]
    ax.plot(xs, ys, marker="o", color=COLOR)
    ax.set_xlabel("Vocabulary size (BPE)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return _save(fig, name)


def plot_top_token_frequency(eval_report: dict, name: str) -> Path:
    """Figure 8: rank vs. frequency for the top 50 tokens (log-log). NOT the
    full piece-frequency distribution - evaluation_report.json only kept the
    top 50; getting the full distribution would mean re-encoding the entire
    validation/test split with the model, which this script deliberately
    does not do. Labelled as top-50, not a full Zipf curve."""
    freqs = [t["count"] for t in eval_report["top_tokens"]]
    ranks = range(1, len(freqs) + 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(ranks, freqs, marker="o", markersize=3, color=COLOR)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Token rank, top 50 only (log scale)")
    ax.set_ylabel("Frequency in training corpus (log scale)")
    ax.set_title("Nepali: top-50 token frequency, vocab_size=10000")
    return _save(fig, name)


def plot_token_count_vs_target(stats: dict, name: str) -> Path:
    """Figure 9: final token count vs. the assignment's 500M planning target."""
    total = stats["overall_tokens"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar([0], [total / 1e6], color=COLOR)
    ax.axhline(500, color="#c0392b", linestyle="--", linewidth=1.5, label="500M target")
    ax.set_xticks([0])
    ax.set_xticklabels(["Nepali (Model L)"])
    ax.set_ylabel("Total tokens, train+val+test (millions)")
    ax.set_title(f"Token count vs. 500M target ({total/5e6:.1f}% of target)")
    ax.legend()
    return _save(fig, name)


def plot_lane_balancing(stats: dict, name: str) -> Path:
    """Figure 10: manual-share ratio before/after S10's lane balancing,
    unique to Nepali's absolute-target-driven pipeline (auto_grow_and_tokenize.py)."""
    lb = stats["lane_balancing"]
    fig, ax = plt.subplots(figsize=(5, 5))
    xs = ["before S10", "after S10"]
    vals = [lb["ratio_before"] * 100, lb["ratio_after"] * 100]
    ax.bar(xs, vals, color=[ "#9aa5ad", COLOR])
    ax.set_ylabel("Manual share of train pool (%)")
    ax.set_title(f"Lane balancing: dropped {lb['downloaded_docs_dropped']:,} downloaded docs")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom")
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    stats = load_corpus_stats()
    split_stats = load_split_stats()
    selection = load_selection_report()
    evaluation = load_evaluation_report()
    candidates = selection["candidates"]

    figures = [
        ("01_retention_funnel.png", plot_retention_funnel, (stats,)),
        ("02_manual_share.png", plot_manual_share, (stats,)),
        ("03_drop_reasons.png", plot_drop_reasons, ()),
        ("04_split_sizes.png", plot_split_sizes, (split_stats,)),
        ("05_vocab_chars_per_token.png", plot_vocab_metric,
         (candidates, "chars_per_token", "Characters per token", "Vocab sweep: chars/token")),
        ("06_vocab_fertility.png", plot_vocab_metric,
         (candidates, "fertility", "Fertility (tokens/word approx.)", "Vocab sweep: fertility")),
        ("07_vocab_unk_rate.png", plot_vocab_metric,
         (candidates, "unk_rate", "UNK rate", "Vocab sweep: UNK rate")),
        ("08_top_token_frequency.png", plot_top_token_frequency, (evaluation,)),
        ("09_token_count_vs_target.png", plot_token_count_vs_target, (stats,)),
        ("10_lane_balancing.png", plot_lane_balancing, (stats,)),
    ]

    written = []
    for fname, fn, args in figures:
        path = fn(*args, fname)
        written.append(path)

    metrics_dump = {
        "corpus": {
            "overall_tokens": stats["overall_tokens"],
            "overall_manual_tokens": stats["overall_manual_tokens"],
            "overall_downloaded_tokens": stats["overall_downloaded_tokens"],
            "meets_manual_target": stats["meets_manual_target"],
            "meets_downloaded_target": stats["meets_downloaded_target"],
        },
        "splits": {s: split_stats["splits"][s]["docs"] for s in ("train", "val", "test")},
        "tokenizer_selection": {
            "chosen_vocab_size": selection["chosen"]["vocab_size"],
            "chosen_fertility": selection["chosen"]["fertility"],
            "chosen_chars_per_token": selection["chosen"]["chars_per_token"],
            "candidates": candidates,
        },
        "tokenizer_test_eval": {
            "test_fertility": evaluation["test_fertility"],
            "test_unk_rate": evaluation["test_unk_rate"],
            "test_chars_per_token": evaluation["test_chars_per_token"],
        },
    }
    METRICS_OUT.write_text(json.dumps(metrics_dump, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {METRICS_OUT.relative_to(REPO_ROOT)}")
    print(f"Figures written to {FIGURES_DIR.relative_to(REPO_ROOT)}:")
    for p in written:
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()

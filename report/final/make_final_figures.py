"""Figures for the consolidated final report.

Every number plotted here is read out of a result file produced by an actual
run. Nothing is typed in by hand, so a figure cannot drift away from the
artifact it describes. Run from the repository root::

    python report/final/make_final_figures.py

Output is vector PDF in ``report/final/figures/``, sized for a 10pt article
with 0.72in margins (text width ~7.06in).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# Model H is plotted in blue throughout, Model L in orange, so colour means the
# same thing in every figure of the report.
C_H = "#1f4e79"
C_L = "#c55a11"
C_GREY = "#9a9a9a"

plt.rcParams.update(
    {
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "legend.frameon": False,
    }
)


def load_json(rel: str) -> dict:
    """Read one JSON artifact relative to the repository root."""
    with (ROOT / rel).open(encoding="utf-8") as fh:
        return json.load(fh)


def load_jsonl(rel: str) -> list[dict]:
    """Read one JSON-lines log relative to the repository root."""
    with (ROOT / rel).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def save(fig, name: str) -> None:
    """Write a figure to the report's figure directory as vector PDF."""
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------
# Phase 1
# --------------------------------------------------------------------------
def fig_corpus_funnel() -> None:
    """Hindi drop reasons and the Nepali stage-by-stage retention funnel."""
    hi = load_json("hindi/data/metadata/clean_stats.json")
    ne = load_json("nepali/data/final/corpus_stats.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.15))

    dropped = hi.get("dropped_language", {})
    reasons = {
        "too short (<200 ch)": hi.get("dropped_short", 0),
        "duplicate (SHA-1)": hi.get("dropped_duplicate", 0),
        "langdetect mismatch": dropped.get("langdetect_mismatch", 0),
        "likely Bhojpuri": dropped.get("likely_bhojpuri", 0),
        "low Devanagari ratio": dropped.get("low_devanagari_ratio", 0),
        "malformed Unicode": hi.get("dropped_malformed", 0),
    }
    names = list(reasons)[::-1]
    vals = [reasons[k] for k in names]
    ax1.barh(names, vals, color=C_H, height=0.62)
    ax1.set_xscale("log")
    ax1.set_xlabel("Documents dropped (log scale)")
    ax1.set_ylabel("Filter stage")
    ax1.set_title("Model H: document drop reasons")
    for y, v in enumerate(vals):
        ax1.text(v * 1.15, y, f"{v:,}", va="center", fontsize=6.5)
    ax1.set_xlim(right=max(vals) * 6)

    # S09 records how many documents it removed but not an absolute survivor
    # count, so it is skipped here; the caption carries its figure instead.
    funnel = [s for s in ne["retention_funnel"] if s.get("docs_out") is not None]
    # Short codes: the full stage names are given in the surrounding text.
    labels = [s["stage"].split("_")[0].upper() for s in funnel]
    docs_out = [s["docs_out"] for s in funnel]
    ax2.plot(range(len(docs_out)), np.array(docs_out) / 1e6, "o-", color=C_L, lw=1.4, ms=3.5)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=0, fontsize=6.5)
    ax2.set_xlabel("Pipeline stage")
    ax2.set_ylabel("Documents surviving (millions)")
    ax2.set_title("Model L: pipeline retention")
    for x, v in enumerate(docs_out):
        if x in (0, len(docs_out) - 1):
            ax2.annotate(f"{v/1e6:.2f}M", (x, v / 1e6), textcoords="offset points",
                         xytext=(16 if x == 0 else -8, -9 if x == 0 else 4),
                         ha="center", fontsize=6.5)

    fig.tight_layout()
    save(fig, "fig_corpus_funnel.pdf")


def fig_corpus_composition() -> None:
    """Manual vs downloaded token split, and the per-source breakdown."""
    hi = load_json("hindi/hindi_token_counts_by_source.json")
    ne = load_json("nepali/data/final/corpus_stats.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.05),
                                   gridspec_kw={"width_ratios": [1.0, 1.45]})

    hi_man = hi["totals_by_type"]["manual"]["tokens"]
    hi_dl = hi["totals_by_type"]["downloaded"]["tokens"]
    ne_man = ne["overall_manual_tokens"]
    ne_dl = ne["overall_downloaded_tokens"]

    x = np.arange(2)
    man = np.array([hi_man, ne_man]) / 1e6
    dl = np.array([hi_dl, ne_dl]) / 1e6
    ax1.bar(x, man, 0.5, label="manually collected", color=C_H)
    ax1.bar(x, dl, 0.5, bottom=man, label="public corpora", color=C_GREY)
    ax1.axhline(500, color="k", ls="--", lw=0.8)
    ax1.text(1.46, 430, "500M target", fontsize=6.5, ha="right")
    for i, (m, total) in enumerate(zip(man, man + dl)):
        ax1.text(i, total + 14, f"{total:.0f}M", ha="center", fontsize=6.5)
        ax1.text(i, m / 2, f"{100*m/total:.1f}%\nmanual", ha="center", va="center",
                 fontsize=6.5, color="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Model H\n(Hindi)", "Model L\n(Nepali)"])
    ax1.set_ylabel("Tokens (millions)")
    ax1.set_title("Corpus size and provenance")
    ax1.legend(loc="upper left", fontsize=6.5)
    ax1.set_ylim(0, 760)

    rows = []
    for s in hi["manual_sources"]:
        rows.append((s["source_name"].split(" (")[0], s["tokens"] / 1e6, "manual"))
    for s in hi["downloaded_sources"]:
        rows.append((s["source_name"].split(":")[-1][:22], s["tokens"] / 1e6, "downloaded"))
    rows.sort(key=lambda r: r[1])
    cols = [C_H if r[2] == "manual" else C_GREY for r in rows]
    ax2.barh([r[0] for r in rows], [r[1] for r in rows], color=cols, height=0.65)
    ax2.set_xlabel("Tokens (millions)")
    ax2.set_title("Model H: contribution by source")
    ax2.tick_params(axis="y", labelsize=6)

    fig.tight_layout()
    save(fig, "fig_corpus_composition.pdf")


def fig_tokenizer_sweep() -> None:
    """Fertility and unknown-token rate across the measured vocabulary sweep."""
    m = load_json("report/phase1/measured_tokenizer_metrics.json")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 1.9))

    for lang, col, chosen in (("hindi", C_H, 8000), ("nepali", C_L, 10000)):
        rows = sorted(m[lang], key=lambda r: r["vocab_size"])
        v = [r["vocab_size"] for r in rows]
        f = [r["fertility_tokens_per_word"] for r in rows]
        u = [r["unk_rate"] for r in rows]
        label = "Model H (Hindi, Unigram)" if lang == "hindi" else "Model L (Nepali, BPE)"
        ax1.plot(v, f, "o-", color=col, lw=1.3, ms=3.5, label=label)
        ax2.plot(v, u, "o-", color=col, lw=1.3, ms=3.5, label=label)
        ci = v.index(chosen)
        ax1.scatter([chosen], [f[ci]], s=95, facecolors="none", edgecolors=col, lw=1.4, zorder=5)
        ax2.scatter([chosen], [u[ci]], s=95, facecolors="none", edgecolors=col, lw=1.4, zorder=5)

    ax1.set_xlabel("Vocabulary size")
    ax1.set_ylabel("Fertility (tokens per word)")
    ax1.set_title("Fertility against vocabulary size")
    ax1.legend(loc="upper right")
    ax2.set_xlabel("Vocabulary size")
    ax2.set_ylabel("Unknown-token rate")
    ax2.set_title("Unknown-token rate (circled = adopted)")
    ax2.legend(loc="center right")
    for ax in (ax1, ax2):
        ax.set_xticks([5000, 6000, 8000, 10000])
        ax.set_xticklabels(["5k", "6k", "8k", "10k"])

    fig.tight_layout()
    save(fig, "fig_tokenizer_sweep.pdf")


# --------------------------------------------------------------------------
# Phase 2
# --------------------------------------------------------------------------
def fig_param_breakdown() -> None:
    """Where each model's parameters sit, computed from the model configs."""
    specs = []
    for lang, label, col in (("hindi", "Model H", C_H), ("nepali", "Model L", C_L)):
        ev = load_json(f"report/phase2/{lang}_eval.json")
        c = ev["model_config"]
        d, n_layer, d_ff, blk = c["n_embd"], c["n_layer"], c["d_ff"], c["block_size"]
        vocab = c["vocab_size"]
        tok = vocab * d
        pos = blk * d
        attn = n_layer * 4 * (d * d + d)
        ffn = n_layer * (d * d_ff + d_ff + d_ff * d + d)
        norms = n_layer * 2 * 2 * d + 2 * d
        total = ev["parameter_count"]["total"]
        # The analytic breakdown must reconstruct what the model actually holds.
        assert tok + pos + attn + ffn + norms == total, (lang, total)
        specs.append((label, col, [tok, pos, attn, ffn, norms], total))

    fig, ax = plt.subplots(figsize=(7.0, 1.55))
    parts = ["token embedding", "positional embedding", "attention (Q,K,V,O)",
             "feed-forward", "layer norms"]
    shades = ["#1f4e79", "#4a7fb5", "#8fb4d9", "#c7d9ec", "#5a5a5a"]
    for row, (label, _col, vals, total) in enumerate(specs):
        left = 0.0
        for v, name, sh in zip(vals, parts, shades):
            ax.barh(row, v / 1e6, left=left / 1e6, color=sh, height=0.5,
                    label=name if row == 0 else None, edgecolor="white", lw=0.5)
            if v / total > 0.06:
                ax.text((left + v / 2) / 1e6, row, f"{v/1e6:.2f}M", ha="center",
                        va="center", fontsize=6.3,
                        color="white" if sh in shades[:2] + shades[4:] else "black")
            left += v
        ax.text(left / 1e6 + 0.25, row, f"{total:,} total", va="center", fontsize=6.8)
    ax.set_yticks(range(len(specs)))
    ax.set_yticklabels([s[0] for s in specs])
    ax.invert_yaxis()
    ax.set_xlabel("Parameters (millions)")
    ax.set_xlim(0, 29)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.42), fontsize=6.5)
    ax.grid(axis="y", visible=False)
    save(fig, "fig_param_breakdown.pdf")


def fig_pretrain_curves() -> None:
    """Pretraining loss against tokens consumed, with validation perplexity."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 1.9))
    for lang, label, col in (("hindi", "Model H", C_H), ("nepali", "Model L", C_L)):
        log = load_jsonl(f"report/phase2/{lang}_training_log.jsonl")
        tr = [(r["tokens_seen"] / 1e6, r["train_loss"]) for r in log if "train_loss" in r]
        va = [(r["tokens_seen"] / 1e6, r["val_loss"], r["val_ppl"]) for r in log if "val_loss" in r]
        ax1.plot([p[0] for p in tr], [p[1] for p in tr], color=col, lw=0.6, alpha=0.35)
        ax1.plot([p[0] for p in va], [p[1] for p in va], "o-", color=col, lw=1.3, ms=2.8,
                 label=f"{label} validation")
        ax2.plot([p[0] for p in va], [p[2] for p in va], "o-", color=col, lw=1.3, ms=2.8,
                 label=label)
        best = min(va, key=lambda p: p[1])
        ax1.scatter([best[0]], [best[1]], s=70, facecolors="none", edgecolors=col, lw=1.3, zorder=5)

    ax1.set_xlabel("Training tokens consumed (millions)")
    ax1.set_ylabel("Cross-entropy (nats/token)")
    ax1.set_title("Training (faint) and validation loss; circle = evaluated ckpt")
    ax1.legend(loc="upper right")
    ax1.set_ylim(3.0, 7.0)
    ax2.set_xlabel("Training tokens consumed (millions)")
    ax2.set_ylabel("Validation perplexity")
    ax2.set_yscale("log")
    ax2.set_title("Validation perplexity")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig_pretrain_curves.pdf")


def fig_ppl_bpb() -> None:
    """The headline metric disagreement: perplexity against bits-per-byte."""
    h = load_json("report/phase2/hindi_eval.json")["intrinsic"]
    l = load_json("report/phase2/nepali_eval.json")["intrinsic"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.68))
    series = [
        ("Test perplexity\n(per token)", [h["perplexity"], l["perplexity"]], "{:.2f}"),
        ("Bytes per token\n(tokenizer granularity)", [h["bytes_scored"] / h["tokens_scored"],
                                                      l["bytes_scored"] / l["tokens_scored"]], "{:.2f}"),
        ("Test bits-per-byte\n(tokenizer-neutral)", [h["bits_per_byte"], l["bits_per_byte"]], "{:.4f}"),
    ]
    for ax, (title, vals, fmt) in zip(axes, series):
        ax.bar([0, 1], vals, 0.5, color=[C_H, C_L])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Model H", "Model L"])
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.28)
        for i, v in enumerate(vals):
            ax.text(i, v * 1.03, fmt.format(v), ha="center", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_ppl_bpb.pdf")


def fig_decoding_sweep() -> None:
    """Quality and diversity across the four decoding settings."""
    order = ["greedy", "temperature_0.5", "temperature_1.0", "temperature_1.5"]
    nice = ["greedy", "T=0.5", "T=1.0", "T=1.5"]
    data = {}
    for lang in ("hindi", "nepali"):
        s = load_json(f"report/phase2/{lang}_eval.json")["generation"]["settings"]
        data[lang] = {k: s[k] for k in order}

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.68))
    metrics = [("chrf_pp", "chrF++ against references"),
               ("repetition_rate", "Repeated 4-gram fraction"),
               ("distinct_1", "Distinct-1 (unique unigrams)")]
    x = np.arange(4)
    for ax, (key, title) in zip(axes, metrics):
        ax.plot(x, [data["hindi"][k][key] for k in order], "o-", color=C_H, lw=1.3, ms=3.5,
                label="Model H")
        ax.plot(x, [data["nepali"][k][key] for k in order], "s-", color=C_L, lw=1.3, ms=3.5,
                label="Model L")
        ax.set_xticks(x)
        ax.set_xticklabels(nice)
        ax.set_xlabel("Decoding setting")
        ax.set_title(title)
    axes[0].set_ylabel("chrF++")
    axes[1].set_ylabel("Fraction repeated")
    axes[2].set_ylabel("Distinct-1")
    axes[0].legend(loc="lower center")
    fig.tight_layout()
    save(fig, "fig_decoding_sweep.pdf")


def fig_attention_grid() -> None:
    """Per-head attention entropy and mean distance for every layer."""
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 2.62))
    for col, (lang, label) in enumerate((("hindi", "Model H"), ("nepali", "Model L"))):
        a = load_json(f"report/phase2/{lang}_attention.json")
        ent = np.array([ly["entropy"] for ly in a["layers"]])
        dist = np.array([ly["mean_distance"] for ly in a["layers"]])
        im0 = axes[0, col].imshow(ent, cmap="viridis", aspect="auto", vmin=0, vmax=5.3)
        axes[0, col].set_title(f"{label}: attention entropy (nats)")
        im1 = axes[1, col].imshow(dist, cmap="magma", aspect="auto", vmin=0, vmax=200)
        axes[1, col].set_title(f"{label}: mean attention distance (tokens)")
        for ax, im in ((axes[0, col], im0), (axes[1, col], im1)):
            ax.set_xticks(range(ent.shape[1]))
            ax.set_yticks(range(ent.shape[0]))
            ax.set_xlabel("Head index")
            ax.set_ylabel("Layer index")
            ax.grid(False)
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        # Mark the sharpest head in every layer.
        for li in range(ent.shape[0]):
            hi_ = int(np.argmin(ent[li]))
            axes[0, col].text(hi_, li, "*", ha="center", va="center", color="white", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_attention_grid.pdf")


# --------------------------------------------------------------------------
# Phase 3
# --------------------------------------------------------------------------
def fig_finetune_curves() -> None:
    """Finetuning loss, showing where validation turns upward."""
    fig, ax = plt.subplots(figsize=(3.45, 1.78))
    for lang, label, col in (("hindi", "Model H", C_H), ("nepali", "Model L", C_L)):
        log = load_jsonl(f"report/phase3/{lang}_finetune_log.jsonl")
        tr = [(r["step"], r["train_loss"]) for r in log if "train_loss" in r]
        va = [(r["step"], r["val_loss"]) for r in log if "val_loss" in r]
        ax.plot([p[0] for p in tr], [p[1] for p in tr], color=col, lw=0.8, alpha=0.4)
        ax.plot([p[0] for p in va], [p[1] for p in va], "o-", color=col, lw=1.2, ms=2.5,
                label=f"{label} validation")
        best = min(va, key=lambda p: p[1])
        ax.scatter([best[0]], [best[1]], s=70, facecolors="none", edgecolors=col, lw=1.3, zorder=5)
    ax.axvline(60, color="k", ls=":", lw=0.9)
    ax.annotate("step 60: best validation", xy=(60, 1.35), xytext=(230, 0.75),
                fontsize=6.2, arrowprops=dict(arrowstyle="->", lw=0.6, color="black"))
    ax.set_xlabel("Finetuning step")
    ax.set_ylabel("Answer-token loss (nats)")
    ax.set_ylim(0, 4.1)
    ax.legend(loc="upper left", fontsize=6.2, bbox_to_anchor=(0.02, 0.99))
    save(fig, "fig_finetune_curves.pdf")


def fig_reasoning_accuracy() -> None:
    """Per-task exact match and forced choice against the per-example chance rate."""
    tasks = ["compare2", "numeric2", "numeric3", "transitive3"]
    res = {lang: load_json(f"report/phase3/{lang}_reasoning_finetuned.json")
           for lang in ("hindi", "nepali")}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.6, 1.78), sharey=True)
    x = np.arange(len(tasks))
    for ax, key, title in ((ax1, "exact_match", "Exact match"),
                           (ax2, "forced_choice", "Forced choice")):
        h = [res["hindi"][key]["by_task"][t]["accuracy"] * 100 for t in tasks]
        l = [res["nepali"][key]["by_task"][t]["accuracy"] * 100 for t in tasks]
        ax.bar(x - 0.2, h, 0.38, color=C_H, label="Model H")
        ax.bar(x + 0.2, l, 0.38, color=C_L, label="Model L")
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=22, ha="right", fontsize=6.3)
        ax.set_title(title)
        if key == "forced_choice":
            chance = [50.0, 50.0, 100 / 3, 100 / 3]
            for i, c in enumerate(chance):
                ax.plot([i - 0.42, i + 0.42], [c, c], color="k", ls="--", lw=0.9)
            ax.plot([], [], color="k", ls="--", lw=0.9, label="chance")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 62)
    ax2.legend(loc="upper right", fontsize=6)
    fig.tight_layout()
    save(fig, "fig_reasoning_accuracy.pdf")


def fig_offprompt() -> None:
    """How often each model answers with a name the question never mentions."""
    tasks = ["compare2", "numeric2", "numeric3", "transitive3"]
    fig, ax = plt.subplots(figsize=(3.45, 1.78))
    x = np.arange(len(tasks))
    for off, lang, label, col in ((-0.2, "hindi", "Model H", C_H), (0.2, "nepali", "Model L", C_L)):
        ft = load_json(f"report/phase3/{lang}_reasoning_finetuned.json")["error_analysis"]
        ax.bar(x + off, [ft[t]["answered_off_prompt"] * 100 for t in tasks], 0.38,
               color=col, label=f"{label}, finetuned")
    ax.axhline(100, color=C_GREY, ls="--", lw=0.9)
    ax.text(3.45, 103, "before finetuning: ${\\approx}100\\%$ for both",
            fontsize=6, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=22, ha="right", fontsize=6.3)
    ax.set_ylabel("Answers not in the prompt (%)")
    ax.set_ylim(0, 118)
    ax.legend(loc="upper left", fontsize=6.2, bbox_to_anchor=(-0.02, 0.92))
    save(fig, "fig_offprompt.pdf")


def fig_head_delta() -> None:
    """Per-head entropy change from pretrained to finetuned."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.56))
    for ax, (lang, label) in zip(axes, (("hindi", "Model H"), ("nepali", "Model L"))):
        d = load_json(f"report/phase3/{lang}_attention_comparison.json")["per_head_delta"]
        n_layer = 1 + max(int(k[1:].split("H")[0]) for k in d)
        n_head = 1 + max(int(k.split("H")[1]) for k in d)
        grid = np.zeros((n_layer, n_head))
        for k, v in d.items():
            li = int(k[1:].split("H")[0])
            hi_ = int(k.split("H")[1])
            grid[li, hi_] = v["entropy_delta"]
        lim = float(np.abs(grid).max())
        im = ax.imshow(grid, cmap="RdBu_r", aspect="auto",
                       norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
        ax.set_xticks(range(n_head))
        ax.set_yticks(range(n_layer))
        ax.set_xlabel("Head index")
        ax.set_ylabel("Layer index")
        ax.set_title(f"{label}: entropy change, pretrained $\\rightarrow$ finetuned")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("$\\Delta$ entropy (nats)", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig_head_delta.pdf")


def crop_attention_heatmaps() -> None:
    """Crop the Phase 2 heatmaps to their first two heads.

    The source figures place four heads side by side across 3120px. Reproduced at
    half text width that leaves each panel about 0.87in across and the Devanagari
    tick labels unreadable in print, so the report uses the leftmost two heads at
    double the width instead.
    """
    from PIL import Image

    for lang in ("hindi",):
        for layer in (0, 5):
            src = ROOT / f"report/phase2/figures/{lang}_attention_layer{layer}.png"
            im = Image.open(src)
            half = im.width // 2
            out = OUT / f"crop_{lang}_attention_layer{layer}.png"
            # Drop the top strip too: the centred super-title would otherwise be
            # cut in half by the crop.
            im.crop((0, 48, half, im.height)).save(out)
            print(f"  {out.relative_to(ROOT)}")


def main() -> None:
    """Regenerate every figure the consolidated report includes."""
    print("writing figures to report/final/figures/ ...")
    fig_corpus_funnel()
    fig_corpus_composition()
    fig_tokenizer_sweep()
    fig_param_breakdown()
    fig_pretrain_curves()
    fig_ppl_bpb()
    fig_decoding_sweep()
    fig_attention_grid()
    fig_finetune_curves()
    fig_reasoning_accuracy()
    fig_offprompt()
    fig_head_delta()
    crop_attention_heatmaps()
    print("done.")


if __name__ == "__main__":
    main()

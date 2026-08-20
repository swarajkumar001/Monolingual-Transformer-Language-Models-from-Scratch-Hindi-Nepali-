"""Generate every Phase 1 figure for the Hindi (Model H) / Nepali (Model L) report.

Run from the repo root:
    .venv/Scripts/python.exe report/phase1/make_plots.py

Hindi and Nepali are measured by structurally different pipelines (Hindi:
a single-pass cleaner instrumented with clean_stats.json, char-based;
Nepali: a stage-based S00-S14 pipeline instrumented with corpus_stats.json,
doc/word/token-based, no per-lane character tracking). Rather than force
both into one schema, each compute_* function below branches per language
and reads from that language's own authoritative file:
  - Hindi: hindi/data/metadata/clean_stats.json,
    hindi/hindi_pipeline_manifest.json (corrected final counts),
    hindi/tokenizer/hindi_unigram_<vocab>.model (live re-measured)
  - Nepali: nepali/data/final/corpus_stats.json, split_stats.json,
    nepali/tokenizer/sweep/nepali_bpe_bpe_<vocab>.model (live re-measured),
    log.md's S04/S05 entries (drop-reason counts, transcribed as recorded)

Tokenizer sweep metrics (figures 6-9) are measured fresh for BOTH languages
by loading each trained .model file directly and encoding the first 2,000
non-empty documents of that language's own held-out validation split -
same methodology, different source format (Hindi: individual .txt files
under data/splits/validation/text/; Nepali: nepali/data/final/val.jsonl).

Computation functions return plain data (dict/list/tuple); plotting
functions take that data and a Path and only call matplotlib.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sentencepiece as spm

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
METRICS_OUT = Path(__file__).resolve().parent / "measured_tokenizer_metrics.json"

LANGS = ("hindi", "nepali")
LANG_LABEL = {"hindi": "Hindi (Model H)", "nepali": "Nepali (Model L)"}
LANG_COLOR = {"hindi": "#1f6f8b", "nepali": "#7a4fa3"}

VOCAB_SIZES = {"hindi": [5000, 8000, 10000], "nepali": [5000, 6000, 8000, 10000]}
SELECTED_VOCAB = {"hindi": 8000, "nepali": 10000}
MAX_ANALYSIS_DOCS = 2000  # matches hindi/tokenizer/analyze_tokenizer.py's default

# S04 language-ID reject reasons and S05 heuristic-filter drop reasons for
# Nepali, as recorded in log.md's [2026-08-19 11:36:58] S04 and
# [2026-08-19 11:59:10] S05 entries (transcribed, not re-derived; long-tail
# single-digit script_ratio_* buckets folded together for legibility).
NEPALI_S04_REJECT_REASONS = {
    "wrong_language": 1599, "low_confidence": 1189, "narrow_margin_vs_hi": 600,
    "too_short": 251, "other script-ratio (<=5 docs each)": 121,
}
NEPALI_S05_DROP_REASONS = {
    "repeated_lines": 1858, "too_few_sentences": 1226, "too_short": 382,
    "link_farm": 481, "low_lexical_diversity": 5, "too_few_words": 8,
    "mean_word_too_long": 1,
}


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def load_hindi_clean_stats() -> dict:
    path = REPO_ROOT / "hindi" / "data" / "metadata" / "clean_stats.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_hindi_manifest() -> dict:
    """Hindi's clean_stats.json predates the 2-document cross-language
    correction; the manifest holds the corrected final counts."""
    path = REPO_ROOT / "hindi" / "hindi_pipeline_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_nepali_corpus_stats() -> dict:
    path = REPO_ROOT / "nepali" / "data" / "final" / "corpus_stats.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_nepali_split_stats() -> dict:
    path = REPO_ROOT / "nepali" / "data" / "final" / "split_stats.json"
    return json.loads(path.read_text(encoding="utf-8"))


def compute_corpus_size_data() -> dict:
    """WHY: figure 1 needs a raw-vs-final size comparison per language on one
    axis. Hindi tracks characters; Nepali's pipeline never kept a per-lane
    character tally past S03 (only docs/words/tokens). Documents are the one
    unit both pipelines measured at every stage, so figure 1 uses DOCUMENT
    COUNTS for both languages rather than forcing a char estimate for Nepali.
    OUTPUT SHAPE: {lang: {'raw_docs': int, 'final_docs': int}}."""
    hindi_stats = load_hindi_clean_stats()
    hindi_final = load_hindi_manifest()["cleaned"]["documents"]
    nepali_stats = load_nepali_corpus_stats()
    nepali_raw_docs = next(r["docs_in"] for r in nepali_stats["retention_funnel"] if r["stage"] == "s03_normalized")
    nepali_final_docs = sum(nepali_stats["splits"][s]["docs"] for s in ("train", "val", "test"))
    return {
        "hindi": {"raw_docs": hindi_stats["docs_in"], "final_docs": hindi_final},
        "nepali": {"raw_docs": nepali_raw_docs, "final_docs": nepali_final_docs},
    }


def compute_manual_share_data() -> dict:
    """WHY: figure 2 checks each language against the hard 20% manual-token
    floor. Hindi's share is char-weighted (what validate_phase1.py gates on
    for Hindi); Nepali's is token-weighted (corpus_stats.json never tracked
    per-lane characters - see report §2 of the Nepali write-up). Both are
    hard-coded from each language's own independently-verified source rather
    than re-derived here.
    OUTPUT SHAPE: {lang: {'manual': int, 'total': int, 'manual_pct': float, 'weighting': str}}."""
    return {
        "hindi": {
            "manual": 505_509_566, "total": 1_764_596_782,
            "manual_pct": 28.65, "weighting": "characters",
        },
        "nepali": {
            "manual": 167_787_251, "total": 635_204_333,
            "manual_pct": 26.41, "weighting": "tokens",
        },
    }


def compute_cleaning_funnel(lang: str) -> list[tuple[str, int]]:
    """WHY: figure 3 shows how many documents survive each cleaning stage in
    the order the pipeline actually applies them. Hindi and Nepali have
    genuinely different pipelines (single-pass cleaner vs. stage-based
    S00-S14), so their funnels have different stage counts/names - each gets
    its own subplot with its own x-axis, which is fine since figure 3 never
    plots the two languages on a shared axis.
    OUTPUT SHAPE: list of (stage_label, docs_remaining), raw first, final last."""
    if lang == "hindi":
        stats = load_hindi_clean_stats()
        docs = stats["docs_in"]
        funnel = [("raw", docs)]
        docs -= stats["dropped_malformed"]
        funnel.append(("malformed", docs))
        docs -= stats["dropped_short"]
        funnel.append(("short", docs))
        docs -= sum(stats["dropped_language"].values())
        funnel.append(("language filter", docs))
        docs -= stats["dropped_duplicate"]
        funnel.append(("duplicate", docs))
        docs -= 2  # scripts/remove_cross_language_duplicates.py
        funnel.append(("cleaned", docs))
        return funnel
    stats = load_nepali_corpus_stats()
    funnel = {row["stage"]: row for row in stats["retention_funnel"]}
    s08_out = funnel["s08_dedup"]["docs_out"]
    s09_out = s08_out - stats["decontamination"]["contaminated_removed"]
    return [
        ("S03 normalise", funnel["s03_normalized"]["docs_out"]),
        ("S04 lang ID", funnel["s04_langid"]["docs_out"]),
        ("S07 train pool", funnel["s07_train_pool"]["docs_out"]),
        ("S05 heuristics", funnel["s05_heuristics"]["docs_out"]),
        ("S06 perplexity", funnel["s06_perplexity"]["docs_out"]),
        ("S08 dedup", s08_out),
        ("S09 decontam", s09_out),
        ("S10 balance", funnel["s10_balanced"]["docs_out"]),
    ]


def compute_drop_reasons(lang: str) -> dict[str, int]:
    """WHY: figure 4 breaks the funnel's losses down by individual reason
    rather than cumulative stage.
    OUTPUT SHAPE: {reason_label: doc_count}."""
    if lang == "hindi":
        stats = load_hindi_clean_stats()
        reasons = {
            "malformed Unicode": stats["dropped_malformed"],
            "too short (<200 chars)": stats["dropped_short"],
        }
        for reason, count in stats["dropped_language"].items():
            reasons[reason.replace("_", " ")] = count
        reasons["exact/near duplicate"] = stats["dropped_duplicate"]
        reasons["cross-language duplicate"] = 2
        return reasons
    return {**NEPALI_S04_REJECT_REASONS, **NEPALI_S05_DROP_REASONS}


def compute_split_sizes() -> dict:
    """WHY: figure 5 needs train/validation/test document counts per language.
    OUTPUT SHAPE: {lang: {'train': int, 'validation': int, 'test': int}}."""
    nepali_splits = load_nepali_split_stats()["splits"]
    return {
        "hindi": {"train": 675_937, "validation": 14_082, "test": 14_082},
        "nepali": {
            "train": nepali_splits["train"]["docs"],
            "validation": nepali_splits["val"]["docs"],
            "test": nepali_splits["test"]["docs"],
        },
    }


def _load_validation_texts(lang: str, max_docs: int = MAX_ANALYSIS_DOCS) -> list[str]:
    """WHY: every sweep metric and the token-frequency plot must be measured
    on held-out text the tokenizer never trained on.
    WHAT: Hindi - sorted glob over data/splits/validation/text/*.txt, first
    max_docs non-empty files. Nepali has no per-document file layout; reads
    the first max_docs non-empty 'text' fields from data/final/val.jsonl,
    stopping early rather than loading the full ~800MB file.
    OUTPUT SHAPE: list of document text strings, len <= max_docs."""
    if lang == "hindi":
        val_dir = REPO_ROOT / lang / "data" / "splits" / "validation" / "text"
        files = sorted(val_dir.glob("*.txt"))
        texts = []
        for fp in files[:max_docs]:
            text = fp.read_text(encoding="utf-8").strip()
            if text:
                texts.append(text)
        return texts
    val_path = REPO_ROOT / "nepali" / "data" / "final" / "val.jsonl"
    texts = []
    with val_path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(texts) >= max_docs:
                break
            line = line.strip()
            if not line:
                continue
            text = json.loads(line).get("text", "").strip()
            if text:
                texts.append(text)
    return texts


def _model_path(lang: str, vocab_size: int) -> Path:
    if lang == "hindi":
        return REPO_ROOT / "hindi" / "tokenizer" / f"hindi_unigram_{vocab_size}.model"
    return REPO_ROOT / "nepali" / "tokenizer" / "sweep" / f"nepali_bpe_bpe_{vocab_size}.model"


def measure_tokenizer(lang: str, vocab_size: int, texts: list[str]) -> dict:
    """WHY: this is the single fresh measurement every sweep plot (6, 7, 8)
    and the sweep table in the report are built from, for both languages.
    WHAT: encodes every validation text with the SentencePiece model (as ids
    and as piece strings), accumulates chars/tokens/words/UNK/byte-fallback
    counts and a full piece-frequency Counter.
    OUTPUT SHAPE: dict with keys vocab_size, chars_per_token,
    fertility_tokens_per_word, unk_rate, byte_fallback_rate,
    vocab_utilization, used_ids, unused_ids, piece_freq (Counter)."""
    model_path = _model_path(lang, vocab_size)
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    unk_id = sp.unk_id()

    total_chars = total_tokens = total_words = unk_count = byte_fallback_count = 0
    piece_freq: Counter[str] = Counter()
    used_ids: set[int] = set()

    for text in texts:
        ids = sp.encode(text, out_type=int)
        pieces = sp.encode(text, out_type=str)
        total_chars += len(text)
        total_tokens += len(ids)
        total_words += len(text.split())
        for tid, piece in zip(ids, pieces):
            used_ids.add(tid)
            piece_freq[piece] += 1
            if tid == unk_id:
                unk_count += 1
            if piece.startswith("<0x") and piece.endswith(">") and len(piece) == 6:
                byte_fallback_count += 1

    return {
        "vocab_size": vocab_size,
        "chars_per_token": total_chars / total_tokens,
        "fertility_tokens_per_word": total_tokens / total_words,
        "unk_rate": unk_count / total_tokens,
        "byte_fallback_rate": byte_fallback_count / total_tokens,
        "vocab_utilization": len(used_ids) / sp.vocab_size(),
        "used_ids": len(used_ids),
        "unused_ids": sp.vocab_size() - len(used_ids),
        "piece_freq": piece_freq,
    }


def measure_all_sweeps() -> dict:
    """OUTPUT SHAPE: {lang: [dict, ...]}, one dict per vocab size, ascending."""
    results = {}
    for lang in LANGS:
        texts = _load_validation_texts(lang)
        results[lang] = [measure_tokenizer(lang, vs, texts) for vs in VOCAB_SIZES[lang]]
    return results


def compute_token_count_vs_target() -> dict:
    """WHY: figure 10 checks each language's exact, full-corpus token count
    against the 500M-token planning target.
    OUTPUT SHAPE: {lang: {'tokens': int, 'pct_of_target': float}, 'target': int}."""
    return {
        "hindi": {"tokens": 505_852_028, "pct_of_target": 101.2},
        "nepali": {"tokens": 635_204_333, "pct_of_target": 127.0},
        "target": 500_000_000,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save(fig, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_corpus_size(data: dict, name: str) -> Path:
    """Figure 1: raw vs. final documents, both languages, grouped bar."""
    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(LANGS))
    width = 0.35
    raw_vals = [data[l]["raw_docs"] / 1e6 for l in LANGS]
    final_vals = [data[l]["final_docs"] / 1e6 for l in LANGS]
    ax.bar([i - width / 2 for i in x], raw_vals, width, label="Raw", color="#9aa5ad")
    ax.bar([i + width / 2 for i in x], final_vals, width, label="Final (cleaned/split)", color="#1f6f8b")
    ax.set_xticks(list(x))
    ax.set_xticklabels([LANG_LABEL[l] for l in LANGS])
    ax.set_xlabel("Language")
    ax.set_ylabel("Documents (millions)")
    ax.set_title("Corpus size: raw vs. final documents")
    ax.legend()
    return _save(fig, name)


def plot_manual_share(data: dict, name: str) -> Path:
    """Figure 2: manual vs. downloaded share vs. the 20% floor. Hindi is
    char-weighted, Nepali is token-weighted - noted in the axis label rather
    than hidden, since the two pipelines never tracked the same unit."""
    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(LANGS))
    manual_pct = [data[l]["manual_pct"] for l in LANGS]
    downloaded_pct = [100 - m for m in manual_pct]
    ax.bar(x, manual_pct, label="Manual", color="#2e8b57")
    ax.bar(x, downloaded_pct, bottom=manual_pct, label="Downloaded", color="#9aa5ad")
    ax.axhline(20.0, color="#c0392b", linestyle="--", linewidth=1.5, label="20% floor")
    ax.set_xticks(list(x))
    labels = [f"{LANG_LABEL[l]}\n({data[l]['weighting']}-weighted)" for l in LANGS]
    ax.set_xticklabels(labels)
    ax.set_xlabel("Language")
    ax.set_ylabel("Share of final corpus (%)")
    ax.set_title("Manual vs. downloaded share, against the 20% floor")
    ax.set_ylim(0, 100)
    ax.legend()
    return _save(fig, name)


def plot_cleaning_funnel(funnels: dict, name: str) -> Path:
    """Figure 3: documents remaining after each cleaning stage, one subplot
    per language (different pipelines -> different stage counts/names)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, lang in zip(axes, LANGS):
        stages, counts = zip(*funnels[lang])
        ax.bar(stages, counts, color=LANG_COLOR[lang])
        ax.set_title(LANG_LABEL[lang])
        ax.set_xlabel("Cleaning stage")
        ax.set_ylabel("Documents remaining")
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle("Cleaning funnel: documents remaining after each filter stage")
    fig.tight_layout()
    return _save(fig, name)


def plot_drop_reasons(reasons: dict, name: str) -> Path:
    """Figure 4: individual drop reasons ranked by count, one subplot per language."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, lang in zip(axes, LANGS):
        items = sorted(reasons[lang].items(), key=lambda kv: kv[1])
        labels, counts = zip(*items)
        ax.barh(labels, counts, color=LANG_COLOR[lang])
        ax.set_title(LANG_LABEL[lang])
        ax.set_xlabel("Documents dropped")
    fig.suptitle("Drop reasons breakdown")
    fig.tight_layout()
    return _save(fig, name)


def plot_split_sizes(data: dict, name: str) -> Path:
    """Figure 5: train/validation/test split sizes, log y-axis."""
    fig, ax = plt.subplots(figsize=(7, 5))
    splits = ["train", "validation", "test"]
    x = range(len(splits))
    width = 0.35
    for i, lang in enumerate(LANGS):
        offset = (i - 0.5) * width
        vals = [data[lang][s] for s in splits]
        ax.bar([xi + offset for xi in x], vals, width, label=LANG_LABEL[lang], color=LANG_COLOR[lang])
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels([s.capitalize() for s in splits])
    ax.set_xlabel("Split")
    ax.set_ylabel("Documents (log scale)")
    ax.set_title("Train / validation / test split sizes")
    ax.legend()
    return _save(fig, name)


def plot_vocab_metric(sweeps: dict, metric_key: str, ylabel: str, title: str, name: str) -> Path:
    """Shared by figures 6-8 (chars/token, fertility, byte-fallback rate)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for lang in LANGS:
        rows = sweeps[lang]
        xs = [r["vocab_size"] for r in rows]
        ys = [r[metric_key] for r in rows]
        ax.plot(xs, ys, marker="o", label=LANG_LABEL[lang], color=LANG_COLOR[lang])
    ax.set_xlabel("Vocabulary size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return _save(fig, name)


def plot_token_frequency_loglog(sweeps: dict, name: str) -> Path:
    """Figure 9: rank/frequency (log-log), each language at its OWN selected
    vocab size (Hindi 8,000; Nepali 10,000 - they differ, unlike the
    Hindi/Bhojpuri report where both happened to choose 8,000)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for lang in LANGS:
        vocab = SELECTED_VOCAB[lang]
        row = next(r for r in sweeps[lang] if r["vocab_size"] == vocab)
        freqs = sorted(row["piece_freq"].values(), reverse=True)
        ranks = range(1, len(freqs) + 1)
        ax.plot(ranks, freqs, label=f"{LANG_LABEL[lang]} (vocab={vocab})", color=LANG_COLOR[lang], linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Token rank (log scale)")
    ax.set_ylabel("Frequency in validation sample (log scale)")
    ax.set_title("Token frequency distribution, each language's selected vocab size")
    ax.legend()
    return _save(fig, name)


def plot_token_count_vs_target(data: dict, name: str) -> Path:
    """Figure 10: total token count vs. the 500M target."""
    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(LANGS))
    vals = [data[l]["tokens"] / 1e6 for l in LANGS]
    colors = [LANG_COLOR[l] for l in LANGS]
    ax.bar(x, vals, color=colors)
    ax.axhline(data["target"] / 1e6, color="#c0392b", linestyle="--", linewidth=1.5, label="500M target")
    ax.set_xticks(list(x))
    ax.set_xticklabels([LANG_LABEL[l] for l in LANGS])
    ax.set_xlabel("Language")
    ax.set_ylabel("Total tokens, train+val+test (millions)")
    ax.set_title("Token count vs. the 500M target")
    ax.legend()
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    corpus_size = compute_corpus_size_data()
    manual_share = compute_manual_share_data()
    funnels = {lang: compute_cleaning_funnel(lang) for lang in LANGS}
    drop_reasons = {lang: compute_drop_reasons(lang) for lang in LANGS}
    split_sizes = compute_split_sizes()
    sweeps = measure_all_sweeps()
    token_vs_target = compute_token_count_vs_target()

    figures = [
        ("01_corpus_size.png", "Corpus size: raw vs. final documents", "x-axis, y-axis, legend"),
        ("02_manual_share.png", "Manual vs. downloaded share vs. 20% floor", "x-axis, y-axis, legend"),
        ("03_cleaning_funnel.png", "Cleaning funnel per language", "x-axis, y-axis per subplot"),
        ("04_drop_reasons.png", "Drop reasons breakdown per language", "x-axis, y-axis per subplot"),
        ("05_split_sizes.png", "Train/validation/test split sizes", "x-axis, y-axis, legend"),
        ("06_vocab_chars_per_token.png", "Vocab sweep: chars/token", "x-axis, y-axis, legend"),
        ("07_vocab_fertility.png", "Vocab sweep: fertility", "x-axis, y-axis, legend"),
        ("08_vocab_byte_fallback.png", "Vocab sweep: byte-fallback rate", "x-axis, y-axis, legend"),
        ("09_token_frequency_loglog.png", "Token frequency distribution (log-log)", "x-axis, y-axis, legend"),
        ("10_token_count_vs_target.png", "Token count vs. 500M target", "x-axis, y-axis, legend"),
    ]

    plot_corpus_size(corpus_size, figures[0][0])
    plot_manual_share(manual_share, figures[1][0])
    plot_cleaning_funnel(funnels, figures[2][0])
    plot_drop_reasons(drop_reasons, figures[3][0])
    plot_split_sizes(split_sizes, figures[4][0])
    plot_vocab_metric(sweeps, "chars_per_token", "Characters per token", "Vocab sweep: chars/token", figures[5][0])
    plot_vocab_metric(sweeps, "fertility_tokens_per_word", "Fertility (tokens/word)", "Vocab sweep: fertility", figures[6][0])
    plot_vocab_metric(sweeps, "byte_fallback_rate", "Byte-fallback rate", "Vocab sweep: byte-fallback rate", figures[7][0])
    plot_token_frequency_loglog(sweeps, figures[8][0])
    plot_token_count_vs_target(token_vs_target, figures[9][0])

    metrics_dump = {
        lang: [
            {k: v for k, v in row.items() if k != "piece_freq"}
            for row in sweeps[lang]
        ]
        for lang in LANGS
    }
    METRICS_OUT.write_text(json.dumps(metrics_dump, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote {METRICS_OUT.relative_to(REPO_ROOT)}\n")
    print("Figures generated (all in report/phase1/figures/):")
    for fname, title, note in figures:
        print(f"  - {fname}: title set, {note}")
    print("\nFresh sweep measurements:")
    for lang in LANGS:
        print(f"  {lang}:")
        for row in sweeps[lang]:
            print(
                f"    vocab={row['vocab_size']:>5}  chars/tok={row['chars_per_token']:.4f}  "
                f"fertility={row['fertility_tokens_per_word']:.4f}  "
                f"byte_fb={row['byte_fallback_rate']:.6f}  "
                f"util={row['vocab_utilization']:.4f}  unused_ids={row['unused_ids']}"
            )


if __name__ == "__main__":
    main()

"""Generate every Phase 1 figure for the Hindi (Model H) / Bhojpuri (Model L) report.

Run from the repo root:
    .venv/Scripts/python.exe report/phase1/make_plots.py

Every number plotted here comes from one of:
  - hindi/data/metadata/clean_stats.json, bhojpuri/data/metadata/clean_stats.json
  - hindi/hindi_pipeline_manifest.json (Hindi's corrected final counts, post
    cross-language-duplicate removal: 704,101 docs, not clean_stats.json's
    pre-correction 704,103)
  - the trained SentencePiece models in hindi/tokenizer/ and bhojpuri/tokenizer/,
    re-measured fresh against each language's data/splits/validation/ split
    (same methodology as hindi/tokenizer/analyze_tokenizer.py: sorted file
    glob, first 2000 non-empty docs) so every sweep number and the token
    frequency distribution are exact and reproducible, not transcribed by hand
  - validate_final.log, report/phase1/dataset_statistics.md,
    report/phase1/tokenizer_analysis.md for cross-checking

Computation functions return plain data (dict/list/tuple); plotting functions
take that data and a Path and only call matplotlib. Nothing here decides that
matplotlib and computation should mix beyond that boundary.
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

LANGS = ("hindi", "bhojpuri")
LANG_LABEL = {"hindi": "Hindi (Model H)", "bhojpuri": "Bhojpuri (Model L)"}
LANG_COLOR = {"hindi": "#1f6f8b", "bhojpuri": "#c7562b"}

HINDI_VOCAB_SIZES = [5000, 8000, 10000]
BHOJPURI_VOCAB_SIZES = [5000, 6000, 8000, 10000]
SELECTED_VOCAB = 8000
MAX_ANALYSIS_DOCS = 2000  # matches hindi/tokenizer/analyze_tokenizer.py's default


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def load_clean_stats(lang: str) -> dict:
    """WHY: clean_stats.json is the cleaner's own instrumentation, the single
    authoritative source for drop counts and the raw/interim/cleaned document
    and character checkpoints for that language's run.
    WHAT: reads <lang>/data/metadata/clean_stats.json.
    INPUT SHAPE: lang - 'hindi' or 'bhojpuri'.
    OUTPUT SHAPE: dict, the parsed JSON file as-is."""
    path = REPO_ROOT / lang / "data" / "metadata" / "clean_stats.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_hindi_manifest() -> dict:
    """WHY: Hindi's clean_stats.json predates the 2-document cross-language
    correction (see scripts/remove_cross_language_duplicates.py); the
    manifest holds the corrected final counts (704,101 docs, not 704,103)
    that every other Hindi number in this report is built on.
    WHAT: reads hindi/hindi_pipeline_manifest.json.
    INPUT SHAPE: none.
    OUTPUT SHAPE: dict, the parsed JSON file as-is."""
    path = REPO_ROOT / "hindi" / "hindi_pipeline_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def compute_corpus_size_data() -> dict:
    """WHY: figure 1 needs raw vs. cleaned character counts per language on
    one axis, each measured by the same instrument (the cleaner itself), so
    the comparison is apples-to-apples rather than mixing a raw-collection
    tally with a post-correction cleaned figure.
    WHAT: pulls chars_in/chars_cleaned from clean_stats.json for both
    languages; Hindi's chars_cleaned is overridden with the corrected
    1,764,596,782 from the pipeline manifest (clean_stats.json alone still
    reports the pre-cross-language-fix 1,764,598,107).
    INPUT SHAPE: none.
    OUTPUT SHAPE: {lang: {'raw_chars': int, 'cleaned_chars': int}}."""
    out = {}
    for lang in LANGS:
        stats = load_clean_stats(lang)
        raw_chars = stats["chars_in"]
        cleaned_chars = stats["chars_cleaned"]
        if lang == "hindi":
            cleaned_chars = load_hindi_manifest()["cleaned"]["characters"]
        out[lang] = {"raw_chars": raw_chars, "cleaned_chars": cleaned_chars}
    return out


def compute_manual_share_data() -> dict:
    """WHY: figure 2 checks each language against the hard 20% manual-token
    floor; the char-weighted split (manual vs. downloaded characters in the
    final cleaned corpus) is what scripts/validate_phase1.py actually gates
    on, so that is what the stacked bar must show.
    WHAT: hard-coded from validate_final.log / hindi_pipeline_manifest.json,
    the two independently-verified sources for these exact figures (both
    give 28.65% Hindi / 28.59% Bhojpuri char-weighted).
    INPUT SHAPE: none.
    OUTPUT SHAPE: {lang: {'manual_chars': int, 'downloaded_chars': int, 'total_chars': int, 'manual_pct': float}}."""
    return {
        "hindi": {
            "manual_chars": 505_509_566,
            "downloaded_chars": 1_764_596_782 - 505_509_566,
            "total_chars": 1_764_596_782,
            "manual_pct": 28.65,
        },
        "bhojpuri": {
            "manual_chars": 80_843_912,
            "downloaded_chars": 282_815_177 - 80_843_912,
            "total_chars": 282_815_177,
            "manual_pct": 28.59,
        },
    }


def compute_cleaning_funnel(lang: str) -> list[tuple[str, int]]:
    """WHY: figure 3 shows how many documents survive each cleaning stage in
    the order the pipeline actually applies them, so a reader can see where
    most of the loss happens. The stage order is verified, not assumed: the
    two sub-totals in clean_stats.json (docs_in, docs_interim) checkpoint
    exactly after malformed+short-doc removal and before language-ID, and
    docs_interim - dropped_language - dropped_duplicate == docs_cleaned
    for both languages, confirming raw -> malformed -> short -> language ->
    duplicate is the real order, not just a plausible-sounding one.
    WHAT: builds a cumulative-remaining-documents list by subtracting each
    stage's drop count in that verified order. Hindi gets one extra final
    stage for the 2-document cross-language correction, since that happened
    after clean_hindi.py ran and is not in clean_stats.json's own counters.
    INPUT SHAPE: lang - 'hindi' or 'bhojpuri'.
    OUTPUT SHAPE: list of (stage_label, docs_remaining), raw first, cleaned last."""
    stats = load_clean_stats(lang)
    docs = stats["docs_in"]
    funnel = [("raw", docs)]
    docs -= stats["dropped_malformed"]
    funnel.append(("malformed", docs))
    docs -= stats["dropped_short"]
    funnel.append(("short", docs))
    lang_dropped = sum(stats["dropped_language"].values())
    docs -= lang_dropped
    funnel.append(("language filter", docs))
    docs -= stats["dropped_duplicate"]
    funnel.append(("duplicate", docs))
    if lang == "hindi":
        docs -= 2  # scripts/remove_cross_language_duplicates.py
        funnel.append(("cleaned", docs))
    else:
        funnel.append(("cleaned", docs))
    return funnel


def compute_drop_reasons(lang: str) -> dict[str, int]:
    """WHY: figure 4 breaks the funnel's losses down by individual reason
    instead of cumulative stage, so the biggest single cause of document loss
    (too-short documents, for both languages) is visible on its own.
    WHAT: flattens clean_stats.json's drop-reason fields into one dict;
    Hindi's nested dropped_language sub-reasons are kept separate rather than
    summed, and Hindi's 2-document cross-language correction is added since
    it is a real, measured drop not captured in clean_stats.json.
    INPUT SHAPE: lang - 'hindi' or 'bhojpuri'.
    OUTPUT SHAPE: {reason_label: doc_count}."""
    stats = load_clean_stats(lang)
    reasons = {
        "malformed Unicode": stats["dropped_malformed"],
        "too short (<200 chars)": stats["dropped_short"],
    }
    for reason, count in stats["dropped_language"].items():
        reasons[reason.replace("_", " ")] = count
    reasons["exact/near duplicate"] = stats["dropped_duplicate"]
    if lang == "hindi":
        reasons["cross-language duplicate"] = 2
    return reasons


def compute_split_sizes() -> dict:
    """WHY: figure 5 needs train/validation/test document counts per
    language; these are the same figures validate_phase1.py's no-leakage
    check confirms and hindi_pipeline_manifest.json / dataset_statistics.md
    both independently agree on, so they are cited rather than re-derived
    from the multi-hundred-thousand-line split manifests at plot time.
    WHAT: returns the fixed, cross-verified split sizes.
    INPUT SHAPE: none.
    OUTPUT SHAPE: {lang: {'train': int, 'validation': int, 'test': int}}."""
    return {
        "hindi": {"train": 675_937, "validation": 14_082, "test": 14_082},
        "bhojpuri": {"train": 487_674, "validation": 10_160, "test": 10_160},
    }


def _load_validation_texts(lang: str, max_docs: int = MAX_ANALYSIS_DOCS) -> list[str]:
    """WHY: every sweep metric and the token-frequency plot must be measured
    on held-out text the tokenizer never trained on, using the exact same
    sample every existing tokenizer_analysis.md number was measured on, so a
    fresh run here reproduces (not contradicts) the published figures.
    WHAT: sorted glob over data/splits/validation/text/*.txt, first max_docs
    non-empty files read and stripped - identical to
    hindi/tokenizer/analyze_tokenizer.py's load_validation_text().
    INPUT SHAPE: lang - 'hindi' or 'bhojpuri'; max_docs - sample cap.
    OUTPUT SHAPE: list of document text strings, len <= max_docs."""
    val_dir = REPO_ROOT / lang / "data" / "splits" / "validation" / "text"
    files = sorted(val_dir.glob("*.txt"))
    texts = []
    for fp in files[:max_docs]:
        text = fp.read_text(encoding="utf-8").strip()
        if text:
            texts.append(text)
    return texts


def measure_tokenizer(lang: str, vocab_size: int, texts: list[str]) -> dict:
    """WHY: this is the single fresh measurement every sweep plot (6, 7, 8)
    and the sweep table in the report are built from. Re-measuring directly
    against the trained .model files, instead of transcribing
    tokenizer_analysis.md by hand, also yields the exact unused-vocab-id
    count and the full per-piece frequency table that the published report
    only kept the top 15 of.
    WHAT: encodes every validation text with the SentencePiece model (as ids
    and as piece strings), accumulates chars/tokens/words/UNK/byte-fallback
    counts and a full piece-frequency Counter.
    INPUT SHAPE: lang; vocab_size - one swept size; texts - from
    _load_validation_texts(lang), shared across all vocab sizes for that lang.
    OUTPUT SHAPE: dict with keys vocab_size, chars_per_token,
    fertility_tokens_per_word, unk_rate, byte_fallback_rate,
    vocab_utilization, used_ids, unused_ids, piece_freq (Counter)."""
    model_path = REPO_ROOT / lang / "tokenizer" / f"{lang}_unigram_{vocab_size}.model"
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
    """WHY: bundles the per-vocab-size measurement across both languages'
    full sweeps into one pass, so plotting functions and the report-writing
    step share one consistent set of freshly-measured numbers.
    WHAT: calls measure_tokenizer() for every (lang, vocab_size) pair in
    HINDI_VOCAB_SIZES / BHOJPURI_VOCAB_SIZES, loading each language's
    validation sample once and reusing it across that language's sweep.
    INPUT SHAPE: none.
    OUTPUT SHAPE: {lang: [dict, ...]}, one dict per vocab size (see
    measure_tokenizer's OUTPUT SHAPE), ordered by ascending vocab size."""
    sizes = {"hindi": HINDI_VOCAB_SIZES, "bhojpuri": BHOJPURI_VOCAB_SIZES}
    results = {}
    for lang in LANGS:
        texts = _load_validation_texts(lang)
        results[lang] = [measure_tokenizer(lang, vs, texts) for vs in sizes[lang]]
    return results


def compute_token_count_vs_target() -> dict:
    """WHY: figure 10 checks each language's exact, full-corpus (train +
    validation + test, unsampled) token count at its selected vocab size
    against the 500M-token planning target from the assignment brief.
    WHAT: hard-coded from scripts/count_tokens.py's exhaustive, unsampled
    pass over every document (see hindi_pipeline_manifest.json
    tokenizers[8000].full_corpus_exact and
    hindi/hindi_token_counts_by_source.json for Hindi; bhojpuri_tokenizer_
    sweep.log's 'full-corpus token count vocab_size=8000' block for Bhojpuri).
    INPUT SHAPE: none.
    OUTPUT SHAPE: {lang: {'tokens': int, 'pct_of_target': float}}, plus a
    'target' key giving the 500,000,000 reference value."""
    return {
        "hindi": {"tokens": 505_852_028, "pct_of_target": 101.2},
        "bhojpuri": {"tokens": 84_217_032, "pct_of_target": 16.8},
        "target": 500_000_000,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save(fig, name: str) -> Path:
    """WHY: every figure gets identical save settings (tight bounding box so
    rotated tick labels are not clipped, a fixed DPI so file sizes and
    on-GitHub rendering are predictable) and the figure is closed immediately
    so 10 sequential plots do not accumulate open figures in memory.
    WHAT: writes fig to FIGURES_DIR/name and closes it.
    INPUT SHAPE: fig - a matplotlib Figure; name - filename including .png.
    OUTPUT SHAPE: Path to the written file."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_corpus_size(data: dict, name: str) -> Path:
    """WHY: shows how much of each language's raw collection survived
    cleaning, in absolute characters, so the two languages' scale difference
    (Hindi ~6.2x Bhojpuri) and cleaning loss are both visible in one figure.
    WHAT: grouped bar chart, one group per language, raw vs. cleaned bars.
    INPUT SHAPE: data - from compute_corpus_size_data(); name - output filename.
    OUTPUT SHAPE: Path to the saved PNG."""
    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(LANGS))
    width = 0.35
    raw_vals = [data[l]["raw_chars"] / 1e9 for l in LANGS]
    cleaned_vals = [data[l]["cleaned_chars"] / 1e9 for l in LANGS]
    ax.bar([i - width / 2 for i in x], raw_vals, width, label="Raw", color="#9aa5ad")
    ax.bar([i + width / 2 for i in x], cleaned_vals, width, label="Cleaned", color="#1f6f8b")
    ax.set_xticks(list(x))
    ax.set_xticklabels([LANG_LABEL[l] for l in LANGS])
    ax.set_xlabel("Language")
    ax.set_ylabel("Characters (billions)")
    ax.set_title("Corpus size: raw vs. cleaned characters")
    ax.legend()
    return _save(fig, name)


def plot_manual_share(data: dict, name: str) -> Path:
    """WHY: the 20% manual-token floor is a hard pass/fail requirement for
    both models; drawing it as a reference line against the stacked
    manual/downloaded split makes the margin visually obvious.
    WHAT: stacked bar per language (manual + downloaded = 100%), with a
    horizontal dashed line at 20%.
    INPUT SHAPE: data - from compute_manual_share_data(); name - output filename.
    OUTPUT SHAPE: Path to the saved PNG."""
    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(LANGS))
    manual_pct = [data[l]["manual_chars"] / data[l]["total_chars"] * 100 for l in LANGS]
    downloaded_pct = [100 - m for m in manual_pct]
    ax.bar(x, manual_pct, label="Manual", color="#2e8b57")
    ax.bar(x, downloaded_pct, bottom=manual_pct, label="Downloaded", color="#9aa5ad")
    ax.axhline(20.0, color="#c0392b", linestyle="--", linewidth=1.5, label="20% floor")
    ax.set_xticks(list(x))
    ax.set_xticklabels([LANG_LABEL[l] for l in LANGS])
    ax.set_xlabel("Language")
    ax.set_ylabel("Share of cleaned corpus characters (%)")
    ax.set_title("Manual vs. downloaded share, against the 20% floor")
    ax.set_ylim(0, 100)
    ax.legend()
    return _save(fig, name)


def plot_cleaning_funnel(funnels: dict, name: str) -> Path:
    """WHY: shows where each language's document loss happens across the
    cleaning pipeline's real stage order, one subplot per language since the
    two corpora differ by ~3.5x in scale and a single shared y-axis would
    flatten Bhojpuri's bars.
    WHAT: two side-by-side bar charts, x-axis = pipeline stage, y-axis =
    documents remaining after that stage.
    INPUT SHAPE: funnels - {lang: list of (stage_label, docs_remaining)} from
    compute_cleaning_funnel(); name - output filename.
    OUTPUT SHAPE: Path to the saved PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, lang in zip(axes, LANGS):
        stages, counts = zip(*funnels[lang])
        ax.bar(stages, counts, color=LANG_COLOR[lang])
        ax.set_title(LANG_LABEL[lang])
        ax.set_xlabel("Cleaning stage")
        ax.set_ylabel("Documents remaining")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Cleaning funnel: documents remaining after each filter stage")
    fig.tight_layout()
    return _save(fig, name)


def plot_drop_reasons(reasons: dict, name: str) -> Path:
    """WHY: complements the funnel by ranking individual drop reasons
    (rather than cumulative stages), showing that too-short documents, not
    language-ID or dedup, account for most of each language's document loss.
    WHAT: two side-by-side horizontal bar charts, one per language, reasons
    sorted by count.
    INPUT SHAPE: reasons - {lang: {reason_label: count}} from
    compute_drop_reasons(); name - output filename.
    OUTPUT SHAPE: Path to the saved PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, lang in zip(axes, LANGS):
        items = sorted(reasons[lang].items(), key=lambda kv: kv[1])
        labels, counts = zip(*items)
        ax.barh(labels, counts, color=LANG_COLOR[lang])
        ax.set_title(LANG_LABEL[lang])
        ax.set_xlabel("Documents dropped")
        ax.set_ylabel("Reason")
    fig.suptitle("Drop reasons breakdown")
    fig.tight_layout()
    return _save(fig, name)


def plot_split_sizes(data: dict, name: str) -> Path:
    """WHY: confirms visually that both languages follow the same
    train/validation/test proportions despite very different absolute sizes,
    on a log y-axis since train dwarfs validation/test by ~48x (Hindi) and
    ~48x (Bhojpuri).
    WHAT: grouped bar chart, one group per split, one bar per language,
    legend distinguishes language.
    INPUT SHAPE: data - from compute_split_sizes(); name - output filename.
    OUTPUT SHAPE: Path to the saved PNG."""
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
    """WHY: shared by figures 6, 7, 8 (chars/token, fertility, byte-fallback
    rate vs. vocab size) since all three are the same shape of plot - one
    line per language across that language's swept vocab sizes - and
    duplicating this three times would just be copy-paste risk.
    WHAT: line+marker plot, x = vocab size, y = the requested metric, one
    line per language, legend distinguishes language.
    INPUT SHAPE: sweeps - from measure_all_sweeps(); metric_key - a key
    present in each per-vocab-size dict measure_tokenizer() returns; ylabel,
    title - axis/plot text; name - output filename.
    OUTPUT SHAPE: Path to the saved PNG."""
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
    """WHY: a Zipf-shaped log-log rank/frequency curve is the standard way
    to show a subword vocabulary's usage is concentrated in a small number of
    high-frequency pieces (postpositions, punctuation) with a long tail -
    this is measured on the selected vocab_size=8000 model for each language,
    the size actually used going into Phase 2.
    WHAT: for each language, takes the full piece-frequency Counter measured
    at vocab_size=8000, sorts descending, plots frequency vs. rank on
    log-log axes.
    INPUT SHAPE: sweeps - from measure_all_sweeps(); name - output filename.
    OUTPUT SHAPE: Path to the saved PNG."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for lang in LANGS:
        row = next(r for r in sweeps[lang] if r["vocab_size"] == SELECTED_VOCAB)
        freqs = sorted(row["piece_freq"].values(), reverse=True)
        ranks = range(1, len(freqs) + 1)
        ax.plot(ranks, freqs, label=LANG_LABEL[lang], color=LANG_COLOR[lang], linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Token rank (log scale)")
    ax.set_ylabel("Frequency in validation sample (log scale)")
    ax.set_title(f"Token frequency distribution, vocab_size={SELECTED_VOCAB}")
    ax.legend()
    return _save(fig, name)


def plot_token_count_vs_target(data: dict, name: str) -> Path:
    """WHY: figure 10 is the headline Phase 1 number for each language - how
    close the selected tokenizer's exact, full-corpus token count comes to
    the assignment's ~500M-token planning target - with the target drawn as
    a reference line so Hindi's 101.2% and Bhojpuri's documented shortfall
    are both legible on one axis.
    WHAT: bar chart, one bar per language, with a horizontal line at the
    500M target.
    INPUT SHAPE: data - from compute_token_count_vs_target(); name - output filename.
    OUTPUT SHAPE: Path to the saved PNG."""
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
    """WHY: single entry point so `python report/phase1/make_plots.py`
    regenerates every figure and the measured-metrics JSON in one run, in a
    fixed, numbered order matching the report's REQUIRED PLOTS list.
    WHAT: runs every compute_* function, feeds the results to the matching
    plot_* function, writes measured_tokenizer_metrics.json (the fresh sweep
    measurements, minus the large piece_freq counters, for the report text
    to cite), and prints a checklist of every figure with its title/axis/
    legend status.
    INPUT SHAPE: none (CLI entry point).
    OUTPUT SHAPE: none (writes files, prints a report to stdout)."""
    corpus_size = compute_corpus_size_data()
    manual_share = compute_manual_share_data()
    funnels = {lang: compute_cleaning_funnel(lang) for lang in LANGS}
    drop_reasons = {lang: compute_drop_reasons(lang) for lang in LANGS}
    split_sizes = compute_split_sizes()
    sweeps = measure_all_sweeps()
    token_vs_target = compute_token_count_vs_target()

    figures = [
        ("01_corpus_size.png", "Corpus size: raw vs. cleaned characters", "x-axis, y-axis, legend"),
        ("02_manual_share.png", "Manual vs. downloaded share vs. 20% floor", "x-axis, y-axis, legend"),
        ("03_cleaning_funnel.png", "Cleaning funnel per language", "x-axis, y-axis per subplot (single series - no legend needed)"),
        ("04_drop_reasons.png", "Drop reasons breakdown per language", "x-axis, y-axis per subplot (single series - no legend needed)"),
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
    print("\nFresh sweep measurements (cross-check against tokenizer_analysis.md):")
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

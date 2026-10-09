"""Cross-check the report and the viva guide against the result artifacts.

Three checks:

1. Every headline number is read out of the JSON/JSONL files the runs produced,
   formatted the way each document writes it, and must appear in that document.
2. The report and the viva guide must not disagree: a value is required in both
   (in their respective formats), so one cannot be edited without the other.
3. Every ``Name()`` or ``ClassName`` the viva guide cites as code must actually
   exist in the repository, so the guide cannot invent an API.

    python report/final/check_report_facts.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = (ROOT / "report/final/final_report.tex").read_text(encoding="utf-8")
VIVA_PATH = ROOT / "VIVA_GUIDE.md"
VIVA = VIVA_PATH.read_text(encoding="utf-8") if VIVA_PATH.exists() else None
VIVA_TEX_PATH = ROOT / "docs/VIVA_GUIDE.tex"
VIVA_TEX = VIVA_TEX_PATH.read_text(encoding="utf-8") if VIVA_TEX_PATH.exists() else None


def load(rel: str) -> dict:
    """Read one JSON artifact relative to the repository root."""
    with (ROOT / rel).open(encoding="utf-8") as fh:
        return json.load(fh)


def tex_int(n: int) -> str:
    """An integer as the LaTeX source writes it (thin-space groups)."""
    return f"{n:,}".replace(",", "{,}")


def md_int(n: int) -> str:
    """An integer as the Markdown guide writes it."""
    return f"{n:,}"


checks: list[tuple[str, str, str]] = []


def expect(label: str, tex: str, viva: str | None = None) -> None:
    """Require `tex` in the report and `viva` (default: same) in the guide."""
    checks.append((label, tex, tex if viva is None else viva))


h_eval = load("report/phase2/hindi_eval.json")
l_eval = load("report/phase2/nepali_eval.json")
h_fine = load("report/phase3/hindi_reasoning_finetuned.json")
l_fine = load("report/phase3/nepali_reasoning_finetuned.json")
h_pre = load("report/phase3/hindi_reasoning_pretrained.json")
l_pre = load("report/phase3/nepali_reasoning_pretrained.json")
h_att = load("report/phase2/hindi_attention.json")
l_att = load("report/phase2/nepali_attention.json")
hi_src = load("hindi/hindi_token_counts_by_source.json")
ne_cor = load("nepali/data/final/corpus_stats.json")

# ---- Phase 1 ----
expect("Hindi corpus tokens", tex_int(hi_src["total_tokens"]), md_int(hi_src["total_tokens"]))
expect("Nepali corpus tokens", tex_int(ne_cor["overall_tokens"]), md_int(ne_cor["overall_tokens"]))
pct = hi_src["totals_by_type"]["manual"]["pct_of_total"]
expect("Hindi manual share", f"{pct:.2f}\\%", f"{pct:.2f}%")
expect("Nepali dedup near", tex_int(ne_cor["dedup"]["near_removed"]), md_int(ne_cor["dedup"]["near_removed"]))
expect(
    "Nepali decontamination",
    tex_int(ne_cor["decontamination"]["contaminated_removed"]),
    md_int(ne_cor["decontamination"]["contaminated_removed"]),
)

# ---- Phase 2: parameters and intrinsic metrics ----
for tag, ev in (("H", h_eval), ("L", l_eval)):
    pc = ev["parameter_count"]
    expect(f"{tag} total params", tex_int(pc["total"]), md_int(pc["total"]))
    expect(f"{tag} non-embedding params", tex_int(pc["non_embedding"]), md_int(pc["non_embedding"]))
    intr = ev["intrinsic"]
    expect(f"{tag} perplexity", f'{intr["perplexity"]:.2f}')
    expect(f"{tag} bits-per-byte", f'{intr["bits_per_byte"]:.4f}')
    expect(f"{tag} cross-entropy", f'{intr["cross_entropy_nats"]:.4f}')
    expect(f"{tag} bytes scored", tex_int(intr["bytes_scored"]), md_int(intr["bytes_scored"]))
    expect(f"{tag} tokens scored", tex_int(intr["tokens_scored"]), md_int(intr["tokens_scored"]))
    bpt = intr["bytes_scored"] / intr["tokens_scored"]
    expect(f"{tag} bytes per token", f"{bpt:.3f}")

# ---- Phase 2: generation, every setting and metric ----
for tag, ev in (("H", h_eval), ("L", l_eval)):
    for name, m in ev["generation"]["settings"].items():
        expect(f"{tag} {name} chrF++", f'{m["chrf_pp"]:.2f}')
        expect(f"{tag} {name} BLEU", f'{m["bleu"]:.2f}')
        expect(f"{tag} {name} ROUGE-L", f'{m["rouge_l"]:.3f}')
        expect(f"{tag} {name} distinct-2", f'{m["distinct_2"]:.3f}')

# ---- Phase 2: attention ----
for tag, at in (("H", h_att), ("L", l_att)):
    expect(f"{tag} mean entropy", f'{at["mean_entropy"]:.4f}')
    expect(f"{tag} mean distance", f'{at["mean_distance"]:.2f}')
expect("uniform baseline", f'{h_att["uniform_baseline_entropy"]:.4f}')

# ---- Phase 3 ----
for tag, pre, fine in (("H", h_pre, h_fine), ("L", l_pre, l_fine)):
    expect(f"{tag} exact match finetuned",
           f'{fine["exact_match"]["accuracy"]*100:.2f}\\%', f'{fine["exact_match"]["accuracy"]*100:.2f}%')
    expect(f"{tag} forced choice pretrained",
           f'{pre["forced_choice"]["accuracy"]*100:.2f}\\%', f'{pre["forced_choice"]["accuracy"]*100:.2f}%')
    expect(f"{tag} forced choice finetuned",
           f'{fine["forced_choice"]["accuracy"]*100:.2f}\\%', f'{fine["forced_choice"]["accuracy"]*100:.2f}%')
    expect(f"{tag} answer-token loss pre", f'{pre["answer_token_loss"]["answer_token_loss"]:.4f}')
    expect(f"{tag} answer-token loss post", f'{fine["answer_token_loss"]["answer_token_loss"]:.4f}')
chance = h_fine["forced_choice"]["chance_accuracy"] * 100
expect("chance rate", f"{chance:.2f}\\%", f"{chance:.2f}%")

tr = h_fine["error_analysis"]["transitive3"]
expect("H transitive3 middle", f'{tr["model_answer_position"][1]*100:.1f}\\%', f'{tr["model_answer_position"][1]*100:.1f}%')
expect("H transitive3 off-prompt", f'{tr["answered_off_prompt"]*100:.1f}\\%', f'{tr["answered_off_prompt"]*100:.1f}%')

# ---- Run the value checks ----
failures: list[str] = []
for label, tex_form, viva_form in checks:
    if tex_form not in TEX:
        failures.append(f"  report    {label:32s} expected '{tex_form}'")
    if VIVA is not None and viva_form not in VIVA:
        failures.append(f"  viva md   {label:32s} expected '{viva_form}'")
    if VIVA_TEX is not None:
        # The LaTeX guide writes plain thousands separators but keeps \% for
        # percent signs, so it is the report form with the groups unpacked.
        vt = tex_form.replace("{,}", ",")
        if vt not in VIVA_TEX:
            failures.append(f"  viva tex  {label:32s} expected '{vt}'")

targets = ["final_report.tex"]
if VIVA is not None:
    targets.append("VIVA_GUIDE.md")
if VIVA_TEX is not None:
    targets.append("docs/VIVA_GUIDE.tex")
print(f"checked {len(checks)} values against " + ", ".join(targets))

# ---- Check that every code name the viva guide cites really exists ----
if VIVA is not None:
    sources = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in ROOT.rglob("*.py")
        if ".venv" not in p.as_posix() and "__pycache__" not in p.as_posix()
    )
    cited = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)\(\)`", VIVA))
    cited |= {m for m in re.findall(r"`([A-Z][A-Za-z0-9]+)`", VIVA)}
    # Names that are PyTorch API or prose, not this repository's own code.
    external = {
        "LayerNorm", "GradScaler", "DEFAULT_NUM_PERM", "DEFAULT_BANDS",
        "DEFAULT_ROWS", "DEFAULT_SHINGLE", "None", "predict",
    }
    missing_names = sorted(
        n for n in cited - external
        if not re.search(rf"\b(def|class)\s+{re.escape(n)}\b", sources)
    )
    print(f"checked {len(cited)} code names cited by the viva guide")
    for n in missing_names:
        failures.append(f"  viva    code name '{n}' not found as a def/class in the repo")

if failures:
    print(f"\n{len(failures)} problem(s):")
    for line in failures:
        print(line)
    sys.exit(1)
print("all values traced to an artifact; report and guide agree.")

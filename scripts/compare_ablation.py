"""Compare the no-positional-embeddings ablation against the standard Model H.

The ablation is wall-clock limited and therefore stops far short of the
baseline's 14,520 steps. Comparing its final numbers against the baseline's
final numbers would confuse "no positional embeddings" with "trained for half
as long", so every comparison here is made at a **matched step**: the largest
validation checkpoint both runs reached.

Reads, for each run, the artifacts the Kaggle notebook produces:

    <tag>_training_log.jsonl   per-step train loss, per-500-step val loss
    <tag>_eval.json            perplexity, bits-per-byte, generation metrics
    <tag>_attention.json       entropy, mean distance, per-layer breakdown

Writes ``report/bonus/ablation_comparison.json`` and a loss-curve figure, and
prints the markdown tables used in the report.

Run::

    python scripts/compare_ablation.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE2 = REPO_ROOT / "report" / "phase2"
BONUS = REPO_ROOT / "report" / "bonus"

BASELINE = {
    "name": "standard (learned positions)",
    "log": PHASE2 / "hindi_training_log.jsonl",
    "eval": PHASE2 / "hindi_eval.json",
    "attention": PHASE2 / "hindi_attention.json",
}
ABLATED = {
    "name": "ablated (no positions)",
    "log": BONUS / "hindi_nopos_training_log.jsonl",
    "eval": BONUS / "hindi_nopos_eval.json",
    "attention": BONUS / "hindi_nopos_attention.json",
}


def read_log(path: Path) -> tuple[list[dict], list[dict]]:
    """Split a training log into its train-loss and val-loss records."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return ([r for r in rows if "train_loss" in r],
            [r for r in rows if "val_loss" in r])


def load(spec: dict) -> dict:
    """Load one run's artifacts, tolerating missing optional pieces."""
    out = {"name": spec["name"]}
    train, val = read_log(spec["log"])
    out["train"], out["val"] = train, val
    out["final_step"] = max(r["step"] for r in train) if train else 0
    for key in ("eval", "attention"):
        path = spec[key]
        out[key] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return out


def matched_step(a: dict, b: dict) -> int:
    """Largest validation step both runs reached.

    Raises:
        SystemExit: If the runs share no validation checkpoint at all.
    """
    steps_a = {r["step"] for r in a["val"]}
    steps_b = {r["step"] for r in b["val"]}
    common = steps_a & steps_b
    if not common:
        raise SystemExit("the two runs share no validation step; cannot compare")
    return max(common)


def val_at(run: dict, step: int) -> float:
    """Validation loss for one run at an exact step."""
    for record in run["val"]:
        if record["step"] == step:
            return record["val_loss"]
    raise KeyError(step)


def pct(new: float, old: float) -> str:
    """Signed percentage change of ``new`` relative to ``old``."""
    return f"{(new - old) / old * 100:+.1f}%"


def main() -> None:
    """Build the comparison, write it out, and print the report tables."""
    sys.stdout.reconfigure(encoding="utf-8")

    missing = [str(p) for spec in (BASELINE, ABLATED)
               for p in (spec["log"],) if not p.exists()]
    if missing:
        raise SystemExit("missing required input(s):\n  " + "\n  ".join(missing))

    base, abl = load(BASELINE), load(ABLATED)
    step = matched_step(base, abl)

    base_val, abl_val = val_at(base, step), val_at(abl, step)
    summary = {
        "matched_step": step,
        "note": (
            "Every figure below is taken at the same step in both runs. The "
            "ablation was wall-clock limited and did not train as long as the "
            "baseline, so its own final numbers are not comparable to the "
            "baseline's final numbers."
        ),
        "runs": {},
    }

    print(f"\n## Matched at step {step:,}\n")
    print("| | standard | ablated (no pos) | change |")
    print("|---|---:|---:|---|")
    print(f"| Validation loss (nats) | {base_val:.4f} | {abl_val:.4f} | "
          f"{pct(abl_val, base_val)} |")
    print(f"| Validation perplexity | {math.exp(base_val):.2f} | "
          f"{math.exp(abl_val):.2f} | {pct(math.exp(abl_val), math.exp(base_val))} |")

    for tag, run in (("standard", base), ("ablated", abl)):
        entry = {
            "final_step": run["final_step"],
            "val_loss_at_matched_step": val_at(run, step),
            "val_ppl_at_matched_step": math.exp(val_at(run, step)),
        }
        if run["eval"]:
            ev = run["eval"]
            entry["pos_encoding"] = ev["model_config"]["pos_encoding"]
            entry["parameter_count"] = ev["parameter_count"]
            entry["intrinsic"] = ev["intrinsic"]
            # "settings" is keyed by decoding label ("greedy", "T=0.5", ...).
            entry["generation"] = ev["generation"].get("settings", {})
        if run["attention"]:
            at = run["attention"]
            entry["attention"] = {
                "mean_entropy": at["mean_entropy"],
                "mean_distance": at["mean_distance"],
                "uniform_baseline_entropy": at["uniform_baseline_entropy"],
                "local_head_fraction": at["local_head_fraction"],
            }
        summary["runs"][tag] = entry

    if base["eval"] and abl["eval"]:
        bi, ai = base["eval"]["intrinsic"], abl["eval"]["intrinsic"]
        print("\n## Test-set intrinsic metrics\n")
        print("| | standard | ablated | change |")
        print("|---|---:|---:|---|")
        for label, key in (("Cross-entropy (nats)", "cross_entropy_nats"),
                           ("Perplexity", "perplexity"),
                           ("Bits-per-byte", "bits_per_byte")):
            print(f"| {label} | {bi[key]:.4f} | {ai[key]:.4f} | "
                  f"{pct(ai[key], bi[key])} |")
        print("\n*The ablated checkpoint is far less trained than the "
              "baseline's, so read these alongside the matched-step row above.*")

        bg = base["eval"]["generation"]["settings"]
        ag = abl["eval"]["generation"]["settings"]
        print("\n## Generation quality\n")
        print("| Decoding | chrF++ std | chrF++ abl | BLEU std | BLEU abl "
              "| repetition std | repetition abl |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for label in bg:
            if label not in ag:
                continue
            b, a = bg[label], ag[label]
            print(f"| {label} | {b['chrf_pp']:.2f} | {a['chrf_pp']:.2f} "
                  f"| {b['bleu']:.2f} | {a['bleu']:.2f} "
                  f"| {b['repetition_rate']:.3f} | {a['repetition_rate']:.3f} |")

    if base["attention"] and abl["attention"]:
        ba, aa = base["attention"], abl["attention"]
        print("\n## Attention\n")
        print("| | standard | ablated | uniform baseline |")
        print("|---|---:|---:|---:|")
        print(f"| Mean entropy (nats) | {ba['mean_entropy']:.4f} | "
              f"{aa['mean_entropy']:.4f} | {ba['uniform_baseline_entropy']:.4f} |")
        print(f"| Mean distance (tokens) | {ba['mean_distance']:.2f} | "
              f"{aa['mean_distance']:.2f} | - |")
        print(f"| Local-head fraction | {ba['local_head_fraction']:.4f} | "
              f"{aa['local_head_fraction']:.4f} | - |")

    BONUS.mkdir(parents=True, exist_ok=True)
    out = BONUS / "ablation_comparison.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")

    # Loss curves, both runs on shared axes.
    sys.path.insert(0, str(REPO_ROOT))
    from hindi.eval.viz import plot_loss_curves

    figdir = BONUS / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for axis in ("step", "tokens"):
        path = plot_loss_curves(
            {"standard": BASELINE["log"], "no positions": ABLATED["log"]},
            figdir / f"ablation_loss_curves_by_{axis}.png",
            x_axis=axis,
        )
        print(f"wrote {Path(path).relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

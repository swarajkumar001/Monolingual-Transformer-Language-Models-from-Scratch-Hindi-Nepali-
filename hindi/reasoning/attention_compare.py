"""Compare Model H's attention before and after reasoning finetuning.

Deliverable 4 asks for a pretrained-vs-finetuned attention comparison on at
least one early and one late layer, specifically on comparative-reasoning
prompts. Three choices make that comparison mean something.

**The same prompts go through both models.** Attention entropy depends heavily
on what is being attended to, so feeding the two checkpoints different text
would confound the finetuning effect with a text effect. The prompts here come
from the reasoning *test* split -- held-out names, held-out templates -- so
neither checkpoint has memorised them and the finetuned model is not being
flattered by its training data.

**Prompts are bucketed to a single length.** Mean attention distance is measured
in token positions and entropy's uniform baseline is ``log(t)`` at position
``t``, so both grow with sequence length. Comparing a 20-token prompt against a
40-token one would show a difference that is purely arithmetic. Every prompt in
a comparison batch has the same token count.

**Per-head deltas are reported, not just per-layer means.** Finetuning a small
model for a narrow task tends to change a few heads sharply rather than shift
every head a little; a layer mean hides exactly that.

This file belongs to **Model H** (Hindi). Model L has its own independent copy
under ``nepali/reasoning/``.

Usage
-----
    python -m hindi.reasoning.attention_compare \\
        --pretrained checkpoints/hindi/ckpt_best.pt \\
        --finetuned  checkpoints/hindi_reasoning/ckpt_best.pt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from ..eval import attention as attn_lib
from ..eval import viz
from ..model.config import GPTConfig
from ..model.transformer import GPT
from .dataset import ReasoningDataset, load_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]

LANGUAGE = "hindi"


def load_model(checkpoint: str | Path, device: str) -> tuple[GPT, dict]:
    """Restore a model from either a pretrained or a finetuned checkpoint.

    Args:
        checkpoint: Path to the ``.pt`` file.
        device: Device to place the model on.

    Returns:
        The model in eval mode, and a dict of provenance fields for the report.
    """
    path = Path(checkpoint)
    if not path.is_absolute():
        path = REPO_ROOT / path
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = GPTConfig.from_dict(ckpt["config"]["model"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    meta = {
        "checkpoint": str(path),
        "stage": ckpt.get("stage", "pretrained"),
        "step": ckpt.get("step", -1),
    }
    return model, meta


def reasoning_prompt_batch(
    data: ReasoningDataset, batch_size: int, device: str
) -> tuple[torch.Tensor, list[dict]]:
    """Take one batch of equal-length comparative-reasoning prompts.

    The largest same-length bucket is used, so the batch is as representative as
    a single length allows.

    Args:
        data: The reasoning split to draw from.
        batch_size: How many prompts to include.
        device: Device for the returned tensor.

    Returns:
        Token ids of shape ``(B, T)`` and the matching dataset records.
    """
    buckets: dict[int, list[dict]] = defaultdict(list)
    for row in data.rows:
        buckets[len(row["prompt_ids"])].append(row)
    # Prefer a long prompt bucket: two-hop transitive questions are the ones
    # where long-range attention would have to show up if it shows up at all.
    length = max(buckets, key=lambda k: (len(buckets[k]) >= batch_size, k))
    rows = buckets[length][:batch_size]
    ids = torch.tensor([r["prompt_ids"] for r in rows],
                       dtype=torch.long, device=device)
    return ids, rows


def per_head_table(summary: dict) -> dict[str, dict]:
    """Flatten a model attention summary to ``"L{layer}H{head}"`` keys.

    Args:
        summary: Output of :func:`attention.summarise_model_attention`.

    Returns:
        Mapping from head label to its entropy and mean distance.
    """
    out = {}
    for layer in summary["layers"]:
        index = layer["layer"]
        for head, (entropy, distance) in enumerate(
            zip(layer["entropy"], layer["mean_distance"])
        ):
            out[f"L{index}H{head}"] = {
                "entropy": round(float(entropy), 4),
                "mean_distance": round(float(distance), 4),
            }
    return out


def compare(
    pretrained: str | Path,
    finetuned: str | Path,
    data_dir: str | Path | None = None,
    tokenizer_model: str | Path | None = None,
    split: str = "test",
    batch_size: int = 16,
    device: str = "auto",
    fig_dir: str | Path | None = None,
) -> dict:
    """Run both checkpoints on identical prompts and summarise the difference.

    Args:
        pretrained: Path to the Phase 2 checkpoint.
        finetuned: Path to the Phase 3 checkpoint.
        data_dir: Override for the reasoning data directory.
        tokenizer_model: Override for the tokenizer.
        split: Which reasoning split to draw prompts from.
        batch_size: Prompts per comparison batch.
        device: ``"auto"``, ``"cuda"`` or ``"cpu"``.
        fig_dir: Where heatmaps are written.

    Returns:
        The comparison report, including per-head deltas and the layers plotted.
    """
    resolved = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto" else device
    )
    tok = Path(tokenizer_model) if tokenizer_model else None
    if tok and not tok.is_absolute():
        tok = REPO_ROOT / tok
    tokenizer = load_tokenizer(tok)
    data = ReasoningDataset(split, tokenizer, data_dir, device=resolved)
    ids, rows = reasoning_prompt_batch(data, batch_size, resolved)
    tokens = [tokenizer.decode([int(t)]) for t in ids[0].cpu().numpy()]

    figures = Path(fig_dir) if fig_dir else REPO_ROOT / "report/phase3/figures"
    figures.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "language": LANGUAGE,
        "split": split,
        "prompts_compared": len(rows),
        "sequence_length": int(ids.size(1)),
        "uniform_baseline_entropy": round(
            attn_lib.uniform_baseline_entropy(int(ids.size(1))), 4
        ),
        "example_prompts": [r["prompt"] for r in rows[:3]],
        "example_tokens": tokens,
        "models": {},
        "figures": [],
    }

    summaries = {}
    for name, checkpoint in (("pretrained", pretrained), ("finetuned", finetuned)):
        model, meta = load_model(checkpoint, resolved)
        attentions = attn_lib.collect_attention(model, ids)
        early, late = 0, len(attentions) - 1
        for layer in (early, late):
            path = viz.plot_attention_heatmap(
                attentions[layer][0].cpu().numpy(), tokens, layer, [0, 1, 2, 3],
                figures / f"{LANGUAGE}_{name}_reasoning_layer{layer}.png",
                language=f"{LANGUAGE} ({name})",
            )
            report["figures"].append(str(path.name))
        summary = attn_lib.summarise_model_attention(model, ids)
        summary.update(meta)
        summaries[name] = summary
        report["models"][name] = {
            **meta,
            "mean_entropy": summary.get("mean_entropy"),
            "mean_distance": summary.get("mean_distance"),
            "per_head": per_head_table(summary),
            "layers_plotted": [early, late],
        }
        del model

    before = report["models"]["pretrained"]["per_head"]
    after = report["models"]["finetuned"]["per_head"]
    deltas = {
        head: {
            "entropy_delta": round(
                after[head]["entropy"] - before[head]["entropy"], 4),
            "distance_delta": round(
                after[head]["mean_distance"] - before[head]["mean_distance"], 4),
        }
        for head in before if head in after
    }
    report["per_head_delta"] = deltas
    # The heads that moved most are what the written analysis should discuss;
    # ranking them here means the report quotes a measurement, not an impression.
    report["largest_entropy_shifts"] = sorted(
        deltas.items(), key=lambda kv: abs(kv[1]["entropy_delta"]), reverse=True
    )[:8]
    report["largest_distance_shifts"] = sorted(
        deltas.items(), key=lambda kv: abs(kv[1]["distance_delta"]), reverse=True
    )[:8]

    path = viz.plot_attention_summary(
        summaries, figures / f"{LANGUAGE}_attention_pretrained_vs_finetuned.png"
    )
    report["figures"].append(str(path.name))
    return report


def main() -> None:
    """Command-line entry point for the attention comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--finetuned", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--tokenizer-model", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fig-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = compare(
        args.pretrained, args.finetuned, data_dir=args.data_dir,
        tokenizer_model=args.tokenizer_model, split=args.split,
        batch_size=args.batch_size, device=args.device, fig_dir=args.fig_dir,
    )
    out = Path(args.out) if args.out else (
        REPO_ROOT / f"report/phase3/{LANGUAGE}_attention_comparison.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    print(f"  {LANGUAGE}: {report['prompts_compared']} prompts, "
          f"T={report['sequence_length']}")
    for name, block in report["models"].items():
        print(f"    {name:11s} mean entropy {block['mean_entropy']}  "
              f"mean distance {block['mean_distance']}")
    print("    largest entropy shifts:")
    for head, delta in report["largest_entropy_shifts"][:4]:
        print(f"      {head}: {delta['entropy_delta']:+.4f} nats, "
              f"{delta['distance_delta']:+.3f} positions")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()

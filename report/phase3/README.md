# Phase 3 artifacts — index

Everything Phase 3 produced, and where each required deliverable lives. The main
write-up is **[phase3_report.md](phase3_report.md)**.

## Deliverables

| # | Required deliverable | Where |
|---|---|---|
| 1 | Reasoning data generation scripts, finetune scripts, configs | [`hindi/reasoning/`](../../hindi/reasoning/) · [`nepali/reasoning/`](../../nepali/reasoning/) · [`hindi/configs/finetune.yaml`](../../hindi/configs/finetune.yaml) · [`nepali/configs/finetune.yaml`](../../nepali/configs/finetune.yaml) |
| 2 | Finetuned checkpoints (H and L) via Drive links | [main README → Checkpoints](../../README.md#checkpoints-google-drive) |
| 3 | Finetuning logs, metrics, qualitative examples | [`hindi_finetune_log.jsonl`](hindi_finetune_log.jsonl) · [`nepali_finetune_log.jsonl`](nepali_finetune_log.jsonl) · the four `*_reasoning_*.json` files · [report §3–§4](phase3_report.md) |
| 4 | Pretrain vs. finetune attention comparison | [report §5](phase3_report.md) · [`figures/`](figures/) · [`hindi_attention_comparison.json`](hindi_attention_comparison.json) · [`nepali_attention_comparison.json`](nepali_attention_comparison.json) |
| 5 | Final comparison tables and error analysis | [report §3, §4, §6](phase3_report.md) |
| 6 | Complete final report and reproduction instructions | [phase3_report.md](phase3_report.md) · [§9 Reproduction](phase3_report.md#9-reproduction) |

## Files

### Metric reports

| File | What it holds |
|---|---|
| `hindi_reasoning_pretrained.json` | Model H scored on the reasoning test set **before** finetuning |
| `hindi_reasoning_finetuned.json` | Model H after finetuning, same code path |
| `nepali_reasoning_pretrained.json` | Model L before finetuning |
| `nepali_reasoning_finetuned.json` | Model L after finetuning |

Each contains exact-match and forced-choice accuracy (overall, by task family, by
hop count, by attribute), the computed chance rate, answer-token loss and
perplexity, tokenized sequence statistics, the `error_analysis` block that
records which prompt position each answer came from, and worked qualitative
examples of both successes and failures.

### Attention

| File | What it holds |
|---|---|
| `hindi_attention_comparison.json` | Per-head entropy and mean distance for Model H, pretrained and finetuned, on identical prompts, plus per-head deltas ranked by magnitude |
| `nepali_attention_comparison.json` | The same for Model L |

### Training

| File | What it holds |
|---|---|
| `hindi_finetune_log.jsonl` | Model H's finetuning log — train loss, validation loss every 20 steps, learning rate |
| `nepali_finetune_log.jsonl` | Model L's finetuning log |
| `hindi_finetune_config.yaml` | The exact resolved config the run used |
| `nepali_finetune_config.yaml` | The same for Model L |

### Figures

| File | What it shows |
|---|---|
| `figures/finetune_loss_curves_both.png` | Both models' finetuning loss and validation perplexity on shared axes. The overfitting is the point: validation turns upward at step 60 for both. |
| `figures/hindi_pretrained_reasoning_layer0.png` | Model H, early layer, 4 heads, **before** finetuning |
| `figures/hindi_finetuned_reasoning_layer0.png` | Model H, early layer, **after** |
| `figures/hindi_pretrained_reasoning_layer5.png` | Model H, late layer, before |
| `figures/hindi_finetuned_reasoning_layer5.png` | Model H, late layer, after |
| `figures/nepali_*_reasoning_layer{0,5}.png` | The same four for Model L |
| `figures/hindi_attention_pretrained_vs_finetuned.png` | Per-layer entropy and distance summary, both checkpoints |
| `figures/nepali_attention_pretrained_vs_finetuned.png` | The same for Model L |

All heatmaps use Devanagari token labels rather than position indices, and show
query position against key position with a probability colour scale. Every panel
is strictly lower-triangular, which is the visual form of the causal mask.

### Superseded runs

`run_a_40names/` holds the first finetuning run, which selected its checkpoint on
a 100-step evaluation grid. It is kept because the report refers to it: moving to
a 20-step grid, with nothing else changed, raised Model L from 14.90% to 29.45%
exact match. The conclusions did not change, but the numbers did, and the coarse
grid understated both models.

## Reading order

1. **[phase3_report.md §0](phase3_report.md)** — what was and was not learned, in
   five paragraphs.
2. **§3** — the pretrained vs. finetuned tables, and why two metrics are needed.
3. **§4** — the error analysis. This is where the result is actually explained.
4. **§7** — the four questions the specification asks.

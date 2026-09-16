# Bonus ablation — Model H trained without positional embeddings

**One variable changed: `pos_encoding: learned` → `pos_encoding: none`.**

Everything else — corpus, tokenizer, vocabulary, depth, width, head count,
feed-forward size, dropout, weight tying, learning rate, schedule shape, warmup,
weight decay, betas, gradient clipping, batch size, gradient accumulation,
evaluation cadence and **the random seed** — is identical to the standard
Model H run. The seed matters as much as the hyperparameters: the data loader is
seeded from it, so both runs saw the same batches in the same order.

Hindi was chosen over Nepali because its tokenizer has byte fallback and a
measured **0.000000** unknown-token rate, so nothing in the result below can be
attributed to `<unk>` noise in the corpus.

| | Standard | Ablated | Δ |
|---|---:|---:|---:|
| Positional embeddings | learned absolute | **none** | — |
| Position table parameters | 262,144 | **0** | −262,144 |
| Total parameters | 23,273,472 | 23,011,328 | −1.1% |
| **Non-embedding parameters** | **18,915,328** | **18,915,328** | **identical** |

The Transformer stack being compared is exactly the same size. The only thing
removed is the model's access to position.

## Setup

* Config: [`hindi/configs/model_nopos.yaml`](../../hindi/configs/model_nopos.yaml)
* Notebook: [`notebooks/kaggle_ablation.ipynb`](../../notebooks/kaggle_ablation.ipynb),
  **generated** from `kaggle_pretrain.ipynb` by
  [`scripts/make_ablation_notebook.py`](../../scripts/make_ablation_notebook.py)
  so the ablation cannot drift from the code path it is compared against. The
  generator patches three lines and asserts it found all of them.
* Comparison: [`scripts/compare_ablation.py`](../../scripts/compare_ablation.py)
* Kernel: `swaraj001/lma-bonus-ablation-no-pos-model-h-hindi`

`max_steps` was left at 15,000 even though the run was wall-clock limited,
because `max_steps` sets the cosine schedule's horizon. Shortening it would have
decayed the learning rate faster and confounded the ablation with a
learning-rate change. As it happened the run was allocated a much faster GPU
than the baseline had — **41.6k tokens/sec against 12.0k** — and finished the
full 15,000 steps in 3.35 hours, where the baseline stopped at 14,520 when its
11-hour budget fired.

**That asymmetry runs in the ablation's favour**, which is the safe direction: the
ablated model is if anything the better-trained of the two, so the degradation
below cannot be a training-length artefact. The headline comparison is
nevertheless made at a matched step.

## Result 1 — language modelling

At **step 14,500**, the last validation checkpoint both runs reached:

| | Standard | Ablated | Δ |
|---|---:|---:|---|
| Validation loss (nats) | 3.3402 | 3.4047 | +1.9% |
| Validation perplexity | 28.22 | 30.11 | +6.7% |

On the held-out **test** split:

| | Standard | Ablated | Δ |
|---|---:|---:|---|
| Cross-entropy (nats) | 3.3109 | 3.3677 | +1.7% |
| Perplexity | 27.41 | 29.01 | +5.8% |
| Bits-per-byte | 0.5382 | 0.5474 | +1.7% |

## Result 2 — attention

| | Standard | Ablated | Uniform baseline |
|---|---:|---:|---:|
| Mean entropy (nats) | 3.2047 | **3.6005** | 5.2565 |
| Mean distance (tokens) | 62.36 | **77.90** | — |
| Local-head fraction | 0.0625 | **0.2083** | — |

Sharpest head per layer, entropy @ mean distance:

| Layer | Standard | Ablated |
|---|---|---|
| 0 | 4.16 @ 45.8 | 4.56 @ 117.6 |
| 1 | 3.09 @ 21.9 | 3.92 @ 90.6 |
| 2 | 1.49 @ 7.6 | 3.34 @ 14.9 |
| 3 | **0.06 @ 1.3** | 1.11 @ 1.8 |
| 4 | 1.49 @ 22.4 | **0.11 @ 1.2** |
| 5 | 2.99 @ 151.4 | 2.06 @ 16.1 |

Layer-mean attention distance:

| | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---:|---:|---:|---:|---:|---:|
| Standard | 99.6 | 77.0 | 22.2 | 20.5 | 49.4 | 105.5 |
| Ablated | 123.3 | 123.0 | 105.1 | 5.1 | 30.7 | 80.3 |

## Result 3 — generation

| Decoding | chrF++ std | chrF++ abl | BLEU std | BLEU abl | repetition std | repetition abl |
|---|---:|---:|---:|---:|---:|---:|
| greedy | 12.49 | 12.73 | 1.96 | 2.08 | 0.598 | 0.598 |
| T = 0.5 | 14.33 | 15.14 | 1.80 | 2.13 | 0.212 | 0.269 |
| T = 1.0 | 16.29 | 17.07 | 1.01 | 1.64 | 0.002 | 0.002 |
| T = 1.5 | 14.67 | 14.96 | 0.16 | 0.25 | 0.000 | 0.000 |

## Figures

| File | Shows |
|---|---|
| [`figures/ablation_loss_curves_by_step.png`](figures/ablation_loss_curves_by_step.png) | Both runs' train and validation loss against step |
| [`figures/ablation_loss_curves_by_tokens.png`](figures/ablation_loss_curves_by_tokens.png) | The same against tokens consumed |
| [`figures/hindi_nopos_attention_layer0.png`](figures/hindi_nopos_attention_layer0.png) | Ablated model, early layer, 4 heads |
| [`figures/hindi_nopos_attention_layer5.png`](figures/hindi_nopos_attention_layer5.png) | Ablated model, late layer |
| [`figures/hindi_nopos_attention_summary.png`](figures/hindi_nopos_attention_summary.png) | Per-layer entropy and distance |
| [`figures/hindi_nopos_loss_curve.png`](figures/hindi_nopos_loss_curve.png) | Ablated run alone |

Compare against the standard model's equivalents in
[`report/phase2/figures/`](../phase2/figures/).

## Data files

| File | Holds |
|---|---|
| `hindi_nopos_eval.json` | Full Phase 2 evaluation suite on the ablated model |
| `hindi_nopos_attention.json` | Per-layer, per-head entropy and distance |
| `hindi_nopos_training_log.jsonl` | Per-step train loss, per-500-step validation loss |
| `hindi_nopos_config.yaml` | The exact resolved config the run used |
| `ablation_comparison.json` | Both runs side by side, matched step recorded |

## Reproduction

```bash
python scripts/make_ablation_notebook.py
python scripts/kaggle_push.py --what code
python scripts/kaggle_run.py --lang hindi --mode ablation
python scripts/kaggle_run.py --lang hindi --mode ablation --status
python scripts/kaggle_run.py --lang hindi --mode ablation --fetch
python scripts/compare_ablation.py
```

---

## What breaks without position information — read this critically

> **Note.** The tables above are measurements and are what they are. The
> interpretation in this section is a starting point, not a finished argument;
> it should be rewritten in the author's own words for the final report, and
> must be defensible in the viva.

**The degradation is real but far smaller than the theory alone predicts.**
Self-attention is permutation invariant — it treats its input as a set — so a
model with no positional signal should not be able to distinguish "A is bigger
than B" from "B is bigger than A" at all. The project's own test suite asserts
exactly this property at
[`hindi/tests/test_model.py:191`](../../hindi/tests/test_model.py#L191). Yet
removing positions entirely costs only **5.8% test perplexity**.

The likely reason is that **a causal decoder leaks positional information
through the mask itself**. Query position *t* can attend to exactly *t + 1*
keys, so the *number* of visible tokens is itself a positional signal, and a
model with no position table can still recover an approximate position from it.
The permutation-invariance argument applies to unmasked self-attention; causal
masking breaks the symmetry it depends on.

**The attention statistics are where the cost actually shows.** The ablated
model is markedly less selective — mean entropy rises from 3.2047 to 3.6005
against a uniform baseline of 5.2565, so it closes about 19% of the distance
back toward simply averaging its context. It also attends 25% further on average
(77.90 vs 62.36 tokens), which is what a model that cannot localise precisely
would be expected to do.

**The most interesting single result is that the previous-token head survives
but moves.** The standard model builds a near-perfect previous-token head at
layer 3 — entropy 0.06 at distance 1.3, essentially one-hot on the immediately
preceding token. The ablated model builds one too, at **entropy 0.11 and
distance 1.2**, but at **layer 4** rather than layer 3. Layers 0–2 stay diffuse
much longer (layer-mean distances 123.3 / 123.0 / 105.1 against the standard
model's 99.6 / 77.0 / 22.2). A reading consistent with the mask-leakage
explanation is that the ablated model must first *infer* position from the
causal structure before it can build anything that depends on position, and that
inference consumes roughly one extra layer of depth.

**The generation metrics do not show the damage, and that is a caution about the
metrics rather than a result about the model.** chrF++ is slightly *better* for
the ablated model at every decoding setting. chrF++ scores character n-gram
overlap, which is largely insensitive to word order — precisely the thing the
ablated model has lost. A metric that cannot see the defect will not report it.
Note also that the ablated checkpoint had 500 more steps of training, which is
not controlled for in that table.

**Honest limitations.**

* One ablated run, one seed. The differences above are not bracketed by
  run-to-run variance.
* Only Hindi was ablated, as the bonus permits. Whether the same holds for
  Nepali's coarser tokenizer is untested.
* The mask-leakage explanation is consistent with the evidence here but is not
  established by it. The direct test would be to ablate the causal mask's
  length signal, or to probe whether position is linearly decodable from hidden
  states in the ablated model.
* The two runs used different GPUs, so wall-clock and throughput are not
  comparable between them. Step counts, token counts, data order and every
  hyperparameter are.

# Phase 2 — Model Implementation, Pretraining, and Evaluation

Deliverables index and requirement-by-requirement compliance record for
Phase 2 (40 marks). Model H is **Hindi**, Model L is **Nepali**.

> **Status:** code complete and tested; pretraining runs pending.
> Rows marked *pending run* are produced by
> [`notebooks/kaggle_pretrain.ipynb`](../../notebooks/kaggle_pretrain.ipynb)
> and land in this directory once each model has trained.

---

## 0. Repository layout

Per the spec's Repository layout guideline — "Organize the repo by language, with
each language directory self-contained" — Phase 2's code lives **inside each
language directory**, not in a shared package:

```
hindi/                          nepali/
  configs/model.yaml              configs/model.yaml
  model/config.py                 model/config.py
  model/transformer.py            model/transformer.py
  data/dataset.py                 data/dataset.py
  data/pack_corpus.py             data/pack_corpus.py
  train/train.py                  train/train.py
  eval/evaluate.py                eval/evaluate.py
  eval/attention.py               eval/attention.py
  eval/viz.py                     eval/viz.py
  tests/test_model.py             tests/test_model.py
  tokenizer/                      tokenizer/      (Phase 1)
```

The two trees are independent Python packages. Neither imports the other, and
`grep -r "import nepali" hindi/` and its mirror both return nothing. Each is
runnable on its own: `python -m hindi.train.train` never touches `nepali/`.

Within each tree the spec's "separate data, model, training, evaluation, and
visualization code" guideline is honoured by the `data/` / `model/` / `train/` /
`eval/` split, with `eval/viz.py` holding plotting only and never computing a
metric.

## 1. The eight required deliverables

| # | Deliverable (spec §2 Deliverables) | Where |
|---|---|---|
| 1 | Transformer implementation (MHA, positional embeddings, causal mask) | [`hindi/model/transformer.py`](../../hindi/model/transformer.py), [`nepali/model/transformer.py`](../../nepali/model/transformer.py) |
| 2 | Model configuration files (H and L) | [`hindi/configs/model.yaml`](../../hindi/configs/model.yaml), [`nepali/configs/model.yaml`](../../nepali/configs/model.yaml) |
| 3 | Parameter counts; training scripts; checkpoint links | counts below; [`hindi/train/train.py`](../../hindi/train/train.py), [`nepali/train/train.py`](../../nepali/train/train.py); Drive links in the root README |
| 4 | Training logs and loss curves | `*_training_log.jsonl`, `figures/*_loss_curve.png` *(pending run)* |
| 5 | PPL / BPB tables; BLEU, chrF, ROUGE-L | `*_eval.json`, tables below *(pending run)* |
| 6 | Generated samples and diversity / repetition stats | `*_eval.json` → `generation.samples` *(pending run)* |
| 7 | Attention heatmaps plus entropy / distance summaries | `figures/*_attention_*.png`, `*_attention.json` *(pending run)* |
| 8 | Resource-level comparison write-up | `phase2_report.md` *(pending run)* |

## 2. Architecture requirements (spec §2.1)

Every stage the spec enumerates, and where it is implemented. All of it is
built from `nn.Linear`, `nn.Embedding`, `nn.Dropout` and raw tensor ops — no
`nn.Transformer*`, no `nn.MultiheadAttention`, no `F.scaled_dot_product_attention`,
no HuggingFace model class. `<lang>/tests/test_model.py::test_no_prebuilt_attention_modules_used`
walks the module tree and fails if any of those ever appear. Both languages'
suites pass independently: **9/9 for Hindi, 9/9 for Nepali**.

| Required stage | Implementation |
|---|---|
| Token embeddings `(V × d)` | `GPT.token_embedding` |
| Positional embeddings, with the max-length consequence stated | `GPT.position_embedding` (learned absolute); `block_size` is a hard ceiling — see `GPT` docstring |
| Combine by addition + embedding dropout | `GPT.forward` |
| Q/K/V projections | `MultiHeadCausalSelfAttention.W_q/W_k/W_v` — kept as three separate `nn.Linear`s so they map one-to-one onto the spec's notation |
| Reshape `(B,T,d) → (B,h,T,dk)` and back | `MultiHeadCausalSelfAttention.forward`, commented at each transpose |
| Scaled dot-product with `1/√dk`, and why | `MultiHeadCausalSelfAttention` docstring |
| Causal mask, self-implemented | `causal_mask` buffer + `masked_fill(-inf)` before softmax |
| **Empirical proof the model cannot see the future** | `<lang>/tests/test_model.py::test_causal_mask_blocks_the_future` |
| Concatenate heads + output projection `W_O` | `MultiHeadCausalSelfAttention.W_o` |
| FFN, two linears, non-linearity, wider inner dim | `FeedForward` (GELU, `d_ff = 4d`) |
| Residual connections on both sublayers | `TransformerBlock.forward` |
| LayerNorm, stating pre- vs post-norm and why | `TransformerBlock` docstring (pre-norm) |
| Dropout in the appropriate places | attention probabilities, sublayer outputs, embeddings |
| Final normalisation after N blocks | `GPT.ln_f` |
| Linear head to vocab; weight tying stated with saving | `GPT.lm_head`, tied; saves 4,096,000 (H) / 5,120,000 (L) parameters |
| Cross-entropy at `t` against token `t+1` | `GPT.forward` |

### Causal-mask verification result

The spec asks to "show that changing token *t+1* does not change the logits at
position *t*". Measured across every position of a full-length sequence:

```
max logit leakage = 0.000e+00
```

Exactly zero, not merely within tolerance. `softmax` of `-inf` is exactly `0`,
and `0 × v = 0` exactly in IEEE arithmetic, so masked positions contribute
nothing at all. The test also asserts the converse — that the perturbation *does*
change logits at positions ≥ *t+1* — so it cannot pass on a model that simply
ignores its input.

## 3. Configuration and parameter counts (spec §2.1, "Configuration and reporting")

| | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| Vocabulary (from Phase 1) | 8,000 (Unigram) | 10,000 (BPE) |
| Layers / heads / `d_model` / `d_ff` | 6 / 8 / 512 / 2048 | 6 / 8 / 512 / 2048 |
| Context length | 512 | 512 |
| **Total parameters** | **23,273,472** | **24,297,472** |
| Embedding parameters | 4,358,144 | 5,382,144 |
| **Non-embedding parameters** | **18,915,328** | **18,915,328** |
| Weight tying | yes | yes |
| Positional encoding | learned absolute | learned absolute |
| Token budget | 491.5M | 491.5M |

**Depth/width justification.** At `d_model=512` each block costs ~3.15M
parameters, so six blocks plus the embedding table lands just under the ~25M
target; a seventh block would overshoot to 26.4M. Depth was spent on six blocks
rather than a wider four-block model because composing multi-hop dependencies
needs several rounds of attention, while widening grows the feed-forward cost
quadratically for the same budget.

**The two models are deliberately identical in every dimension except
vocabulary.** Their non-embedding parameter counts are exactly equal
(18,915,328) and their token budgets are exactly equal (491.5M), so any
difference in results is attributable to data scale, data quality and tokenizer
behaviour rather than to model capacity — which is precisely what Phase 3's
final report is asked to explain.

## 4. Independence (spec Hard constraints)

| Requirement | How it is enforced |
|---|---|
| Separate datasets | Separate packed streams, separate Kaggle datasets; the notebook **raises** if both are attached |
| Separate tokenizers and vocabularies | Phase 1 artifacts, 8,000 Unigram vs 10,000 BPE, never cross-loaded |
| Separate weights | Separate `out_dir` per language; neither model is initialised from the other |
| Not a multilingual model | One monolingual corpus per run; no shared vocabulary |
| Different seeds | 20260902 (H) / 20260903 (L), so the runs are independent draws |

Nothing at all is shared at runtime — not even source code. Each language has
its own copy of every module, so the "each language directory self-contained"
layout guideline holds literally: `hindi/` and `nepali/` are independent
packages that can be run, read or graded without reference to each other.

## 5. Checkpoint resume (spec §2.2 — mandatory)

Each checkpoint contains model weights, optimizer state, scheduler state,
current step, tokens consumed, best validation loss, the full run
configuration, and both RNG states.

Verified across three consecutive sessions (12 → 18 → 24 steps): training
resumed at the exact step with optimizer and scheduler state intact.

One deliberate exception to "restore everything": the learning-rate schedule
restores only its *progress*, taking its shape from the live config. A resume
that extends `max_steps` — the normal case after a Kaggle session is cut off —
must follow the new horizon; restoring the old one would leave
`step > max_steps` and pin the learning rate at `min_lr` for the whole
extension, which looks exactly like training silently stalling. Any difference
is printed rather than applied quietly.

## 6. Packed corpora used for pretraining

Produced by each language's own packer -
[`hindi/data/pack_corpus.py`](../../hindi/data/pack_corpus.py) and
[`nepali/data/pack_corpus.py`](../../nepali/data/pack_corpus.py) - from that
language's Phase 1 splits, with one EOS token appended per document.

| Split | Hindi tokens | Nepali tokens |
|---|---:|---:|
| train | 485,956,678 | 521,426,359 |
| validation | 10,498,722 | 58,038,824 |
| test | 10,100,729 | 57,657,415 |
| **total** | **506,556,129** | **637,122,598** |

These exceed the Phase 1 report's counts by exactly one token per document —
the EOS marker that lets the model learn document boundaries in a concatenated
stream.

### A measured correction to Phase 1's unknown-token rate

Phase 1 reported Nepali `test_unk_rate = 0.0`. Re-measuring during packing, by
counting `unk_id` occurrences across every packed token, gives:

| | Hindi | Nepali |
|---|---:|---:|
| Unknown-token rate (train) | 0.000000 | 0.001646 |
| Unknown-token rate (test) | 0.000000 | 0.001894 |

Hindi's Unigram model carries byte-fallback pieces (`<0x00>`…`<0xFF>`), so it
can represent any input and genuinely reaches zero. Nepali's BPE model has **no
byte fallback**, so characters outside its vocabulary are replaced by `<unk>`.
Decoding the contexts shows what is being lost — almost entirely Latin-script
characters embedded in Nepali text:

```
" ⁇ isaster  ⁇ isk  ⁇ eduction"     ->  "Disaster Risk Reduction"
"परावर्तित (re ⁇ lection)"          ->  "reflection"
"अपवर्तित (re ⁇ raction)"           ->  "refraction"
```

Capital `D`, `R` and lowercase `f` are simply absent from the vocabulary.

**This was not corrected**, for two reasons: the tokenizer is a graded Phase 1
deliverable and changing it now would invalidate that phase's reported
vocabulary and statistics, and at 0.17% of tokens the effect on training is
negligible. It is reported here because it is a real property of Model L's
tokenizer, it slightly inflates Model L's bits-per-byte (an `<unk>` costs
probability mass while the original bytes still count in the denominator), and
it is direct evidence for Phase 3's question about which tokenizer factors most
affected the lower-resource model.

## 7. Results

*Pending the training runs.* This section will carry, for both models side by
side: validation cross-entropy and perplexity, bits-per-byte, BLEU / chrF++ /
ROUGE-L at greedy and temperatures 0.5 / 1.0 / 1.5, repetition rate,
Distinct-1/2, generated samples, and the attention entropy and mean-distance
summaries.

## 8. Reproduction

Every command below names exactly one language. There is no entry point that
runs both, and no module either language shares with the other.

```
# 1. Pack each language's Phase 1 splits into uint16 token streams
python hindi/data/pack_corpus.py
python nepali/data/pack_corpus.py

# 2. Publish the inputs as private Kaggle datasets
python scripts/kaggle_push.py --what all

# 3. Pretrain on Kaggle (GPU P100), once per language
#    notebooks/kaggle_pretrain.ipynb, set LANGUAGE = "hindi" then "nepali"

# 4. Or pretrain anywhere with a GPU - Model H
python -m hindi.train.train  --config hindi/configs/model.yaml
python -m hindi.eval.evaluate --checkpoint checkpoints/hindi/ckpt_best.pt --split test
python hindi/tests/test_model.py

# 5. ...and Model L, entirely separately
python -m nepali.train.train  --config nepali/configs/model.yaml
python -m nepali.eval.evaluate --checkpoint checkpoints/nepali/ckpt_best.pt --split test
python nepali/tests/test_model.py
```

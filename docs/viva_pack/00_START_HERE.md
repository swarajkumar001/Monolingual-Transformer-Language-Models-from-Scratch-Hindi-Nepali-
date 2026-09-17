# START HERE — how to use this pack

## Instructions for Claude

You are helping Swaraj during a live viva (oral exam) for a university course
called Language Models and Agents. An examiner is asking questions about this
project, out loud, right now.

**Answer fast and answer short.** Swaraj is standing in front of an examiner and
cannot read three paragraphs. Give:

1. The direct answer in one or two sentences.
2. The exact file and line number, as `path/file.py:123`.
3. The relevant code, if it is short — five or ten lines, not the whole function.
4. A "say this" sentence he can read out loud, when the question is conceptual.

Do not explain your reasoning, do not offer alternatives, and do not ask
clarifying questions unless the question is genuinely ambiguous. If you do not
know, say so in one line rather than guessing — a confident wrong answer in a
viva is worse than "I would have to check".

All code in files 01–06 is numbered, and those numbers match the repository
exactly, so a line you cite can be opened directly.

---

## The project in one paragraph

Two independent monolingual language models built from scratch, with no
pretrained weights, no pretrained tokenizer and no HuggingFace Transformer
classes. **Model H = Hindi** (higher-resource), **Model L = Nepali**
(lower-resource). Both are 6-layer decoder-only Transformers with identical
architecture and near-identical token budgets, so every difference in results
comes from data and tokenizer rather than capacity. Phase 1 built the corpora
and tokenizers, Phase 2 implemented and pretrained the models, Phase 3
finetuned both on synthetic comparative-reasoning data and compared attention
before and after.

---

## Every headline number

| | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| Tokenizer | SentencePiece **Unigram, 8,000** | SentencePiece **BPE, 10,000** |
| Byte fallback | yes | **no** ← the consequential Phase 1 mistake |
| Unknown-token rate | **0.000000** | 0.001646 |
| Bytes per token | 8.88 | 11.44 |
| Train corpus | 485,956,678 tokens | 521,426,359 tokens |
| Split ratio | 96 / 2 / 2, seed 42 | 85 / 7.5 / 7.5, seed 20260817 |
| Total parameters | 23,273,472 | 24,297,472 |
| **Non-embedding parameters** | **18,915,328** | **18,915,328** (identical) |
| Architecture | 6 layers, 8 heads, d=512, d_ff=2048, ctx 512 | same |
| Training steps | 14,520 (of 15,000) | 14,440 (of 15,000) |
| Training tokens | 475,791,360 | 473,169,920 |
| Wall clock | 11.0 h on a Tesla T4 @ 12.0k tok/s | 11.0 h |
| Best checkpoint | step 14,500 | step 14,000 |
| Test cross-entropy | 3.3109 nats | 3.8433 nats |
| **Test perplexity** | **27.41** | 46.68 |
| **Test bits-per-byte** | 0.5382 | **0.4846** |
| Mean attention entropy | 3.205 | 3.090 |
| Uniform baseline | 5.256 | 5.256 |
| Mean attention distance | 62.4 tokens | 54.5 tokens |
| Reasoning exact match | 0% → **35.15%** | 0% → **29.45%** |
| Reasoning forced choice | 36.35% | 39.30% |
| Chance rate | 41.67% | 41.67% |
| Answer-token loss | 0.9408 | 1.4351 |
| Off-prompt answer rate | 0.0–7.2% | 19.2–28.8% |

---

## Where is X? — the lookup index

### Phase 2: model and training → **file 01**

| Question | Location |
|---|---|
| Multi-head attention | `hindi/model/transformer.py:69` class `MultiHeadCausalSelfAttention` |
| Causal mask built | `transformer.py:123` `torch.triu(..., diagonal=1)`, buffer at `:127` |
| Mask applied | `transformer.py:173` `masked_fill(causal_mask, -inf)` before softmax |
| The 1/√d_k scaling | inside `MultiHeadCausalSelfAttention.forward` |
| Dropout — 4 places | `transformer.py:178` attention probs · `:189` attention out · `:225` FFN out · `:467` embeddings |
| Positional embeddings | `transformer.py:347` (learned) · `:279` sinusoidal · `:355` "none" for the bonus ablation |
| Feed-forward 512→2048→512 GELU | `transformer.py:193` class `FeedForward`, one-liner at `:225` |
| Weight tying | `transformer.py:365` |
| `generate()` / temperature / top-k | `transformer.py:490` |
| Optimizer, decay only on matrices | `transformer.py:546` `configure_optimizers`, the rule is `param.dim() >= 2` |
| **Training loop** | `hindi/train/train.py:351` `Trainer.train()`, loop at `:378` |
| Gradient accumulation | `train.py:383–390` |
| Unscale then clip | `train.py:396–398` |
| Wall-clock exit (why 14,520 not 15,000) | `train.py:451` |
| LR schedule, warmup + cosine | `train.py:34` `CosineWarmupSchedule`, formula at `:80` |
| **What a checkpoint contains** | `train.py:246` `_checkpoint_payload()` |
| Saving checkpoints | `train.py:270` `save_checkpoint()` |
| Resuming | `train.py:291` `_maybe_resume()` |
| Hyperparameters | `hindi/configs/model.yaml`, `nepali/configs/model.yaml` |

### Phase 2: evaluation and attention → **file 02**

| Question | Location |
|---|---|
| Perplexity and bits-per-byte | `hindi/eval/evaluate.py` |
| BLEU / chrF++ / ROUGE-L, diversity | `hindi/eval/evaluate.py` |
| Attention entropy and distance | `hindi/eval/attention.py` |
| Heatmap and curve plotting | `hindi/eval/viz.py` |
| **Causal masking proof** | `hindi/tests/test_model.py:51` `test_causal_mask_blocks_the_future` |
| Proof no pre-built attention used | `test_model.py:216` `test_no_prebuilt_attention_modules_used` |
| Permutation invariance without positions | `test_model.py:191` `test_positional_encoding_variants` |

### Phase 1: data and tokenizer → **files 03 and 04**

| Question | Location |
|---|---|
| Unicode normalisation, zero-width stripping | `scripts/clean_lib.py` — `normalize_text` |
| Latin-token stripping | `clean_lib.py` — `_PURE_LATIN_TOKEN` |
| Mojibake / malformed detection | `clean_lib.py` — `malformed_ratio` |
| Language ID (3 layers) | `clean_lib.py` — `langdetect_verdict`, `devanagari_ratio`, marker counts |
| Exact dedup (SHA-1) | `clean_lib.py` — `dedup_key` |
| **Hindi train/val/test split** | `hindi/data/pipeline/split_hindi.py` — 96/2/2, seed 42, shuffle |
| **Nepali split** | `nepali/scripts/s07_holdout.py` — 85/7.5/7.5, hash of doc_id, held out *before* quality filtering |
| Near-duplicate removal (MinHash) | `nepali/scripts/s08_dedup.py` |
| Train/test decontamination | `nepali/scripts/s09_decontaminate.py` — removes only from train |
| Tokenizer training | `hindi/tokenizer/train_tokenizer.py` · `nepali/scripts/s12_train_tokenizer.py` |
| Fertility / bytes-per-token analysis | `hindi/tokenizer/analyze_tokenizer.py` |

### Phase 3: reasoning → **file 05**

| Question | Location |
|---|---|
| Synthetic data generation | `hindi/reasoning/generate.py` — 4 task families |
| Leakage control (disjoint names + templates) | `generate.py` — pools 40/10/10 and 14/4/4, asserted in `main()` |
| Answer-only loss masking | `hindi/reasoning/dataset.py` — `ignore_index=-1` |
| Finetuning loop | `hindi/reasoning/finetune.py` — class `FineTuner` |
| Exact match vs forced choice | `hindi/reasoning/evaluate_reasoning.py` |
| Position-level error analysis | `evaluate_reasoning.py` — `position_analysis()` |
| Pretrained vs finetuned attention | `hindi/reasoning/attention_compare.py` |

### Reports and every number → **file 06**

---

## The five answers that matter most

**1. Why do perplexity and bits-per-byte disagree?**
> "Perplexity is per token, and the two tokenizers don't produce comparable
> tokens — Nepali packs 11.44 bytes per token against Hindi's 8.88. Bits-per-byte
> divides by the underlying UTF-8 bytes so the tokenizer cancels out. On the
> neutral measure the two models are close, with Model L marginally ahead. Had I
> reported perplexity alone I'd have wrongly concluded the low-resource model was
> far worse."

**2. Why not RoPE?**
> "The model trains and evaluates at one fixed length of 512. RoPE's advantage is
> relative positioning and length extrapolation, and neither buys anything when
> the length never changes. Learned absolute lets the model put resolution where
> the data needs it. If I were extending context, RoPE would be the right choice."

**3. Prove the model can't see the future.**
> "Perturbing every token after position t changes the logits at positions ≤ t by
> exactly 0.000e+00 — not within tolerance, exactly zero, because softmax(-inf) is
> exactly 0. The same test asserts the converse, that it *does* change later
> positions, so it can't pass on a model that ignores its input."
> `hindi/tests/test_model.py:51`

**4. Why is greedy decoding so repetitive?**
> "Argmax is deterministic. If emitting the top token recreates a similar context,
> the model emits it again forever — 58–60% of generated 4-grams are repeats. At
> temperature 1.0 that drops to 0.002, and chrF++ *improves* at the same time, so
> up to T=1.0 the diversity costs no fidelity."

**5. Did finetuning teach reasoning?**
> "No, and that's the honest finding. Exact match went from 0% to 35.15% and
> 29.45%, but forced-choice accuracy is 36.35% and 39.30% against a chance rate of
> 41.67% — at or below chance. Finetuning taught the output format, not inference.
> The error analysis shows why: on two-hop questions Model H answers the middle
> entity 72.6% of the time, and that's the only name mentioned twice. It's
> salience, not reasoning."

---

## Hard constraints, if challenged

No pretrained models, no pretrained tokenizers, no HuggingFace Transformer
classes. Positional embeddings, multi-head attention and causal masking are all
student-implemented from `nn.Linear`, `nn.Embedding`, `nn.Dropout` and raw
tensor ops — verified by a test that walks the module tree and fails if
`nn.Transformer*`, `nn.MultiheadAttention` or `F.scaled_dot_product_attention`
ever appear. The two languages share source code only: separate corpora,
separate tokenizers, separate vocabularies, separate weights, and no
multilingual training.

## Known weaknesses — admit these, don't hide them

- Both runs stopped short of 15,000 steps (14,520 / 14,440) when an 11-hour
  session budget fired. Reported as-run.
- Nepali's BPE has no byte fallback, leaving an irreducible 0.17% `<unk>` rate.
- Forced-choice reasoning accuracy is below chance for both models.
- The optional no-positional-embeddings ablation was not run — the two
  pretraining runs consumed the 30-hour weekly GPU quota.

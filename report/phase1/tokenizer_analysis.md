# Phase 1 — Tokenizer Analysis

Both languages use a from-scratch SentencePiece **unigram** tokenizer, trained
separately per language on that language's own `data/splits/train/` only
(never validation or test — held-out text is reserved for the measurements
below). Fixed settings across the vocab-size sweep: `character_coverage=0.9995`,
`byte_fallback=True`, `split_digits=True`, `normalization_rule_name=identity`,
`input_sentence_size=2,000,000`, `shuffle_input_sentence=True`.

**Vocab range: 5,000-10,000** (swept at 5k/6k/8k/10k), per a course-TA ruling
— see "Embedding parameter budget" below for why. An earlier pass of this
sweep explored 12k/16k/24k/32k before that ruling; those runs are reported
as a footnote for context but are **not valid candidates** and were dropped.

All numbers below come from `<lang>/tokenizer/analyze_tokenizer.py` run on
each language's `data/splits/validation/` split (2,000 docs sampled) and
`scripts/count_tokens.py` run over the full corpus (train+validation+test,
every document, exact — not sampled/extrapolated). A cell reads
**NOT YET MEASURED** where the corresponding sweep step has not completed
yet; no number in this file is estimated or backfilled from a placeholder
ratio.

**Do not use 3.2 chars/token anywhere in this project** — that was an early
placeholder before any tokenizer existed. The real, measured ratio differs
by vocab size (see table below) and is what all downstream token-budget
planning uses.

---

## Embedding parameter budget

At a ~25M-parameter total budget, vocabulary size directly trades off
against how many parameters are left for the actual Transformer layers. If
the input and output projections are tied (one embedding matrix reused for
both, as this project does — see the model config in Phase 2), the
embedding table costs `vocab_size × model_dim` parameters, independent of
depth/head count.

The course TA's ruling ("a 24k vocab makes the embedding table about 49% of
the model") corresponds almost exactly to `model_dim=512`:
24,000 × 512 = 12,288,000 = **49.15%** of 25,000,000. The table below uses
that same `model_dim=512` reference point (Phase 2 has not finalized the
exact architecture yet, but 512 is a plausible dimension for a ~25M-param
model and is what reproduces the TA's own cited figure) alongside 256 and
384 for context:

| Vocab size | Embed params (d=256) | % of 25M | Embed params (d=384) | % of 25M | Embed params (d=512) | % of 25M |
|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 1,280,000 | 5.12% | 1,920,000 | 7.68% | 2,560,000 | 10.24% |
| 6,000 | 1,536,000 | 6.14% | 2,304,000 | 9.22% | 3,072,000 | 12.29% |
| 8,000 | 2,048,000 | 8.19% | 3,072,000 | 12.29% | 4,096,000 | 16.38% |
| 10,000 | 2,560,000 | 10.24% | 3,840,000 | 15.36% | 5,120,000 | 20.48% |
| ~~24,000~~ (dropped) | 6,144,000 | 24.58% | 9,216,000 | 36.86% | 12,288,000 | **49.15%** |

Even at the low end of this range (5k, d=512), the embedding table is still
~10% of the model; at the top of the allowed range (10k, d=512) it reaches
~20%. Dropping to d=256 roughly halves every figure. Either way, staying at
or below 10k keeps the embedding table a minority share of the budget across
all plausible model_dim choices, whereas 24k pushes it to roughly half the
model regardless of dimension — the actual Transformer stack (attention +
FFN layers) would be left with less capacity than the vocabulary lookup
table itself. If embeddings are **not** tied (separate input/output
projections), every figure above roughly doubles, making the case for a
small vocab even stronger.

---

## Bhojpuri (Model L)

| Vocab size | chars/token (val sample) | chars/token (exact, full corpus) | fertility (tok/word) | UNK rate | byte-fallback rate | vocab utilization | Full-corpus tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED |
| 6,000 | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED |
| 8,000 | 3.3900 | 3.3582 | **1.4814** | 0.000000 | 0.001785 | 0.8776 | **84,217,032** |
| 10,000 | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED |

Measured on 2,000 held-out validation documents (except full-corpus tokens,
which is exact over all splits, no sampling).

**vocab_size=8000 detail:**

Full-corpus exact count (`scripts/count_tokens.py`, every document in
train+validation+test, no sampling):

| Split | Docs | Characters | Tokens |
|---|---:|---:|---:|
| train | 487,674 | 271,000,251 | 80,610,590 |
| validation | 10,160 | 6,632,160 | 2,085,234 |
| test | 10,160 | 5,182,766 | 1,521,208 |
| **Total** | **507,994** | **282,815,177** | **84,217,032** |

- Manual tokens: 25,169,871. Downloaded tokens: 59,047,161. **Manual share
  by token count: 29.89%** — clears the 20% floor at the token level too,
  not just the char level `validate_phase1.py` already checks.
- 84.2M tokens total, against the ~500M-token aspirational scale mentioned
  earlier in planning — confirms Bhojpuri's documented, spec-permitted
  shortfall (all known manual sources are exhausted; see
  `dataset_statistics.md`).
- UNK rate is 0.0 by construction with `byte_fallback=True` — every
  otherwise-unrepresentable character falls back to a raw-byte piece
  instead of `<unk>`, so it is reported here only for completeness and is
  **not** the signal to read for "how much text this vocab struggles with."
  The byte-fallback rate (0.18%) is the metric that actually plays that
  role, and it is low here — the vocab already covers the vast majority of
  observed characters directly (character_coverage came out to 99.95% at
  training time, matching the 0.9995 target).
- Top tokens are dominated by common Devanagari function words/postpositions
  (▁के, ▁में, ▁से, ▁का, ▁पर) and punctuation (।, `,`, `.`, `-`), as expected
  for a unigram model on natural text.
- Sample tokenizations (piece + id) confirm sensible subword segmentation on
  real Bhojpuri text, e.g. `कांग्रेसियन` → `['▁कांग्रेस', 'ियन']`,
  `निश्चिंत` → `['▁नि', 'श्च', 'ि', 'ंत']`.

**Vocab-size choice:** not yet decided — needs 5k/6k/10k measured to compare
against 8k on fertility and byte-fallback rate, combined with the parameter
budget above. To be filled in once the (relaunched, narrowed) sweep
completes.

### Footnote: pre-ruling exploration (12k/16k/24k, dropped)

Before the TA's vocab-budget ruling, this sweep explored larger sizes. Kept
here only as context for why the range narrowed, **not** as valid
candidates:

| Vocab size | chars/token (exact) | Full-corpus tokens | Embed % of 25M (d=512, tied) |
|---:|---:|---:|---:|
| 12,000 | 3.5348 | 80,008,429 | 24.58% |
| 16,000 | 3.6434 | 77,623,979 | 32.77% |
| 24,000 | not completed (training stopped when the range narrowed) | — | 49.15% |

Moving from 16k down to 8k raised the full-corpus token count from
77,623,979 to 84,217,032 — about **+8.5%** — consistent with the general
expectation that a smaller vocab produces more tokens for the same text.

## Hindi (Model H)

**NOT YET MEASURED — no rows.** Hindi tokenizer training has not started.
It is blocked on `clean_hindi.py` (full non-resume run, in progress) and the
subsequent `split_hindi.py` rerun. The tokenizer scripts themselves
(`hindi/tokenizer/train_tokenizer.py`, `analyze_tokenizer.py`, mirroring the
Bhojpuri versions exactly with Hindi paths/model-name prefix) are written
and ready; `run_hindi_tokenizer_sweep.sh` will run the same 5k/6k/8k/10k
sweep the moment the split is available.

---

## Chosen vocab size (final)

**NOT YET DECIDED for either language.** Will be filled in once each
language's narrowed sweep completes, with justification from byte-fallback
rate and fertility on held-out text (course spec) combined with the
embedding parameter budget above (TA ruling).

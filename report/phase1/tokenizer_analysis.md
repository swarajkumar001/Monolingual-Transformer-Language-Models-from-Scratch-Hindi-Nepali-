# Phase 1 — Tokenizer Analysis

Both languages use a from-scratch SentencePiece **unigram** tokenizer, trained
separately per language on that language's own `data/splits/train/` only
(never validation or test — held-out text is reserved for the measurements
below). Fixed settings across the vocab-size sweep (8k/12k/16k/24k/32k):
`character_coverage=0.9995`, `byte_fallback=True`, `split_digits=True`,
`normalization_rule_name=identity`, `input_sentence_size=2,000,000`,
`shuffle_input_sentence=True`.

All numbers below come from `<lang>/tokenizer/analyze_tokenizer.py` run on
each language's `data/splits/validation/` split (2,000 docs sampled) and
`scripts/count_tokens.py` run over the full corpus (train+validation+test,
every document, exact — not sampled/extrapolated). A cell reads
**NOT YET MEASURED** where the corresponding sweep step has not completed
yet; no number in this file is estimated or backfilled from a placeholder
ratio.

**Do not use 3.2 chars/token anywhere in this project** — that was an early
placeholder before any tokenizer existed. The real ratio is measured
per-vocab-size below and differs by vocab size (e.g. 8k measures at 3.39;
16k was separately reported at 3.96 in an earlier run of this same
pipeline — this file will carry the fresh in-sweep measurement for 16k once
that step completes, not the remembered figure).

---

## Bhojpuri (Model L)

| Vocab size | chars/token (val sample) | fertility (tok/word) | UNK rate | byte-fallback rate | vocab utilization | Full-corpus tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 8,000 | 3.3900 | **1.4814** | 0.000000 | 0.001785 | 0.8776 | **84,217,032** |
| 12,000 | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED |
| 16,000 | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED |
| 24,000 | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED |
| 32,000 | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED | NOT YET MEASURED |

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

- Exact whole-corpus chars/token: **3.3582** (vs. 3.3900 measured on the
  2,000-doc validation sample above — close, as expected).
- Manual tokens: 25,169,871. Downloaded tokens: 59,047,161. **Manual share
  by token count: 29.89%** — clears the 20% floor at the token level too,
  not just the char level `validate_phase1.py` already checks.
- 84.2M tokens total, against the ~500M-token aspirational scale mentioned
  earlier in planning — confirms Bhojpuri's documented, spec-permitted
  shortfall (all known manual sources are exhausted; see
  `dataset_statistics.md`). This number will shift slightly across vocab
  sizes (larger vocab → fewer tokens for the same text) but not by an
  order of magnitude.

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

**Vocab-size choice:** not yet decided — needs all 5 sizes' byte-fallback
rate and fertility to compare (see spec: choose using fertility / UNK-rate
on held-out text). To be filled in once the sweep completes.

## Hindi (Model H)

**NOT YET MEASURED — no rows.** Hindi tokenizer training has not started.
It is blocked on `clean_hindi.py` (full non-resume run, in progress) and the
subsequent `split_hindi.py` rerun. The tokenizer scripts themselves
(`hindi/tokenizer/train_tokenizer.py`, `analyze_tokenizer.py`, mirroring the
Bhojpuri versions exactly with Hindi paths/model-name prefix) are written
and ready; `run_hindi_tokenizer_sweep.sh` will run the same 8k/12k/16k/24k/32k
sweep the moment the split is available.

---

## Chosen vocab size (final)

**NOT YET DECIDED for either language.** Will be filled in once each
language's full sweep completes, with justification from byte-fallback rate
and fertility on held-out text, per the course spec.

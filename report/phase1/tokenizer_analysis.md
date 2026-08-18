# Phase 1 — Tokenizer Analysis

Both languages use a from-scratch SentencePiece **unigram** tokenizer, trained
separately per language on that language's own `data/splits/train/` only
(never validation or test — held-out text is reserved for the measurements
below). Fixed settings across the vocab-size sweep: `character_coverage=0.9995`,
`byte_fallback=True`, `split_digits=True`, `normalization_rule_name=identity`,
`input_sentence_size=2,000,000`, `shuffle_input_sentence=True`.

**Vocab range: 5,000-10,000** (swept at 5k/6k/8k/10k), per a course-TA ruling
— see "Embedding parameter budget" below for why. An earlier pass of this
sweep explored 12k/16k/24k before that ruling; those are not valid
candidates and their model files have been removed.

All numbers below come from `<lang>/tokenizer/analyze_tokenizer.py` run on
each language's `data/splits/validation/` split (2,000 docs sampled) and
`scripts/count_tokens.py` run over the full corpus (train+validation+test,
every document, exact — not sampled/extrapolated). Every number is taken
directly from `bhojpuri_tokenizer_sweep.log` / the Hindi sweep log. A cell
reads **NOT YET MEASURED** where the corresponding step has not completed;
nothing here is estimated, and no fabricated number (e.g. the 3.2
chars/token placeholder, or the 3.658/3.960/4.159 figures that appeared in
an earlier planning note but don't match this repo's actual measurements)
is used anywhere.

---

## Embedding parameter budget

At a ~25M-parameter total budget, vocabulary size directly trades off
against how many parameters are left for the actual Transformer layers. If
the input and output projections are tied (one embedding matrix reused for
both, as this project does), the embedding table costs
`vocab_size × model_dim` parameters, independent of depth/head count.

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

Even at the low end of this range (5k, d=512), the embedding table is
~10% of the model; at the top (10k, d=512) it reaches ~20%. Staying at or
below 10k keeps the embedding table a clear minority of the budget across
every plausible model_dim, whereas 24k pushes it to roughly half the model
regardless of dimension. If embeddings are **not** tied (separate
input/output projections), every figure above roughly doubles.

---

## Bhojpuri (Model L) — complete

| Vocab size | chars/token (val sample) | chars/token (exact, full corpus) | fertility (tok/word) | UNK rate | byte-fallback rate | vocab utilization | Full-corpus tokens | Manual % (by token) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 3.1314 | 3.1261 | 1.6037 | 0.000000 | 0.001649 | 0.9230 | 90,470,342 | 29.78% |
| 6,000 | 3.2394 | 3.2188 | 1.5502 | 0.000000 | 0.001706 | 0.9122 | 87,862,945 | 29.84% |
| 8,000 | 3.3900 | 3.3582 | 1.4814 | 0.000000 | 0.001785 | 0.8776 | 84,217,032 | 29.89% |
| 10,000 | 3.4935 | 3.4589 | 1.4375 | 0.000000 | 0.001840 | 0.8321 | 81,763,309 | 29.90% |

Measured on 2,000 held-out validation documents (except full-corpus tokens
and manual %, which are exact over all splits, no sampling). Full-corpus
counts, per split, for the chosen size are broken out below.

UNK rate is 0.0 for every size by construction (`byte_fallback=True` sends
unrepresentable text to raw-byte `<0xNN>` pieces instead of `<unk>`) — not a
useful signal, reported only for completeness. **Byte-fallback rate is
almost flat across the whole range (0.165%-0.184%)**, because it's
primarily governed by `character_coverage=0.9995`, a corpus-level setting
independent of target vocab size — it is not the differentiator between
these four candidates.

**Sample tokenizations** (vocab_size=8000, representative of all four —
subword boundaries shift slightly by size but stay linguistically sensible
throughout): `कांग्रेसियन` → `['▁कांग्रेस', 'ियन']`,
`निश्चिंत` → `['▁नि', 'श्च', 'ि', 'ंत']`, full sentence
`कांग्रेस अब आरोप थोपो पार्टी बन कर रह गई है।` →
`['▁कांग्रेस', '▁अब', '▁आरोप', '▁', 'थो', 'पो', '▁पार्टी', '▁बन', '▁कर', '▁रह', '▁गई', '▁है', '।']`
with ids `[680, 346, 1011, 263, 4466, 1263, 707, 451, 337, 639, 3430, 388, 260]`.
Top tokens at every size are dominated by common Devanagari
postpositions/function words (▁के, ▁में, ▁से, ▁का, ▁बा) and punctuation
(।, `,`, `.`, `-`), as expected for a unigram model on natural text.

### Vocab-size choice: **8,000**

Two real trade-offs move in opposite directions across this range:

- **Fertility improves with larger vocab** (5k: 1.604 → 10k: 1.438
  tokens/word), but the improvement is modest — only ~10% relative across
  the whole allowed range.
- **Vocab utilization degrades with larger vocab** (5k: 92.3% → 10k:
  83.2% of the vocab table actually used on held-out text) — at 10k,
  nearly 1 in 6 embedding rows are for pieces the tokenizer rarely or never
  emits, which is capacity spent on the embedding table's parameter count
  without buying real coverage.
- Byte-fallback rate does not meaningfully differentiate the four sizes.
- Embedding budget grows fast (10.24% → 20.48% of 25M at d=512).

8,000 sits at the point where fertility has already captured most of its
available gain (1.481, most of the way from 5k's 1.604 toward 10k's 1.438)
while vocab utilization is still comfortably high (87.8%, well short of
10k's 83.2% floor) and the embedding budget (16.38% at d=512, or 8.19%/
12.29% at d=256/384) stays a clear minority of the 25M total regardless of
the final model_dim chosen in Phase 2. 10k's extra ~3% relative fertility
gain over 8k does not offset its ~9-point utilization drop and larger
embedding footprint.

**Full-corpus counts for the chosen size (vocab_size=8000):**

| Split | Docs | Characters | Tokens |
|---|---:|---:|---:|
| train | 487,674 | 271,000,251 | 80,610,590 |
| validation | 10,160 | 6,632,160 | 2,085,234 |
| test | 10,160 | 5,182,766 | 1,521,208 |
| **Total** | **507,994** | **282,815,177** | **84,217,032** |

84.2M tokens total, against the ~500M-token aspirational scale mentioned
earlier in planning — confirms Bhojpuri's documented, spec-permitted
shortfall (all known manual sources are exhausted; see
`dataset_statistics.md`). Manual share by token count: 29.89% — clears the
20% floor at the token level, not just the char level.

## Hindi (Model H)

Swept at just the two endpoints of the allowed range (5,000 and 10,000),
not the full 5k/6k/8k/10k grid used for Bhojpuri — Hindi's cleaned corpus
is ~6.2x larger by character count, so the range was narrowed to keep
total sweep time reasonable (`run_hindi.bat`). Both trained on the full,
correct train split (675,937 docs; an earlier pair of runs was discarded
after being found to have trained on a truncated merged-corpus file — see
commit history — and retrained from scratch).

| Vocab size | chars/token (val sample) | chars/token (exact, full corpus) | fertility (tok/word) | UNK rate | byte-fallback rate | vocab utilization | Full-corpus tokens | Manual % (by token) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 3.3230 | NOT MEASURED (only the chosen size gets the full-corpus pass) | 1.4961 | 0.000000 | 0.018301 | 0.9654 | NOT MEASURED | NOT MEASURED |
| 10,000 | 3.6601 | **3.5874** | 1.3583 | 0.000000 | 0.020213 | 0.9767 | **491,883,387** | **30.63%** |

Measured on 2,000 held-out validation documents. Full-corpus exact count
was run only for the chosen size (10,000) — see "Vocab-size choice" below
for why only the winner gets the expensive full-corpus pass; 5,000's
full-corpus figures are deliberately left unmeasured rather than
estimated, since it was not the size actually used.

Byte-fallback rate is notably higher for Hindi than Bhojpuri at every
size (1.8-2.0% vs Bhojpuri's ~0.17%) — expected, given Hindi's source mix
is far more heterogeneous (newspapers, Wikipedia, government transcripts,
web text) with more varied punctuation/symbols than Bhojpuri's more
uniform informal-web-text sources. `<0x0A>` (newline) appears as a
top-frequency byte-fallback token at both sizes, consistent with
paragraph-heavy document structure.

**Sample tokenization** (vocab_size=10000):
`अतुल माहेश्वरी के निधन के कई महीनों बाद...` →
`['▁अतुल', '▁माह', 'ेश्वर', 'ी', '▁के', '▁निधन', '▁के', '▁कई', '▁महीनों', '▁बाद', ...]`
— note `▁अतुल` (a full word) tokenizes as one piece at 10k but splits into
`['▁अ', 'तुल']` at 5k, illustrating the fertility difference directly.

### Vocab-size choice: **10,000**

Unlike Bhojpuri, **vocab utilization improves with the larger size here**
(96.54% at 5k → 97.67% at 10k) rather than degrading — Hindi's much larger,
more diverse corpus (1.76B cleaned chars vs Bhojpuri's 283M) has enough
distinct content that even 10k vocab entries stay meaningfully used. That
removes the main argument that pushed Bhojpuri toward a smaller size
(wasted embedding capacity on rarely-used pieces isn't a real cost here).
With utilization not a concern, the remaining trade-offs favor 10k
cleanly: fertility improves by ~9.2% (1.496 → 1.358 tokens/word), and the
byte-fallback rate difference (1.83% → 2.02%) is small enough to plausibly
be measurement noise rather than a real degradation. At the embedding
level (d=512, tied), 10k is 20.48% of the 25M budget — clearly still a
minority share, nowhere near the ~49% that triggered the TA's ruling at
24k, so there is no budget-based reason to prefer 5k over 10k the way
there might be at a higher vocab.

**Full-corpus counts for the chosen size (vocab_size=10000):**

| Split | Docs | Characters | Tokens |
|---|---:|---:|---:|
| train | 675,937 | 1,692,881,268 | 471,882,513 |
| validation | 14,082 | 36,497,051 | 10,194,536 |
| test | 14,082 | 35,218,463 | 9,806,338 |
| **Total** | **704,101** | **1,764,596,782** | **491,883,387** |

**491,883,387 tokens — 98.4% of the ~500M target**, essentially hitting
it. Manual tokens: 150,668,466. Downloaded tokens: 341,214,921. **Manual
share by token count: 30.63%** — comfortably clears the 20% floor at the
token level.

---

## Chosen vocab size (final)

**Bhojpuri: 8,000** — justified above.

**Hindi: 10,000** — justified above.

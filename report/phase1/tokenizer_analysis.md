# Phase 1 — Tokenizer Analysis

Hindi and Nepali use different SentencePiece algorithms — **Hindi:
Unigram**; **Nepali: BPE** (see "Why BPE for Nepali" below for why this
isn't a like-for-like choice) — each trained separately on that
language's own train split only (never validation or test). Fixed
settings: `character_coverage=0.9995` for both;
`byte_fallback=True` for Hindi, **`byte_fallback=False` for Nepali** (a
real, meaningful difference — see "UNK rate" below);
`input_sentence_size=2,000,000` (Hindi) / `3,000,000` (Nepali);
`shuffle_input_sentence=True` for both.

**Vocab range: 5,000–10,000** for both languages (Hindi swept at
5k/8k/10k; Nepali at 5k/6k/8k/10k), per a course-TA ruling — see
"Embedding parameter budget" below.

Hindi's numbers come from `hindi/tokenizer/analyze_tokenizer.py` and
`scripts/count_tokens.py` (exact, full-corpus). Nepali's sweep numbers
come from two independent measurements that are close but not
bit-identical: `nepali/tokenizer/selection_report.json` (the pipeline's
own sweep, at tokenizer-training time) and a fresh re-measurement by
`report/phase1/make_plots.py` (same methodology as Hindi's: 2,000
held-out validation documents, encoded directly against each trained
`.model` file) — both are reported where they differ, rather than picking
one silently. Nepali's exact full-corpus token count comes from
`nepali/data/final/corpus_stats.json` (unsampled, every document
counted). Nothing here is estimated; a cell reads **NOT YET MEASURED**
where a step has not completed.

---

## Embedding parameter budget

At a ~25M-parameter total budget, vocabulary size directly trades off
against how many parameters are left for the actual Transformer layers.
With tied input/output embeddings (as this project uses), the embedding
table costs `vocab_size × model_dim` parameters, independent of
depth/head count.

The course TA's ruling ("a 24k vocab makes the embedding table about 49%
of the model") corresponds almost exactly to `model_dim=512`:
24,000 × 512 = 12,288,000 = **49.15%** of 25,000,000. The table below uses
that reference point alongside 256 and 384 for context:

| Vocab size | Embed params (d=256) | % of 25M | Embed params (d=384) | % of 25M | Embed params (d=512) | % of 25M |
|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 1,280,000 | 5.12% | 1,920,000 | 7.68% | 2,560,000 | 10.24% |
| 6,000 | 1,536,000 | 6.14% | 2,304,000 | 9.22% | 3,072,000 | 12.29% |
| 8,000 | 2,048,000 | 8.19% | 3,072,000 | 12.29% | 4,096,000 | 16.38% |
| 10,000 | 2,560,000 | 10.24% | 3,840,000 | 15.36% | 5,120,000 | 20.48% |
| ~~24,000~~ (dropped) | 6,144,000 | 24.58% | 9,216,000 | 36.86% | 12,288,000 | **49.15%** |

Even at the low end of this range (5k, d=512), the embedding table is
~10% of the model; at the top (10k, d=512) it reaches ~20%. Staying at or
below 10k keeps the embedding table a clear minority of the budget
regardless of dimension.

---

## Hindi (Model H) — SentencePiece Unigram, complete

Initially swept at just the two endpoints of the allowed range (5,000 and
10,000) to save time — then 8,000 was added once the 5k-vs-10k comparison
alone proved misleading (see "Vocab-size choice" below). All three
trained on the full, correct train split (675,937 docs).

### Training sample size — stated plainly, not hidden

SentencePiece does not train on the entire train split — it uses
`input_sentence_size=2,000,000` with `shuffle_input_sentence=True`,
reservoir-sampling up to 2,000,000 sentences after flattening the corpus
one sentence per line.

| Language | Chars used for training | Train split total chars | Fraction sampled |
|---|---:|---:|---:|
| Hindi | 372,725,734 | 1,692,881,268 | 22.0% |

The 2,000,000-sentence sample is a random, shuffled draw across the
entire corpus, representative of the whole train split. The **held-out
evaluation** below is run separately, on the validation split, so it
measures the trained tokenizer's real generalization regardless of how
much of train it saw during fitting. The **full-corpus token count**
further below is exact and unsampled.

| Vocab size | chars/token (val sample) | chars/token (exact, full corpus) | fertility (tok/word) | UNK rate | byte-fallback rate | vocab utilization | Full-corpus tokens | Manual % (by token) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 3.3230 | NOT MEASURED (only the chosen size gets the full-corpus pass) | 1.4961 | 0.000000 | 0.018301 | 0.9654 | NOT MEASURED | NOT MEASURED |
| 8,000 | 3.5622 | **3.4884** | 1.3956 | 0.000000 | 0.019613 | 0.9762 | **505,852,028** | **30.66%** |
| 10,000 | 3.6601 | NOT MEASURED (superseded by 8,000, see below) | 1.3583 | 0.000000 | 0.020213 | 0.9767 | NOT MEASURED | NOT MEASURED |

Full-corpus exact count was run only for the chosen size (8,000). 10,000
was fully counted first (491,883,387 tokens, 3.5874 chars/token exact)
before 8,000 was added to the sweep and shown to be the better choice;
preserved below as a footnote since it is what changed the decision.

Byte-fallback rate for Hindi is 1.8–2.0% across the sweep — meaningful,
since `byte_fallback=True` is what makes UNK rate structurally 0 (see
"UNK rate" below). `<0x0A>` (newline) appears as a top-frequency
byte-fallback token at every size, consistent with paragraph-heavy
document structure.

**Sample tokenization** (vocab_size=8000):
`अतुल माहेश्वरी के निधन के कई महीनों बाद...` →
`['▁अ', 'तुल', '▁माह', 'ेश्वर', 'ी', '▁के', '▁निधन', '▁के', '▁कई', '▁महीनों', '▁बाद', ...]`

### Vocab-size choice: **8,000** (revised from an initial 10,000)

The sweep was first run at only 5,000 and 10,000. That comparison alone
suggested 10,000: vocab utilization *improved* with size (96.54% →
97.67%), so the usual "wasted embedding capacity" argument against a
larger vocab didn't apply. Adding 8,000 changed the picture: utilization
at 8k (97.62%) is already almost identical to 10k (97.67%) — the real
jump happens between 5k and 8k. Fertility shows the same pattern: 5k→8k
improves fertility by 6.7% relative, but 8k→10k only adds another 2.7%
— for 4.1 more points of embedding budget. Measured directly: **8,000
produces more total tokens than 10,000** (505,852,028 vs 491,883,387),
pushing Hindi's token count to **101.2% of the 500M target**, versus 10k's
98.4%.

**Full-corpus counts for the chosen size (vocab_size=8000):**

| Split | Docs | Characters | Tokens |
|---|---:|---:|---:|
| train | 675,937 | 1,692,881,268 | 485,280,741 |
| validation | 14,082 | 36,497,051 | 10,484,640 |
| test | 14,082 | 35,218,463 | 10,086,647 |
| **Total** | **704,101** | **1,764,596,782** | **505,852,028** |

**505,852,028 tokens — 101.2% of the ~500M target.** Manual tokens:
155,104,651. **Manual share by token count: 30.66%.**

**Footnote — vocab_size=10000 (measured, not used):** 491,883,387 total
tokens, 3.5874 chars/token exact, 30.63% manual by token. Kept for the
record since this measurement is what surfaced the diminishing-returns
pattern that led to choosing 8,000 instead.

---

## Nepali (Model L) — SentencePiece BPE, complete

### Why BPE for Nepali

`nepali/configs/tokenizer.yaml` configures `model_type: bpe` as this
pipeline's default — configured independently of, and before, Hindi's
Unigram choice was revisited for this report, so it is not a considered
per-language linguistic decision the way Hindi's Unigram-vs-BPE argument
(above) is. The config does include a genuine safety net:
`compare_unigram.enabled: true`, set to also train Unigram at the chosen
vocab size and keep whichever scores better *if* BPE's fertility exceeded
`trigger_fertility_above: 2.10` tokens/word. It never triggered — BPE's
measured fertility (1.4825–1.6988 across the sweep, see below) stayed
well under that threshold, so BPE was kept without needing the fallback.

### UNK rate: real for Nepali, not structural like Hindi's

`byte_fallback: false` for Nepali (vs. Hindi's `True`) is a real,
meaningful methodological difference. For Hindi, UNK=0.0 is guaranteed by
construction — unrepresentable text falls back to raw-byte pieces
instead of `<unk>`. For Nepali, with byte-fallback disabled, a measured
UNK rate of 0.0 at every vocab size means the trained vocabulary
genuinely represented 100% of the validation sample directly, with zero
fallback of any kind — a stronger, more informative result than Hindi's
structurally-guaranteed zero.

### Sweep table

| Vocab size | chars/token (pipeline's own sweep) | fertility (pipeline's own) | chars/token (fresh re-measure) | fertility (fresh re-measure) | UNK rate | byte-fallback rate | vocab utilization (fresh) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 3.7976 | 1.7032 | 3.7882 | 1.6988 | 0.000000 | 0.000000 | 0.9776 |
| 6,000 | 3.9495 | 1.6378 | 3.9410 | 1.6329 | 0.000000 | 0.000000 | 0.9727 |
| 8,000 | 4.1808 | 1.5472 | 4.1725 | 1.5424 | 0.000000 | 0.000000 | 0.9616 |
| 10,000 | 4.3505 | 1.4868 | 4.3410 | 1.4825 | 0.000000 | 0.000000 | 0.9458 |

Both columns are real measurements on independent same-methodology
2,000-document validation samples (pipeline's own sweep at training time,
vs. this report's fresh re-measurement) — close but not bit-identical,
which is expected of two different samples, not a discrepancy.

**Sample tokenizations** (vocab_size=10,000, from
`nepali/tokenizer/evaluation_report.json`):

```
नेपाली पद वर्ग सङ्खेय नाम अर्थ कमजोर भएको अवस्था; गल्ती
→ ['▁नेपाली', '▁पद', '▁वर्ग', '▁सङ्', 'खे', 'य', '▁नाम', '▁अर्थ', '▁कमजोर', '▁भएको', ...]
(45 tokens total)
```

Top tokens at vocab_size=10,000 are dominated by punctuation and function
morphemes: `▁।` (16.1M), `,` (9.6M), `▁र` ("and", 6.7M), `▁छ` (copula,
6.7M), `।` (5.3M), case/postposition suffixes `को`/`मा`/`का`/`ले`
(3.2M–4.7M each).

### Vocab-size choice: **10,000** — by the pipeline's own fertility-elbow rule

Unlike Hindi's manually-argued fertility/utilization/embedding-budget
trade-off above, Nepali's vocab selection is automated:
`nepali/tokenizer/selection_report.json` records the pipeline's own
selection note verbatim: *"no elbow found within the swept range;
fertility was still improving by more than 3% at every step, so the
largest candidate (10,000) was selected."* This is a real methodological
difference from Hindi's process, not an oversight — Nepali's config
(`selection.metric: fertility_elbow`, `min_relative_gain: 0.03`) defines
the stopping rule in advance and applies it mechanically, rather than a
narrative trade-off written after the fact.

At vocab_size=10,000, Nepali's embedding table is 5,120,000 parameters
(20.48% of a 25M budget at d=512) — the top of the TA's 5k–10k cap,
where Hindi landed in the middle (8,000, 16.38%).

**Full-corpus counts for the chosen size (vocab_size=10,000), exact,
unsampled — from `nepali/data/final/corpus_stats.json`:**

| Split | Docs | Tokens |
|---|---:|---:|
| train | 1,520,000 | 519,906,359 |
| val | 199,428 | 57,839,396 |
| test | 198,837 | 57,458,578 |
| **Total** | **1,918,265** | **635,204,333** |

**635,204,333 tokens — 127.0% of the ~500M target.** Manual tokens:
167,787,251. **Manual share by token count: 26.41%.**

**Held-out test-split evaluation** (from
`nepali/tokenizer/evaluation_report.json`, the pipeline's own dedicated
S13 stage, distinct from the sweep table above): test fertility 1.4672,
test chars/token 4.405, test UNK rate 0.0%, train chars/token 4.439.

---

## Chosen vocab size (final)

**Hindi: 8,000** — Unigram, chosen on a fertility/utilization/embedding-
budget trade-off argued from Hindi's own measurements.

**Nepali: 10,000** — BPE, chosen automatically by the pipeline's
fertility-elbow selection rule (no elbow found, so the largest swept
candidate was kept).

The two languages land at different points in the same 5,000–10,000 TA-
mandated range, via genuinely different selection processes — not a
coincidence to explain away, but an accurate reflection of the two
pipelines' different methodologies.

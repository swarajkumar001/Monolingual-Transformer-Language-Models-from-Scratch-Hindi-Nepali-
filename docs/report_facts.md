# Verified facts, per report section

Notes and numbers only — **not draft prose**. Everything here was read out of the
code or the result files today, with the file and line noted so you can open it
while writing and check me.

Where documentation and code disagreed, the **code wins** and the disagreement is
flagged.

---

## Discrepancies found — read these first

**1. Hindi does NOT use MinHash or LSH.** This is the big one, because the
report structure asks for "MinHash + LSH / deduplication".

* Hindi: `scripts/clean_lib.py:153` `dedup_key()` — whitespace-collapse, strip
  punctuation, lowercase, **SHA-1**. Its own docstring is explicit that this
  "catches exact dupes and formatting/whitespace variants of the same document,
  **not fuzzy/paraphrase similarity**."
* Nepali: `nepali/scripts/s08_dedup.py` — exact `doc_id` match first, then
  **MinHash + LSH**: 100 permutations, 20 bands × 5 rows, Jaccard threshold
  0.55, 5-character shingles, seed 20260817 (`nepali/scripts/corpuslib/dedup.py:45-48`).

So the two pipelines dedup differently. If you write "I used MinHash" as a
project-wide statement it is wrong, and an examiner who opens `clean_lib.py`
will find that out. Say which language got which, and that this asymmetry is a
real limitation of the Hindi pipeline.

**2. Nepali stage S07 runs before S05 and S06**, despite the numbering. It reads
S04's output; S05/S06 then read S07's *train pool*. That is deliberate — it
keeps the quality filters from curating the evaluation sets.

**3. `nepali/configs/language.yaml` used to say Hindi splits 90/5/5.** It is
96/2/2. I corrected the comment on 2026-09-16; mention it only if asked.

---

## Phase 1

### Sources

* Hindi manual: Gadya Kosh, Kavita Kosh, patrika.com archive, a self-extracted
  Hindi Wikipedia XML dump (160,591 docs / 234.1M chars — largest manual
  contributor), archive.org items (Rajya Sabha / Lok Sabha transcripts,
  religious texts, books).
* A HuggingFace-mirrored Hindi Wikipedia parquet was **quarantined** as
  redundant with the self-extracted dump — `data/raw/_quarantined_hf_hiwiki_mirror/`.
* `scripts/manual_sites.json` records a robots.txt check per site;
  bhaskar.com and amarujala.com were **ruled out** because their robots.txt
  disallows automated agents. Good evidence of doing collection properly.

### Hindi raw → cleaned (`report/phase1/dataset_statistics.md`)

| | Docs | Characters |
|---|---:|---:|
| Manual | 229,011 | 540,367,035 |
| Downloaded | 544,595 | 1,299,990,112 |
| Raw total | 773,606 | 1,840,357,147 |

Manual share of raw characters 29.36%. Cleaned: **704,103 docs**,
**1,764,598,107 chars**, retention **95.9% of raw chars**.

Drop reasons (Hindi): too short <200 chars **63,132**; exact/near-duplicate
**2,983**; langdetect mismatch **1,778**; likely Bhojpuri **1,365**; low
Devanagari ratio **228**; malformed Unicode **17**.

The Bhojpuri reject filter exists because Bhojpuri and Hindi share script and
much vocabulary — an active filter, not incidental.

### Nepali retention funnel (`nepali/data/final/corpus_stats.json`)

| Stage | Docs in | Docs out | Retention |
|---|---:|---:|---:|
| S03 normalise | 2,655,504 | 2,655,504 | 1.0000 |
| S04 language ID | 2,655,504 | 2,649,643 | 0.9978 |
| S07 train pool | 2,649,643 | 2,251,378 | 0.8497 |
| S05 heuristics | 2,251,378 | 2,245,354 | 0.9973 |
| S06 perplexity | 2,245,354 | 2,178,418 | 0.9702 |
| S08 dedup | 2,178,418 | 2,126,334 | 0.9761 |

Nepali dedup removed **2,350 exact** and **49,734 near** duplicates.
**21,273** cross-lane duplicate pairs resolved manual-first, so 21,273 manual
documents were preserved by priority ordering. Overall **635,204,333 tokens**;
both manual and downloaded targets met.

### Preprocessing (`scripts/clean_lib.py:75`)

`normalize_text()` is exactly four operations: NFC normalise → strip
zero-width/control → collapse inline whitespace runs → collapse blank-line runs
→ strip.

The stripping regex removes U+200E, U+200F, U+FEFF and C0/C1 control bytes, and
**deliberately preserves ZWNJ and ZWJ**, which are meaningful in Devanagari.
That is a decision you should be able to defend.

Other filters: `devanagari_ratio()` (min 0.5), `malformed_ratio()` via U+FFFD
(max 0.01), `is_too_short()` (min 200 chars), `langdetect_verdict()` — which
returns True/False/**None**, three-valued, so "undetermined" is not treated as
"wrong". Hindi thresholds are in `hindi/data/pipeline/clean_hindi.py`
(`LANGDETECT_MIN_PROB = 0.85`).

### Tokenizers

Full measured sweep is in `docs/report_numbers.md`. Headline: Hindi Unigram
8,000 — chars/token 3.5622, fertility 1.3956, **unk 0.000000**, byte fallback
0.019613. Nepali BPE 10,000 — chars/token 4.3410, fertility 1.4825, **unk
0.003387**, byte fallback **0.000000**.

Nepali's byte-fallback column is zero at *every* vocabulary size. That is the
defect, not a good score, and it is the root of the unk rate.

### Splits

Hindi `hindi/data/pipeline/split_hindi.py`: 96/2/2, seed 42, shuffle then
slice, hardlinks (`link_or_copy`), per-split manifest JSON. Nepali: 85/7.5/7.5,
seed 20260817, deterministic doc-id hash, held out at S07 *before* quality
filtering, then S09 MinHash decontamination that removes **only from train**.
`split_stats.json` records `zero_doc_id_overlap: true` and per-split manual
ratios 0.2742 / 0.2371 / 0.2311 against an overall 0.267.

---

## Phase 2

Architecture, training config, all metrics: `docs/report_numbers.md`.

Implementation facts worth stating, with locations:

* `hindi/model/transformer.py:69` `MultiHeadCausalSelfAttention` — W_q/W_k/W_v
  kept as **three separate** `nn.Linear`, not one fused matmul, deliberately.
* `:123` causal mask via `torch.triu(..., diagonal=1)`, registered
  `persistent=False`; `:173` applied with `masked_fill(-inf)` before softmax.
* Dropout in four places: `:178` attention probs, `:189` attention output,
  `:225` feed-forward, `:467` embeddings.
* `:347` learned absolute positions; `:279` sinusoidal and `:355` "none" also
  implemented (the "none" path is what the bonus uses).
* `:365` weight tying; `:490` `generate()` with temperature and top-k.
* `:546` `configure_optimizers` — weight decay applied only where
  `param.dim() >= 2`.
* `hindi/train/train.py:351` training loop; `:383-390` gradient accumulation;
  `:396-398` `unscale_` **then** clip; `:451` wall-clock exit; `:34`
  `CosineWarmupSchedule`; `:246` checkpoint payload; `:291` resume.
* Causal-mask proof: `hindi/tests/test_model.py:51`, max logit leakage
  **0.000e+00**, and the test asserts the converse too. 9/9 pass in both suites.

Honest facts: runs stopped at 14,520 and 14,440 of a planned 15,000 when the
11-hour budget fired. Evaluated checkpoints are step 14,500 and 14,000. Final
evaluation ran on CPU for both, deliberately, so temperature sampling drew from
the same RNG.

---

## Phase 3

Dataset stats, accuracy tables, per-task breakdowns and off-prompt rates are all
in `docs/report_numbers.md`.

Facts worth stating:

* Four task families: `compare2`, `transitive3`, `numeric2`, `numeric3`,
  5,000 train examples each.
* Leakage control: 40/10/10 disjoint entity names, 14/4/4 disjoint templates,
  **asserted in `main()`** of `generate.py`, and recorded in
  `dataset_stats.json` as `entity_names_disjoint_across_splits: true` and
  `templates_disjoint_across_splits: true`.
* Answer-only loss via `ignore_index=-1` (`hindi/reasoning/dataset.py`).
* Length bucketing instead of padding — required because learned absolute
  positions mean padding shifts every real token's position index.
* The finetuner **asserts** the tokenizer vocab size matches the checkpoint.
* Forced choice = length-normalised mean log-prob per answer token; chance rate
  **41.67%**, computed not assumed.

The two defects found during construction (both in the report, both worth
mentioning as evidence of checking your own work):

1. Nepali numerals — the Nepali BPE vocabulary contains **no Devanagari digit**
   and has no byte fallback, producing a 4.8% unk rate on the generated data.
   Fixed by switching Nepali numerals to Arabic digits. Hindi's Unigram covers
   both at 0 unk.
2. An adjective-agreement leak, closed by drawing same-gender entities.

Overfitting: validation best at step **60** for both, then rises to 3.5188 and
3.5911 by step 1,250 while train loss falls to 0.44 and 0.60. Best-validation
checkpointing means the evaluated checkpoint is step 60, so this did not
contaminate the reported numbers.

Run A vs run B: a 100-step evaluation grid selected its checkpoint at step 100,
exactly where warmup ends. Moving to a 20-step grid, nothing else changed,
raised Model L from 14.90% to 29.45% exact match and Model H from 32.70% to
35.15%. Kept in `report/phase3/run_a_40names/`.

---

## Bonus

All numbers in `docs/report_numbers.md` and `report/bonus/README.md` (on the
`bonus-no-pos` branch).

Facts: one variable changed (`pos_encoding: learned` → `none`); 262,144 fewer
parameters, non-embedding count **identical** at 18,915,328; `max_steps` kept at
15,000 so the cosine schedule matched; seed held identical so batch order
matched. The ablated run got a much faster GPU (41.6k vs 12.0k tok/s) and
finished all 15,000 steps, so it is the *better*-trained of the two — the
degradation is not a training-length artefact.

Be careful: the mask-leakage explanation is **consistent with** the evidence,
not proven by it. Say it that way.

---

## Things you should NOT claim

* That Hindi used MinHash. It did not.
* That the models perform reasoning. Forced choice is at or below chance.
* That Model L is a better language model because its BPB is lower. The gap is
  mostly tokenizer granularity.
* That the ablation proves causal masking leaks position. It is consistent with
  that; proving it needs a probe you did not run.
* Any wall-clock or throughput comparison between the baseline and the ablation.
  Different GPUs.

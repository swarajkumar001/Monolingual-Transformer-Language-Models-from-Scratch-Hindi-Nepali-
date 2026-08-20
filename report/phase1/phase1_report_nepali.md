# Phase 1 Report — Nepali (Model L)

Author: Swaraj Kumar (2025202016)

This is a standalone Phase 1 report for **Nepali as Model L**, covering
dataset collection, cleaning, splits, and tokenizer construction. It follows
the same structure as `report/phase1/phase1_report.md` (Hindi/Bhojpuri) but
is a separate document because Nepali runs on a different pipeline
(`nepali/scripts/`, stage-based S00–S14) with a different data layout than
the Hindi/Bhojpuri cleaner, so its own numbers are not simply insertable into
that report without changing what each figure actually measures.

## 1. Overview

| | |
|---|---|
| Model | L (lower-resource) |
| Language | Nepali (allowed-list language, course spec §1.1) |
| Final corpus size | 635,204,333 tokens (127.0% of the ~500M planning target) |
| Manual share | 26.41% of final tokens (floor is 20%) |
| Tokenizer | BPE, vocab_size = 10,000 |
| Train / val / test docs | 1,520,000 / 199,428 / 198,837 |

Nepali's pipeline enforces two **absolute** token targets rather than a
single overall total: manual lane ≥ 100,000,000 tokens, downloaded lane ≥
400,000,000 tokens (`nepali/configs/language.yaml`). Both are met:

| Lane | Tokens | Target | Met |
|---|---:|---:|---|
| Manual | 167,787,251 | ≥100,000,000 | Yes |
| Downloaded | 467,417,082 | ≥400,000,000 | Yes |
| **Total** | **635,204,333** | ~500,000,000 (course target) | 127.0% |

Source: `nepali/data/final/corpus_stats.json`.

## 2. Data collection

### Manual vs. downloaded, as defined for this pipeline

Same course ruling as Hindi/Bhojpuri: "manual" means content requiring
active per-source collection work (API/XML-dump parsing, HTML scraping with
site-specific extraction, OCR) as opposed to downloading an already-corpus-
ified public dataset. Unlike the Hindi/Bhojpuri report's char-weighted
manual share, Nepali's split is **token-weighted** — the pipeline tracks
words/tokens per lane past S03, not characters. This is a methodology
difference worth being explicit about if the two reports are ever compared
directly.

### Manual-lane sources

| Source | Docs | Words | Tokens |
|---|---:|---:|---:|
| news_onlinekhabar (web scrape) | 308,139 | 107,614,170 | 153,717,439 |
| wikipedia_api (XML dump, own wikitext parser) | 16,445 | 4,622,944 | 8,052,774 |
| news_bbc_nepali (web scrape) | 12,377 | 3,555,706 | 5,277,060 |
| wikibooks_api | 74 | 230,885 | 412,110 |
| wiktionary_api | 606 | 96,765 | 200,300 |
| news_ekantipur | 119 | 80,432 | 115,316 |
| news_ratopati | 20 | 6,055 | 10,608 |
| ocr_archive_books | 1 | 447 | 866 |
| news_nagarik | 2 | 503 | 778 |
| **Manual total** | **337,783** | **116,207,907** | **167,787,251** |

### Downloaded-lane sources

| Source | Docs | Words | Tokens |
|---|---:|---:|---:|
| sangraha_verified (ai4bharat/sangraha) | 579,743 | 173,902,717 | 258,462,424 |
| sangraha_unverified (ai4bharat/sangraha) | 341,901 | 99,767,699 | 141,886,279 |
| indiccorp_v2 (ai4bharat/IndicCorpV2) | 658,838 | 45,407,223 | 67,068,379 |
| **Downloaded total** | **1,580,482** | **319,077,639** | **467,417,082** |

Source: `nepali/data/final/corpus_stats.json`, `by_source` breakdowns summed
across train/val/test.

### Why the manual lane is dominated by one source

`news_onlinekhabar` alone supplies 91.6% of manual tokens. This is a real,
measured constraint, not a collection shortcut: OnlineKhabar has the largest
scrapable Nepali news archive among the manual sources configured, and
`nepali/scripts/auto_grow_and_tokenize.py`'s growth loop bumped its
`target_words` in `sources.yaml` (capped at 260,000,000 words,
`MANUAL_SOURCE_CAPS`) specifically because it was the only manual source
with enough real headroom to close the gap to the 100M-token manual floor.
The other manual sources (Wikipedia, Wiktionary, Wikibooks, smaller news
sites, one OCR document) are genuinely smaller pools for Nepali than their
Hindi equivalents.

## 3. Cleaning pipeline

Nepali's pipeline is stage-based (`nepali/scripts/s00_doctor.py` through
`s14_corpus_stats.py`), sharing the same `corpuslib` code as Hindi's own
Nepali-parallel pipeline copy (byte-identical, hashed at CI time — see
`nepali/scripts/corpuslib/config.py` module docstring) so the two pipelines
cannot silently drift apart while still operating on fully separate data.

### Document retention through the pipeline

| Stage | Docs in | Docs out | Retention |
|---|---:|---:|---:|
| S03 normalise | 2,655,504 | 2,655,504 | 100.00% |
| S04 language ID | 2,655,504 | 2,649,643 | 99.78% |
| S07 train-pool split-off | 2,649,643 | 2,251,378 | 84.97% |
| S05 heuristic filters | 2,251,378 | 2,245,354 | 99.73% |
| S06 perplexity filter | 2,245,354 | 2,178,418 | 97.02% |
| S08 dedup (exact + near) | 2,178,418 | 2,126,334 | 97.61% |
| S09 decontaminate (vs. val/test) | 2,126,334 | 2,109,223 | 99.20% |
| S10 balance lanes (train only) | 2,109,223 | 1,520,000 | 72.06% |

See `report/phase1/figures/nepali/01_retention_funnel.png`.

### S04 language-ID rejects (largest reasons)

| Reason | Docs |
|---|---:|
| wrong_language | 1,599 |
| low_confidence | 1,189 |
| narrow_margin_vs_hi | 600 |
| too_short | 251 |
| other script-ratio buckets (≤5 docs each) | 121 |

`narrow_margin_vs_hi` (600 docs) is notable given Nepali and Hindi share the
Devanagari script and overlapping vocabulary — this is the language-ID
model catching documents too ambiguous to confidently call Nepali rather
than Hindi, which is exactly the failure mode the cross-language check in
§6 independently verifies did not leak through.

### S05 heuristic-filter drops

| Reason | Docs |
|---|---:|
| repeated_lines | 1,858 |
| too_few_sentences | 1,226 |
| too_short | 382 |
| link_farm | 481 |
| low_lexical_diversity | 5 |
| too_few_words | 8 |
| mean_word_too_long | 1 |

See `report/phase1/figures/nepali/03_drop_reasons.png`.

### S08 deduplication

Removed 52,084 of 2,178,418 documents: 2,350 exact duplicates, 49,734 near
duplicates (MinHash/LSH, `num_perm=100, bands=20, rows=5, shingle_size=5,
threshold=0.55`). 21,273 cross-lane duplicate pairs were found (the same
content collected via both the manual and downloaded lanes); all 21,273
were resolved in favor of the manual copy, per the pipeline's
priority-to-manual policy.

### S09 decontamination

Removed 17,111 train documents that matched held-out (val+test) content
above threshold 0.55, preventing train/eval leakage at the content level (not
just document-id level). A rescan afterward found 0 remaining matches.

### S10 lane balancing

The train pool's raw manual share going into S10 was 23.27%. Rather than
loosening any filter to inflate it, S10 trimmed the **downloaded** lane
(589,223 documents dropped, manual lane untouched) to bring the ratio to
27.42% — safely above the 20% floor without manufacturing volume.
See `report/phase1/figures/nepali/10_lane_balancing.png`.

## 4. Splits

| Split | Docs | Words | Tokens | Manual share |
|---|---:|---:|---:|---:|
| Train | 1,520,000 | 356,691,169 | 519,906,359 | 27.42% |
| Val | 199,428 | 39,432,622 | 57,839,396 | 23.71% |
| Test | 198,837 | 39,161,755 | 57,458,578 | 23.11% |
| **Overall** | **1,918,265** | **435,285,546** | **635,204,333** | **26.70%** (word-weighted) / **26.41%** (token-weighted) |

Val and test were held out at S07, *before* S10's lane balancing — which is
why their manual share (23–24%) differs slightly from train's post-balancing
27.42%; both are comfortably above the 20% floor regardless.

Zero document-id overlap between all three splits, confirmed by
`nepali/data/final/split_stats.json` (`zero_doc_id_overlap: true`,
`flagged_splits: []`) and independently re-checked by
`scripts/validate_phase1_nepali.py`. See
`report/phase1/figures/nepali/04_split_sizes.png`.

## 5. Tokenizer

### Algorithm and sweep

BPE (SentencePiece), swept at vocab_sizes [5000, 6000, 8000, 10000] per
`nepali/configs/tokenizer.yaml`, matching the course-TA ruling capping
recommended vocab size to the 5,000–10,000 range given the ~25M-parameter
model budget (the same ruling applied to Hindi's own vocab_size=8000
choice).

| Vocab size | Fertility (tokens/word) | Chars/token | UNK rate |
|---:|---:|---:|---:|
| 5,000 | 1.7032 | 3.7976 | 0.0% |
| 6,000 | 1.6378 | 3.9495 | 0.0% |
| 8,000 | 1.5472 | 4.1808 | 0.0% |
| **10,000 (chosen)** | **1.4868** | **4.3505** | **0.0%** |

Selection note (from `nepali/tokenizer/selection_report.json`): "no elbow
found within the swept range; fertility was still improving by more than 3%
at every step, so the largest candidate (10,000) was selected" — i.e. the
sweep's own stopping rule, not a manually picked number. See
`report/phase1/figures/nepali/05_vocab_chars_per_token.png`,
`06_vocab_fertility.png`, `07_vocab_unk_rate.png`.

### Held-out test-split evaluation (chosen model, vocab_size=10,000)

| Metric | Value |
|---|---:|
| Test fertility | 1.4672 tokens/word |
| Test chars/token | 4.405 |
| Test UNK rate | 0.0% |
| Train chars/token | 4.439 |

Source: `nepali/tokenizer/evaluation_report.json`.

### Sample tokenizations (real model output, vocab_size=10,000)

```
नेपाली पद वर्ग सङ्खेय नाम अर्थ कमजोर भएको अवस्था; गल्ती
→ ▁नेपाली ▁पद ▁वर्ग ▁सङ् खे य ▁नाम ▁अर्थ ▁कमजोर ▁भएको ▁अवस्था ; ▁गल्ती ...
(45 tokens total)

नेपाली पद वर्ग नाम अर्थ मान्छे; मानिस
→ ▁नेपाली ▁पद ▁वर्ग ▁नाम ▁अर्थ ▁मान्छे ; ▁मानिस ...
(45 tokens total)
```

Full examples with complete piece sequences are in
`nepali/tokenizer/evaluation_report.json`.

### Top tokens by frequency (top 50, training corpus)

Most frequent pieces are punctuation and function morphemes, as expected for
a subword tokenizer on inflected Devanagari text: `▁।` (16.1M), `,` (9.6M),
`▁र` ("and", 6.7M), `▁छ` (copula, 6.7M), `।` (5.3M), `को`/`मा`/`का`/`ले`
(case/postposition suffixes, 3.2–4.7M each). See
`report/phase1/figures/nepali/08_top_token_frequency.png` — labelled
explicitly as top-50 only, not the full piece-frequency distribution, since
the latter would require re-encoding the entire corpus and
`evaluation_report.json` only retained the top 50.

## 6. Cross-language check (Nepali vs. Hindi)

Run via `scripts/check_nepali_hindi_overlap.py`: every one of Nepali's
1,918,265 final documents (train 1,520,000 + val 199,428 + test 198,837)
had its `dedup_key` (sha1 of whitespace-collapsed, punctuation-stripped,
lowercased text — identical algorithm to `scripts/clean_lib.py:dedup_key`,
computed fresh since Nepali's own pipeline doesn't store one) checked
against the full set of 704,101 dedup_keys already computed for Hindi's
cleaned corpus (`hindi/data/cleaned/metadata.jsonl`).

**Result: `[PASS]` — 0 shared dedup_keys.** No document is shared between
the Nepali and Hindi corpora. Full run took 467.7s. This addresses the
spec's "do not share documents across corpora" requirement for the two
languages that are actually part of the graded submission (Nepali vs.
Bhojpuri was not checked, since Bhojpuri is superseded and not part of the
final two-language submission).

## 7. Reproducibility

```
cd nepali/scripts
python s00_doctor.py                    # environment check
python auto_grow_and_tokenize.py        # full S02-S14 pipeline, grows lanes to target
python ../../scripts/validate_phase1_nepali.py
python ../../report/phase1/make_plots_nepali.py
python ../../scripts/check_nepali_hindi_overlap.py
```

All figures and this report's numbers are direct reads of
`nepali/data/final/corpus_stats.json`, `split_stats.json`,
`nepali/tokenizer/selection_report.json`, `evaluation_report.json` — nothing
here is hand-transcribed or estimated.

**Provenance note:** the corpus/tokenizer artifacts referenced in this
report were produced by running `nepali/scripts/` on a separate machine
(the primary machine's disk/time constraints made the ~30+ hour S08
dedup pass impractical there); only the pipeline's output files were
transferred into this repo, not live per-stage console logs. `log.md`'s
S08(production)–S14 entries are reconstructed from those output files
rather than logged in real time — see the note inline in `log.md` for
exactly which entries this applies to and their sourcing.

## 8. Limitations

- **Manual share is token-weighted, not char-weighted** (see §2), unlike
  the Hindi/Bhojpuri report. Character-level lane tracking was never added
  to this pipeline; token-weighted is the honest number available.
- **Cross-language check covers Nepali vs. Hindi only** (§6), not
  Nepali vs. Bhojpuri, since Bhojpuri is not part of the final two-language
  submission.
- **Top-token frequency (Fig. 8) is top-50 only**, not a full Zipf
  distribution, for the reason stated in §5.
- **manual-lane concentration**: 91.6% of manual tokens come from one
  source (OnlineKhabar). This is disclosed in §2 rather than smoothed over.

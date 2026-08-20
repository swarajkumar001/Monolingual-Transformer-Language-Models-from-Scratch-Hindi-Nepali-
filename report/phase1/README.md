# Phase 1 — Deliverables Index

Model H = **Hindi**. Model L = **Nepali**. This page is the entry point
for Phase 1: it maps every required deliverable to its exact file(s), then
summarizes raw data, cleaned data, splits, tokenizers, and resources for
both languages. For the full narrative report with all 10 required
figures, see **[`phase1_report.md`](phase1_report.md)**. Every number here
is also in that report, traceable to the same files on disk. Hindi and
Nepali run on two structurally different pipelines, so several tables
below present them side by side rather than forcing one shared schema —
see `phase1_report.md`'s opening note for why.

`scripts/validate_phase1.py` reports **ALL CHECKS PASSED** for Hindi.
`scripts/validate_phase1_nepali.py` reports **ALL CHECKS PASSED** for
Nepali except the cross-language check (run separately as its own pass —
see `scripts/check_nepali_hindi_overlap.py`, result in
`phase1_report.md` §3: 0 shared documents against Hindi).

## Deliverables checklist

| # | Deliverable | Where |
|---|---|---|
| 1 | Dataset collection scripts (both languages) | `hindi/data/pipeline/{download_hindi.py, ingest_manual.py, ingest_wikipedia_dump.py}`; `nepali/scripts/{s01_collect.py, collect.py, auto_grow_and_tokenize.py, corpuslib/}` |
| 2 | Dataset preprocessing pipelines | `hindi/data/pipeline/clean_hindi.py`, `scripts/clean_lib.py` (shared logic); `nepali/scripts/{s03_normalize.py..s11_finalize_splits.py}` |
| 3 | Per-language dataset statistics reports | [`phase1_report.md`](phase1_report.md) (both languages), [`dataset_statistics.md`](dataset_statistics.md), [`tokenizer_analysis.md`](tokenizer_analysis.md) |
| 4 | Per-language train/validation/test splits | `hindi/data/splits/{train,validation,test}/` + `*_manifest.json`, built by `split_hindi.py`; `nepali/data/final/{train,val,test}.jsonl` + `split_stats.json`, built by `s11_finalize_splits.py` |
| 5 | Tokenizer training code (both languages) | `hindi/tokenizer/train_tokenizer.py`; `nepali/scripts/{s12_train_tokenizer.py, s13_tokenizer_eval.py}` |
| 6 | Vocabulary files (one per language) | `hindi/tokenizer/hindi_unigram_8000.vocab`; `nepali/tokenizer/nepali_bpe.vocab` (plus other swept sizes, kept for the record) |
| 7 | Tokenizer model files (one per language) | `hindi/tokenizer/hindi_unigram_8000.model`; `nepali/tokenizer/nepali_bpe.model` |

Raw and cleaned corpora themselves are too large for git — see the root
[`README.md`](../../README.md)'s "Large artifacts" section (OneDrive, a
deliberate choice — see that section's note).

## Raw / final data

Hindi tracks characters at every stage; Nepali's pipeline tracks
docs/words/tokens instead (no per-lane character tally past
normalization) — units differ by design, not oversight.

| | Hindi | Nepali |
|---|---:|---:|
| Total raw docs | 773,606 | ~2,655,504 (S03 input) |
| Total raw chars | 1,840,357,147 | not tracked |
| Manual docs | 229,011 | 337,783 (final) |
| Downloaded docs | 544,595 | 1,580,482 (final) |

**Resources used — Hindi manual** (self-run collection, counts toward the
manual floor): `hi_wikipedia_dump` (self-extracted Wikipedia XML dump via
WikiExtractor — the largest single manual source), `gadya_kosh` (prose
wiki), `patrika_hindi` (patrika.com newspaper archive), `kavita_kosh`
(poetry wiki), and 67 misc. `archive.org` items (parliamentary debate
transcripts, religious texts, books).

**Resources used — Hindi downloaded**: `HuggingFaceFW/fineweb-2:hin_Deva`,
`ai4bharat/sangraha:unverified/hin`, `ai4bharat/sangraha:verified/hin:{web,pdf,speech}`.
Excluded: the HF-mirrored `wikimedia/wikipedia:20231101.hi` (quarantined —
redundant with the self-extracted dump, and would not count as manual
regardless).

**Resources used — Nepali manual**: `news_onlinekhabar` (web scrape, 91.6%
of manual tokens), `wikipedia_api` (self-parsed XML dump), `news_bbc_nepali`,
`wikibooks_api`, `wiktionary_api`, `news_ekantipur`, `news_ratopati`,
`news_nagarik`, `ocr_archive_books`.

**Resources used — Nepali downloaded**: `ai4bharat/sangraha` (`verified` +
`unverified`), `ai4bharat/IndicCorpV2`.

Full source-by-source tables (docs/words/tokens per source, both
languages) are in [`dataset_statistics.md`](dataset_statistics.md) and
[`phase1_report.md`](phase1_report.md) §2.

## Cleaning → final data

| | Hindi | Nepali |
|---|---:|---:|
| Final docs | 704,101 | 1,918,265 |
| Doc retention (raw→final) | 91.0% | 72.2% (2,655,504→1,918,265) |
| Dedup method | exact-hash | MinHash/LSH (near-dup) |
| Extra stages | — | S09 decontaminate, S10 balance lanes |

Hindi's pipeline order (verified from `clean_stats.json` checkpoints): raw
→ malformed-Unicode removal → too-short removal → language-ID filter →
exact/near-duplicate removal → cleaned. Nepali's is stage-based (S03–S10,
see `phase1_report.md` §3 for the full per-stage retention table) with two
steps Hindi's pipeline has no equivalent of: decontamination (removes
train documents matching held-out content) and lane balancing (trims the
downloaded lane rather than loosening a filter to hit the manual-share
floor). Full drop-reason breakdowns and the historical Hindi↔Bhojpuri
cross-language correction are in [`phase1_report.md`](phase1_report.md) §3.

## Splits

| | Hindi | Nepali |
|---|---:|---:|
| Train | 675,937 (seed-42 hash split) | 1,520,000 (S07 holdout, pre-balancing) |
| Validation | 14,082 | 199,428 |
| Test | 14,082 | 198,837 |
| Leakage check | `validate_phase1.py`: no overlap | `validate_phase1_nepali.py` + `split_stats.json`: no overlap |

## Tokenizers

**Hindi:** SentencePiece **Unigram**, swept 5,000–10,000, **8,000
selected** (fertility/utilization/embedding-budget trade-off, argued from
Hindi's own measurements). **Nepali:** SentencePiece **BPE**, swept
5,000–10,000, **10,000 selected** (by the pipeline's own automated
fertility-elbow rule — no elbow found, so the largest candidate was kept).
Full sweep tables, embedding-budget justification, and sample
tokenizations are in [`phase1_report.md`](phase1_report.md) §5 and
[`tokenizer_analysis.md`](tokenizer_analysis.md).

## Total tokens vs. the 500M target

| | Hindi | Nepali |
|---|---:|---:|
| Total tokens (train+val+test, exact, unsampled) | **505,852,028** | **635,204,333** |
| % of 500M target | **101.2%** | **127.0%** |
| Manual share, char-weighted | 28.65% | not tracked (see below) |
| Manual share, token-weighted | 30.66% | 26.41% |
| Clears 20% manual floor? | Yes | Yes |
| Absolute lane targets | n/a (Hindi's pipeline doesn't use this gate) | manual 167.8M ≥ 100M; downloaded 467.4M ≥ 400M — both met |

Nepali's larger public-data availability (AI4Bharat Sangraha and
IndicCorpV2 both carry sizeable Nepali partitions) let its pipeline's
growth loop (`auto_grow_and_tokenize.py`) close the gap to both absolute
lane targets and exceed the 500M planning target by a wider margin than
Hindi. See [`phase1_report.md`](phase1_report.md) §6 for the full
discussion, including why Bhojpuri (this project's original Model L, a
documented 16.8%-of-target shortfall) was superseded by Nepali.

## OneDrive (raw + cleaned corpora, tokenizer files)

**[LMA DATA SET — shared OneDrive folder](https://iiithydstudents-my.sharepoint.com/personal/swaraj_kumar_students_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fswaraj%5Fkumar%5Fstudents%5Fiiit%5Fac%5Fin%2FDocuments%2FLMA%20DATA%20SET&ga=1)**
(an IIIT-institutional link, used deliberately rather than Google Drive —
see the root README's "Large artifacts" note)

Check the root [`README.md`](../../README.md)'s artifact table for current
upload status per language/artifact.

## About Bhojpuri (superseded)

Bhojpuri was this project's original Model L: fully built, cleaned,
split, tokenized, and passing every `validate_phase1.py` check (507,994
cleaned docs, 84,217,032 total tokens — 16.8% of the 500M target, a
documented, spec-permitted shortfall). Superseded by Nepali once Nepali's
own pipeline cleared both its absolute token targets. `bhojpuri/` and its
numbers (folded into `phase1_report.md` alongside Hindi and Nepali, not a
separate document) are left exactly as they were — nothing deleted,
nothing re-run — as a complete, working fallback and a record of that
earlier work.

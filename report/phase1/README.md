# Phase 1 — Deliverables Index

Model H = **Hindi**. Model L = **Bhojpuri**. This page is the entry point
for Phase 1: it maps every required deliverable to its exact file(s), then
summarizes raw data, cleaned data, splits, tokenizers, and resources for
both languages. For the full narrative report with all 10 required figures,
see **[`phase1_report.md`](phase1_report.md)**. Every number here is also in
that report, traceable to the same files on disk.

`scripts/validate_phase1.py` reports **ALL CHECKS PASSED** for both
languages as of this writing.

## Deliverables checklist

| # | Deliverable | Where |
|---|---|---|
| 1 | Dataset collection scripts (both languages) | `hindi/data/pipeline/{download_hindi.py, ingest_manual.py, ingest_wikipedia_dump.py}`, `bhojpuri/data/pipeline/{download_bhojpuri.py, download_bhojpuri_extra.py, ingest_bhltr.py, ingest_harsh_corpus.py, ingest_manual.py, ingest_wikipedia_dump.py}`, `scripts/crawl_sites.py`, `scripts/fetch_archive_org.py` |
| 2 | Dataset preprocessing pipelines | `hindi/data/pipeline/clean_hindi.py`, `bhojpuri/data/pipeline/clean_bhojpuri.py`, `scripts/clean_lib.py` (shared logic), `scripts/remove_cross_language_duplicates.py` |
| 3 | Per-language dataset statistics reports | [`phase1_report.md`](phase1_report.md) (main report + figures), [`dataset_statistics.md`](dataset_statistics.md), [`tokenizer_analysis.md`](tokenizer_analysis.md) |
| 4 | Per-language train/validation/test splits | `hindi/data/splits/{train,validation,test}/` + `*_manifest.json`, `bhojpuri/data/splits/{train,validation,test}/` + `*_manifest.json`, built by `split_hindi.py` / `split_bhojpuri.py` |
| 5 | Tokenizer training code (both languages) | `hindi/tokenizer/train_tokenizer.py`, `bhojpuri/tokenizer/train_tokenizer.py` |
| 6 | Vocabulary files (one per language) | `hindi/tokenizer/hindi_unigram_8000.vocab`, `bhojpuri/tokenizer/bhojpuri_unigram_8000.vocab` (plus the other swept sizes, kept for the record) |
| 7 | Tokenizer model files (one per language) | `hindi/tokenizer/hindi_unigram_8000.model`, `bhojpuri/tokenizer/bhojpuri_unigram_8000.model` |

Raw and cleaned corpora themselves are too large for git — see
**Google Drive**, below.

## Raw data

| | Hindi | Bhojpuri |
|---|---:|---:|
| Total raw docs | 773,606 | 608,263 |
| Total raw chars | 1,840,357,147 | 318,606,726 |
| Manual docs / chars | 229,011 / 540,367,035 | 47,958 / 90,781,624 |
| Downloaded docs / chars | 544,595 / 1,299,990,112 | 560,305 / 227,825,102 |

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

**Resources used — Bhojpuri manual**: four dedicated crawl sources
(`khabar_bhojpuri`, `anjoria`, `jogira`, `sirijan` — recrawled uncapped and
confirmed exhausted), `bho_wikipedia_dump` (self-extracted, same method as
Hindi's), 88 misc. `archive.org` items, and a prior personally-collected
scrape (`harsh_scraped_manual`).

**Resources used — Bhojpuri downloaded**: 17 HF datasets/partitions, the
largest being the `Satyam810/BhojpuriCorpus` family (multiple partitions),
`harsh_indiccorp` (personal prior collection), JNU's BHLTR corpus, and
`HuggingFaceFW/fineweb-2:bho_Deva`. Six other HF datasets were evaluated and
rejected: CulturaX/sil-ai-bloom-lm (gated), Sangraha/Glot500 (zero Bhojpuri
rows), MADLAD-400 (no `bho` config), CC-100 (unsupported legacy format).

Full source-by-source tables (docs + chars per source) are in
[`dataset_statistics.md`](dataset_statistics.md) §"raw corpus", and per-source
**token** counts for Hindi are in `hindi/hindi_token_counts_by_source.json`.

## Cleaning → cleaned data

| | Hindi | Bhojpuri |
|---|---:|---:|
| Cleaned docs | 704,101 | 507,994 |
| Cleaned chars | 1,764,596,782 | 282,815,177 |
| Doc retention | 91.0% | 83.5% |
| Char retention | 95.9% | 88.8% |

Pipeline order (verified from `clean_stats.json` checkpoints, not assumed):
raw → malformed-Unicode removal → too-short removal → language-ID filter →
exact/near-duplicate removal (manual copy wins on cross-lane ties) →
cleaned. Full drop-reason breakdown, the langdetect-has-no-`bho`-label
limitation, and the 2-document cross-language correction are in
[`phase1_report.md`](phase1_report.md) §3.

## Splits (document-level, seed 42, no leakage)

| | Hindi | Bhojpuri |
|---|---:|---:|
| Train | 675,937 | 487,674 |
| Validation | 14,082 | 10,160 |
| Test | 14,082 | 10,160 |

## Tokenizers

Both: SentencePiece **Unigram**, vocab swept over 5,000–10,000 (course-TA
cap), **8,000 selected for both languages**. Full sweep tables, embedding-
budget justification, and sample tokenizations are in
[`phase1_report.md`](phase1_report.md) §5.

## Total tokens vs. the 500M target

| | Hindi | Bhojpuri |
|---|---:|---:|
| Total tokens (train+val+test, exact, unsampled) | **505,852,028** | **84,217,032** |
| % of 500M target | **101.2%** | **16.8%** |
| Manual share, char-weighted | 28.65% | 28.59% |
| Manual share, token-weighted | 30.66% | 29.89% |
| Clears 20% manual floor? | Yes | Yes |

Bhojpuri's shortfall against 500M is expected and spec-permitted for the
low-resource model, not a gap in effort — all known manual sources are
confirmed exhausted, and this corpus is already orders of magnitude larger
than any previously published Bhojpuri monolingual resource. See
[`phase1_report.md`](phase1_report.md) §6 for the full justification.

## OneDrive (raw + cleaned corpora, tokenizer files)

**[LMA DATA SET — shared OneDrive folder](https://iiithydstudents-my.sharepoint.com/personal/swaraj_kumar_students_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fswaraj%5Fkumar%5Fstudents%5Fiiit%5Fac%5Fin%2FDocuments%2FLMA%20DATA%20SET&ga=1)**
(requires an IIIT account with access)

Check the root [`README.md`](../../README.md)'s artifact table for current
upload status per language/artifact.

## About the Nepali attempt

A second, independent low-resource candidate for Model L — **Nepali**, also
on the assignment's allowed list — was explored in parallel, using a
separate, more elaborate pipeline under `nepali/` (config-driven, stages
S02–S14, see `nepali/scripts/run_pipeline.py`). The goal was to check
whether Nepali's larger public-data availability could produce a stronger
low-resource corpus than Bhojpuri while still clearing the 20% manual floor.

It did not make it into the graded submission: an early run measured only
6.3% manual share (later found to be a stale measurement, taken before a
~72M-word manual news source had finished collecting), and after fixing
that and rerunning against the complete raw corpus, the pipeline stalled at
the deduplication stage (a MinHash/LSH pass over ~1.36M documents) at a
pace that would have taken 10+ hours to finish — too long to safely depend
on before the Phase 1 deadline.

**Bhojpuri is the final, graded Model L.** It was fully built, cleaned,
split, tokenized, and passing every `validate_phase1.py` check well before
the Nepali attempt was even started, and nothing in this report or
`report/phase1/` depends on Nepali finishing. `nepali/` is kept in the repo
only for transparency into that exploration.

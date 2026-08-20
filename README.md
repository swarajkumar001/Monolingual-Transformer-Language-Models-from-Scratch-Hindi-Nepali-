# Language Models and Agents — Individual Project

Two independent, from-scratch, decoder-only Transformer language models in
two Indian languages:

- **Model H** (higher-resource): Hindi
- **Model L** (lower-resource, allowed list): Nepali

Both are Devanagari-script languages. Per the course spec, the two models do
**not** share data, tokenizer, vocabulary, or weights — each language's
pipeline lives in its own self-contained directory (`hindi/`, `nepali/`)
and is run independently end to end.

For Hindi's full written report with all 10 required figures, see
[`report/phase1/phase1_report.md`](report/phase1/phase1_report.md) (or the
LaTeX/PDF version, [`report/phase1/phase1_report.tex`](report/phase1/phase1_report.tex) /
[`phase1_report.pdf`](report/phase1/phase1_report.pdf)). For Nepali's, see
[`report/phase1/phase1_report_nepali.md`](report/phase1/phase1_report_nepali.md).
For a file-by-file deliverables index, see
[`report/phase1/README.md`](report/phase1/README.md). This page summarizes
everything below.

---

## Phase 1 — Dataset Collection & Tokenizer Construction (complete)

`scripts/validate_phase1.py` reports **ALL CHECKS PASSED** for Hindi.
`scripts/validate_phase1_nepali.py` (Nepali uses a different pipeline/data
layout, so it has its own parallel validator) reports **ALL CHECKS PASSED**
for Nepali except the cross-language check, which that script deliberately
flags `SKIPPED` rather than running inline (too large — ~9.2GB — to check
as a validator side effect); that check was run separately as its own pass,
see `scripts/check_nepali_hindi_overlap.py` and the result recorded in
`report/phase1/phase1_report_nepali.md` §6.

### Deliverables

| # | Deliverable | Where |
|---|---|---|
| 1 | Dataset collection scripts (both languages) | `hindi/data/pipeline/{download_hindi.py, ingest_manual.py, ingest_wikipedia_dump.py}`; `nepali/scripts/{s01_collect.py, collect.py, auto_grow_and_tokenize.py, corpuslib/}` |
| 2 | Dataset preprocessing pipelines | `hindi/data/pipeline/clean_hindi.py`, `scripts/clean_lib.py` (shared); `nepali/scripts/{s03_normalize.py, s04_langid.py, s05_heuristics.py, s06_perplexity.py, s08_dedup.py, s09_decontaminate.py, s10_balance_lanes.py, s11_finalize_splits.py}` |
| 3 | Per-language dataset statistics reports | `report/phase1/phase1_report.{md,tex,pdf}` (Hindi), `report/phase1/phase1_report_nepali.md` (Nepali) |
| 4 | Per-language train/validation/test splits | `hindi/data/splits/{train,validation,test}/` + manifests; `nepali/data/final/{train,val,test}.jsonl` + `split_stats.json` |
| 5 | Tokenizer training code | `hindi/tokenizer/train_tokenizer.py`; `nepali/scripts/{s12_train_tokenizer.py, s13_tokenizer_eval.py}` |
| 6 | Vocabulary files | `hindi/tokenizer/hindi_unigram_8000.vocab`; `nepali/tokenizer/nepali_bpe.vocab` (plus other swept sizes in `nepali/tokenizer/sweep/`) |
| 7 | Tokenizer model files | `hindi/tokenizer/hindi_unigram_8000.model`; `nepali/tokenizer/nepali_bpe.model` |

### Hindi (Model H)

| | |
|---|---:|
| Raw | 773,606 docs / 1,840,357,147 chars |
| Cleaned | 704,101 docs / 1,764,596,782 chars (91.0% doc / 95.9% char retention) |
| Manual share | 28.65% char-weighted, 30.66% token-weighted |
| Splits (train/val/test) | 675,937 / 14,082 / 14,082, seed 42, no leakage |
| Tokenizer | SentencePiece Unigram, swept 5k/8k/10k, **8,000 selected** |
| Total tokens | **505,852,028 — 101.2% of the 500M target** |

**Resources used, manual:** `hi_wikipedia_dump` (self-extracted Wikipedia
XML dump via WikiExtractor — largest single manual source), `gadya_kosh`
(prose wiki), `patrika_hindi` (newspaper archive), `kavita_kosh` (poetry
wiki), 67 misc. `archive.org` items (debate transcripts, books).
**Resources used, downloaded:** `HuggingFaceFW/fineweb-2:hin_Deva`,
`ai4bharat/sangraha:unverified/hin`,
`ai4bharat/sangraha:verified/hin:{web,pdf,speech}`. Excluded: the
HF-mirrored `wikimedia/wikipedia:20231101.hi` (redundant with the
self-extracted dump — would not count as manual regardless).

### Nepali (Model L)

| | |
|---|---:|
| Raw (manual + downloaded lanes) | ~2.66M docs pre-normalise (S03 input) |
| Final (train+val+test) | 1,918,265 docs / 635,204,333 tokens |
| Manual share | 26.41% token-weighted (see report §2 for why token- not char-weighted) |
| Splits (train/val/test) | 1,520,000 / 199,428 / 198,837, no leakage |
| Tokenizer | BPE, swept 5k/6k/8k/10k, **10,000 selected** |
| Total tokens | **635,204,333 — 127.0% of the 500M target** (both absolute lane targets exceeded: manual 167.8M ≥ 100M, downloaded 467.4M ≥ 400M) |

**Resources used, manual:** `news_onlinekhabar` (web scrape, 91.6% of
manual tokens), `wikipedia_api` (self-parsed XML dump), `news_bbc_nepali`,
`wikibooks_api`, `wiktionary_api`, `news_ekantipur`, `news_ratopati`,
`news_nagarik`, `ocr_archive_books`. **Resources used, downloaded:**
`ai4bharat/sangraha` (`verified` + `unverified`), `ai4bharat/IndicCorpV2`.

Full per-source tables, cleaning-stage retention funnel, drop-reason
breakdowns, tokenizer sweep table, and all 10 figures are in
[`report/phase1/phase1_report_nepali.md`](report/phase1/phase1_report_nepali.md)
and `report/phase1/figures/nepali/`.

**Provenance note:** Nepali's pipeline was run on a separate machine (the
~30+ hour S08 MinHash dedup pass was impractical on the primary machine
within the deadline window); only output artifacts were transferred into
this repo, not live per-stage logs — `log.md`'s S08(production)–S14
entries are reconstructed from those artifacts, flagged inline as such.

---

## Repository layout

```
<repo>/
  README.md                    # this file
  requirements.txt
  hindi/
    data/
      raw/{manual,downloaded}/ # source text + metadata.jsonl (provenance)
      interim/                 # post-normalization, pre-language-filter snapshot
      cleaned/                 # final cleaned corpus
      splits/                  # train/validation/test (document-level split)
      metadata/                # clean_stats.json, dropped_documents.jsonl
      pipeline/                # download/ingest/clean/split scripts
    tokenizer/                 # SentencePiece model + vocab (Phase 1 deliverable)
    model/ train/ eval/ configs/   # Phase 2+
  nepali/                      # Model L. Different pipeline shape (stage-based
                                # S00-S14 under scripts/, corpuslib/ shared lib)
                                # but the same required subfolders:
    data/raw/{manual,downloaded}/, data/final/{train,val,test}.jsonl
    tokenizer/                 # nepali_bpe.model/.vocab + sweep/
    model/ train/ eval/ configs/   # Phase 2+
  scripts/                     # shared pipeline code (clean_lib.py, crawlers,
                                # validate_phase1.py, validate_phase1_nepali.py,
                                # check_nepali_hindi_overlap.py)
  report/
    phase1/                    # README.md (deliverables index), phase1_report.{md,tex,pdf}
                                # (Hindi), phase1_report_nepali.md (Nepali),
                                # dataset_statistics.md, tokenizer_analysis.md,
                                # make_plots.py / make_plots_nepali.py, figures/
    phase2/ phase3/
  bhojpuri/                    # earlier, fully-complete Model L attempt,
                                # superseded by Nepali — kept as-is, not the
                                # graded submission — see "Bhojpuri (superseded)" below
```

## Reproduction steps

1. **Environment**

   ```
   python -m venv .venv
   .venv/Scripts/activate      # or: source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Hindi pipeline**:

   ```
   python hindi/data/pipeline/download_hindi.py       # public/HF corpora -> raw/downloaded/
   python hindi/data/pipeline/ingest_manual.py         # scraped/self-extracted sources -> raw/manual/
   python hindi/data/pipeline/ingest_wikipedia_dump.py # self-extracted Wikipedia XML dump -> raw/manual/
   python hindi/data/pipeline/clean_hindi.py           # raw/ -> interim/ -> cleaned/ (full rebuild by default)
   python hindi/data/pipeline/split_hindi.py           # cleaned/ -> splits/{train,validation,test}/
   python hindi/tokenizer/train_tokenizer.py --vocab-size 8000
   ```

   `clean_hindi.py` always rebuilds `interim/`/`cleaned/` from `raw/` from
   scratch; pass `--resume` only to continue an interrupted run without
   reprocessing already-seen documents. `clean_hindi.py`/`split_hindi.py`
   take an OS-level single-instance lock (`data/metadata/.clean.lock` /
   `.split.lock`) — delete a stale lock older than 15 minutes if a run was
   killed uncleanly.

3. **Nepali pipeline** (stage-based, config-driven — different shape from
   Hindi's, see `nepali/scripts/` docstrings for each stage):

   ```
   cd nepali/scripts
   python s00_doctor.py                # environment/config sanity check
   python auto_grow_and_tokenize.py    # full S02-S14: collect -> clean -> dedup ->
                                        # decontaminate -> balance -> split -> tokenize,
                                        # growing whichever lane (manual/downloaded) falls
                                        # short of its absolute token target, up to 4 rounds
   ```

   Or run stages individually (`s01_collect.py` ... `s14_corpus_stats.py`)
   via `python run_pipeline.py <stage>` for finer control. S08 (MinHash/LSH
   dedup over ~2.2M documents) is the slow stage — budget several hours.

4. **Validate Phase 1 deliverables**:

   ```
   python scripts/validate_phase1.py            # Hindi
   python scripts/validate_phase1_nepali.py     # Nepali (different data layout)
   python scripts/check_nepali_hindi_overlap.py # cross-language: Nepali vs. Hindi
   ```

5. **Regenerate the report figures**:

   ```
   python report/phase1/make_plots.py           # Hindi (+ Bhojpuri, superseded)
   python report/phase1/make_plots_nepali.py    # Nepali
   ```

## Large artifacts (OneDrive)

Raw corpora, cleaned corpora, and tokenizer files are too large for git and
are hosted on OneDrive instead:

**[LMA DATA SET — shared OneDrive folder](https://iiithydstudents-my.sharepoint.com/personal/swaraj_kumar_students_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fswaraj%5Fkumar%5Fstudents%5Fiiit%5Fac%5Fin%2FDocuments%2FLMA%20DATA%20SET&ga=1)**
(requires an IIIT account with access)

| Artifact | Language | Status |
|---|---|---|
| Raw corpus (manual + downloaded) | Hindi | NOT YET CONFIRMED at this link |
| Cleaned corpus | Hindi | NOT YET CONFIRMED at this link |
| Tokenizer files | Hindi | NOT YET CONFIRMED at this link |
| Raw corpus (manual + downloaded) | Nepali | NOT YET UPLOADED (~9.8GB raw + ~2.1MB interim) |
| Final corpus (train/val/test) | Nepali | NOT YET UPLOADED (~9.2GB) |
| Tokenizer files | Nepali | committed directly to git (small: model 495KB + vocab) |
| Raw/cleaned corpus, tokenizer files | Bhojpuri (superseded) | NOT YET CONFIRMED at this link |
| Pretrained checkpoint (Model H) | Hindi | TODO — Phase 2 |
| Pretrained checkpoint (Model L) | Nepali | TODO — Phase 2 |
| Finetuned checkpoint (Model H) | Hindi | TODO — Phase 3 |
| Finetuned checkpoint (Model L) | Nepali | TODO — Phase 3 |

**Note on hosting:** the course spec names Google Drive specifically for
large artifacts. This project uses an IIIT-institutional OneDrive link
instead — a deliberate choice, not an oversight — since it's the storage
already available and accessible to the course's IIIT-affiliated graders.

## Hard constraints followed

No pretrained models, no pretrained tokenizers, no HuggingFace Transformer
model classes. Positional embeddings, multi-head attention, and causal
masking are student-implemented from primitive PyTorch layers (Phase 2). The
two language corpora, tokenizers, vocabularies, and (from Phase 2 onward)
model weights are fully independent — no sharing, no multilingual training.

---

## Bhojpuri (superseded)

Bhojpuri was the original Model L: fully built, cleaned, split, tokenized,
and passing every `validate_phase1.py` check. It has since been superseded
by **Nepali** (also on the assignment's allowed list) as the graded Model
L, once Nepali's own pipeline finished collecting enough data to clear both
its manual and downloaded absolute token targets — see "Nepali (Model L)"
above and `report/phase1/phase1_report_nepali.md` for the full writeup.

Bhojpuri's numbers, for the record: 507,994 cleaned docs, 282,815,177
chars, 28.59% char-weighted manual share, 84,217,032 total tokens (16.8% of
the 500M target — a documented, spec-permitted shortfall, since all known
manual sources were confirmed exhausted; see
`report/phase1/dataset_statistics.md`). `bhojpuri/` and its report content
(`report/phase1/phase1_report.md` covers both Hindi and Bhojpuri together)
are left exactly as they were — nothing deleted, nothing re-run — as a
complete, working fallback and as a record of that earlier work.

`nepali/scripts/`, `nepali/configs/` (the pipeline code) are versioned in
this repository the same way `hindi/`'s and `bhojpuri/`'s own pipeline code
are; `nepali/data/` (~20GB of raw/interim/final pipeline output) is not,
for the same "code in git, corpus data elsewhere" reason those two
languages' raw/cleaned corpora are not versioned either.

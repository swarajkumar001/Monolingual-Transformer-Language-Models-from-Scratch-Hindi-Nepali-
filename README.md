<div align="center">

# Language Models and Agents — Individual Project

**Two independent, from-scratch, decoder-only Transformer language models, one per Indian language.**

![Phase 1](https://img.shields.io/badge/Phase%201-Complete-brightgreen)
![Model H](https://img.shields.io/badge/Model%20H-Hindi-1f6f8b)
![Model L](https://img.shields.io/badge/Model%20L-Nepali-7a4fa3)
![Script](https://img.shields.io/badge/Script-Devanagari-orange)
![Python](https://img.shields.io/badge/Python-3.14-blue)
![Tokenizer](https://img.shields.io/badge/Tokenizer-SentencePiece-lightgrey)

</div>

Per the course spec, the two models do **not** share data, tokenizer,
vocabulary, or weights — each language's pipeline lives in its own
self-contained directory (`hindi/`, `nepali/`) and is run independently
end to end. Hindi and Nepali also run on two structurally different
*pipelines* (Hindi: a single-pass cleaner; Nepali: a stage-based S00–S14
pipeline with MinHash/LSH dedup and absolute per-lane token targets), so
most of the linked report describes each language separately rather than
forcing one shared table.

**📄 [Full Phase 1 report](report/phase1/phase1_report.md)** ([LaTeX](report/phase1/phase1_report.tex) · [PDF](report/phase1/phase1_report.pdf)) · **📋 [Deliverables index](report/phase1/README.md)**

---

## At a glance

| | 🔵 Hindi — Model H | 🟣 Nepali — Model L |
|---|---:|---:|
| Pipeline | single-pass cleaner | stage-based, S00–S14 |
| Raw docs | 773,606 | ~2,655,504 (pre-clean) |
| Final docs | 704,101 | 1,918,265 |
| **Total tokens** | **505,852,028** | **635,204,333** |
| **% of 500M target** | **101.2%** | **127.0%** |
| Manual share | 28.65% char / 30.66% token | 26.41% token |
| Splits (train / val / test) | 675,937 / 14,082 / 14,082 | 1,520,000 / 199,428 / 198,837 |
| Tokenizer | SentencePiece **Unigram**, **8,000** | SentencePiece **BPE**, **10,000** |
| Leakage check | ✅ no overlap | ✅ no overlap |
| Cross-language check | ✅ 0 shared docs vs. Nepali | ✅ 0 shared docs vs. Hindi |

<div align="center">
<img src="report/phase1/figures/10_token_count_vs_target.png" width="47%" alt="Token count vs. 500M target">
<img src="report/phase1/figures/02_manual_share.png" width="47%" alt="Manual vs. downloaded share vs. 20% floor">
</div>

All 10 required figures (both languages, one series/subplot per language)
are in [`report/phase1/figures/`](report/phase1/figures/) and embedded in
the [full report](report/phase1/phase1_report.md).

---

## Phase 1 — Dataset Collection & Tokenizer Construction ✅ complete

`scripts/validate_phase1.py` reports **ALL CHECKS PASSED** for Hindi.
`scripts/validate_phase1_nepali.py` (Nepali's own parallel validator, since
its pipeline/data layout differs from Hindi's) reports **ALL CHECKS
PASSED** for Nepali except the cross-language check, which it deliberately
flags `SKIPPED` rather than running inline (too large — ~9.2GB — to check
as a validator side effect). That check was run separately —
`scripts/check_nepali_hindi_overlap.py` — result: **0 shared documents**,
recorded in [`phase1_report.md` §3](report/phase1/phase1_report.md).

### Deliverables

| # | Deliverable | Where |
|---|---|---|
| 1 | Dataset collection scripts (both languages) | `hindi/data/pipeline/{download_hindi.py, ingest_manual.py, ingest_wikipedia_dump.py}`; `nepali/scripts/{s01_collect.py, collect.py, auto_grow_and_tokenize.py, corpuslib/}` |
| 2 | Dataset preprocessing pipelines | `hindi/data/pipeline/clean_hindi.py`, `scripts/clean_lib.py` (shared); `nepali/scripts/{s03_normalize.py, s04_langid.py, s05_heuristics.py, s06_perplexity.py, s08_dedup.py, s09_decontaminate.py, s10_balance_lanes.py, s11_finalize_splits.py}` |
| 3 | Per-language dataset statistics reports | `report/phase1/phase1_report.{md,tex,pdf}` (both languages), `report/phase1/dataset_statistics.md`, `report/phase1/tokenizer_analysis.md` |
| 4 | Per-language train/validation/test splits | `hindi/data/splits/{train,validation,test}/` + manifests; `nepali/data/final/{train,val,test}.jsonl` + `split_stats.json` |
| 5 | Tokenizer training code | `hindi/tokenizer/train_tokenizer.py`; `nepali/scripts/{s12_train_tokenizer.py, s13_tokenizer_eval.py}` |
| 6 | Vocabulary files | `hindi/tokenizer/hindi_unigram_8000.vocab`; `nepali/tokenizer/nepali_bpe.vocab` (plus other swept sizes in `nepali/tokenizer/sweep/`) |
| 7 | Tokenizer model files | `hindi/tokenizer/hindi_unigram_8000.model`; `nepali/tokenizer/nepali_bpe.model` |

### 🔵 Hindi (Model H)

**Resources, manual:** `hi_wikipedia_dump` (self-extracted Wikipedia XML
dump via WikiExtractor — largest single manual source), `gadya_kosh`
(prose wiki), `patrika_hindi` (newspaper archive), `kavita_kosh` (poetry
wiki), 67 misc. `archive.org` items (debate transcripts, books).
**Resources, downloaded:** `HuggingFaceFW/fineweb-2:hin_Deva`,
`ai4bharat/sangraha:unverified/hin`,
`ai4bharat/sangraha:verified/hin:{web,pdf,speech}`. Excluded: the
HF-mirrored `wikimedia/wikipedia:20231101.hi` (redundant with the
self-extracted dump — would not count as manual regardless).

### 🟣 Nepali (Model L)

Nepali's pipeline gates on two **absolute** token targets rather than a
ratio: manual ≥ 100M, downloaded ≥ 400M — both cleared (manual 167.8M,
downloaded 467.4M).

**Resources, manual:** `news_onlinekhabar` (web scrape, 91.6% of manual
tokens), `wikipedia_api` (self-parsed XML dump), `news_bbc_nepali`,
`wikibooks_api`, `wiktionary_api`, `news_ekantipur`, `news_ratopati`,
`news_nagarik`, `ocr_archive_books`. **Resources, downloaded:**
`ai4bharat/sangraha` (`verified` + `unverified`), `ai4bharat/IndicCorpV2`.

> **Provenance note:** Nepali's pipeline was run on a separate machine
> (the ~30+ hour S08 MinHash dedup pass was impractical on the primary
> machine within the deadline window); only output artifacts were
> transferred into this repo, not live per-stage logs — `log.md`'s
> S08(production)–S14 entries are reconstructed from those artifacts,
> flagged inline as such.

---

## Repository layout

<details>
<summary>Click to expand</summary>

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
                                # (both languages), dataset_statistics.md,
                                # tokenizer_analysis.md, make_plots.py, figures/
    phase2/ phase3/
  logs/                        # every *.log produced by the pipelines (gitignored)
  # bhojpuri/ removed — earlier, fully-complete Model L attempt,
  # superseded by Nepali; see "Bhojpuri (superseded)" below for its numbers
```

</details>

## Reproduction steps

<details>
<summary>Click to expand</summary>

**1. Environment**

```
python -m venv .venv
.venv/Scripts/activate      # or: source .venv/bin/activate
pip install -r requirements.txt
```

**2. Hindi pipeline**

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

**3. Nepali pipeline** (stage-based, config-driven — different shape from
Hindi's, see `nepali/scripts/` docstrings for each stage)

```
cd nepali/scripts
python s00_doctor.py                # environment/config sanity check
python auto_grow_and_tokenize.py    # full S02-S14: collect -> clean -> dedup ->
                                     # decontaminate -> balance -> split -> tokenize,
                                     # growing whichever lane (manual/downloaded) falls
                                     # short of its absolute token target, up to 4 rounds
```

Or run stages individually (`s01_collect.py` ... `s14_corpus_stats.py`) via
`python run_pipeline.py <stage>` for finer control. S08 (MinHash/LSH dedup
over ~2.2M documents) is the slow stage — budget several hours.

**4. Validate Phase 1 deliverables**

```
python scripts/validate_phase1.py            # Hindi
python scripts/validate_phase1_nepali.py     # Nepali (different data layout)
python scripts/check_nepali_hindi_overlap.py # cross-language: Nepali vs. Hindi
```

**5. Regenerate the report figures** (both languages, one script — it
live-measures each language's tokenizer sweep against its own held-out
validation split)

```
python report/phase1/make_plots.py
```

</details>

## Large artifacts (OneDrive)

Raw corpora, cleaned corpora, and tokenizer files are too large for git and
are hosted on OneDrive instead:

**[LMA DATA SET — shared OneDrive folder](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/swaraj_kumar_students_iiit_ac_in/IgCh3u8O2sr4QIJqmrUS1CllAV1C5HL0R9SOWUUVIzTgcdw?e=bgUHaG)**
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

> **Note on hosting:** the course spec names Google Drive specifically for
> large artifacts. This project uses an IIIT-institutional OneDrive link
> instead — a deliberate choice, not an oversight — since it's the
> storage already available and accessible to the course's
> IIIT-affiliated graders.

## Hard constraints followed

No pretrained models, no pretrained tokenizers, no HuggingFace Transformer
model classes. Positional embeddings, multi-head attention, and causal
masking are student-implemented from primitive PyTorch layers (Phase 2). The
two language corpora, tokenizers, vocabularies, and (from Phase 2 onward)
model weights are fully independent — no sharing, no multilingual training.

---

## Bhojpuri (superseded)

<details>
<summary>Click to expand — Bhojpuri was this project's original Model L</summary>

Bhojpuri was the original Model L: fully built, cleaned, split, tokenized,
and passing every `validate_phase1.py` check, with a documented,
spec-permitted 16.8%-of-target shortfall (84,217,032 tokens). It was
superseded by **Nepali** once Nepali's own pipeline cleared both its
absolute token targets — see [`phase1_report.md` §6](report/phase1/phase1_report.md)
for the full history and Bhojpuri's numbers.

Its tracked pipeline code and tokenizer files (`bhojpuri/`) have since
been removed from the repository; `bhojpuri/data/` (the large, already-
gitignored raw/cleaned corpus) was never tracked in git regardless. Its
measured numbers remain in `phase1_report.md` as the historical record.

`nepali/scripts/`, `nepali/configs/` (the pipeline code) are versioned in
this repository; `nepali/data/` (~20GB of raw/interim/final pipeline
output) is not, for the same "code in git, corpus data elsewhere" reason.

</details>

# Language Models and Agents — Individual Project

Two independent, from-scratch, decoder-only Transformer language models in
two Indian languages:

- **Model H** (higher-resource): Hindi
- **Model L** (lower-resource, allowed list): Bhojpuri

Both are Devanagari-script languages. Per the course spec, the two models do
**not** share data, tokenizer, vocabulary, or weights — each language's
pipeline lives in its own self-contained directory (`hindi/`, `bhojpuri/`)
and is run independently end to end.

For the full written report with all 10 required figures, see
[`report/phase1/phase1_report.md`](report/phase1/phase1_report.md) (or the
LaTeX/PDF version, [`report/phase1/phase1_report.tex`](report/phase1/phase1_report.tex) /
[`phase1_report.pdf`](report/phase1/phase1_report.pdf)). For a file-by-file
deliverables index, see [`report/phase1/README.md`](report/phase1/README.md).
This page summarizes everything below.

---

## Phase 1 — Dataset Collection & Tokenizer Construction (complete)

`scripts/validate_phase1.py` reports **ALL CHECKS PASSED** for both
languages: raw sources present, manual share ≥ 20%, splits exist with no
leakage, no cross-language document overlap, tokenizers trained.

### Deliverables

| # | Deliverable | Where |
|---|---|---|
| 1 | Dataset collection scripts (both languages) | `hindi/data/pipeline/{download_hindi.py, ingest_manual.py, ingest_wikipedia_dump.py}`, `bhojpuri/data/pipeline/{download_bhojpuri.py, download_bhojpuri_extra.py, ingest_bhltr.py, ingest_harsh_corpus.py, ingest_manual.py, ingest_wikipedia_dump.py}`, `scripts/crawl_sites.py`, `scripts/fetch_archive_org.py` |
| 2 | Dataset preprocessing pipelines | `hindi/data/pipeline/clean_hindi.py`, `bhojpuri/data/pipeline/clean_bhojpuri.py`, `scripts/clean_lib.py` (shared), `scripts/remove_cross_language_duplicates.py` |
| 3 | Per-language dataset statistics reports | `report/phase1/phase1_report.{md,tex,pdf}`, `report/phase1/dataset_statistics.md`, `report/phase1/tokenizer_analysis.md` |
| 4 | Per-language train/validation/test splits | `hindi/data/splits/{train,validation,test}/` + manifests, `bhojpuri/data/splits/{train,validation,test}/` + manifests |
| 5 | Tokenizer training code | `hindi/tokenizer/train_tokenizer.py`, `bhojpuri/tokenizer/train_tokenizer.py` |
| 6 | Vocabulary files | `hindi/tokenizer/hindi_unigram_8000.vocab`, `bhojpuri/tokenizer/bhojpuri_unigram_8000.vocab` (plus other swept sizes) |
| 7 | Tokenizer model files | `hindi/tokenizer/hindi_unigram_8000.model`, `bhojpuri/tokenizer/bhojpuri_unigram_8000.model` |

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

### Bhojpuri (Model L)

| | |
|---|---:|
| Raw | 608,263 docs / 318,606,726 chars |
| Cleaned | 507,994 docs / 282,815,177 chars (83.5% doc / 88.8% char retention) |
| Manual share | 28.59% char-weighted, 29.89% token-weighted |
| Splits (train/val/test) | 487,674 / 10,160 / 10,160, seed 42, no leakage |
| Tokenizer | SentencePiece Unigram, swept 5k/6k/8k/10k, **8,000 selected** |
| Total tokens | **84,217,032 — 16.8% of the 500M target** (documented, spec-permitted shortfall — see below) |

**Resources used, manual:** four dedicated crawl sources
(`khabar_bhojpuri`, `anjoria`, `jogira`, `sirijan` — recrawled uncapped,
confirmed exhausted), `bho_wikipedia_dump` (self-extracted), 88 misc.
`archive.org` items, `harsh_scraped_manual` (prior personal scrape).
**Resources used, downloaded:** 17 HF datasets/partitions — largest is the
`Satyam810/BhojpuriCorpus` family, plus `harsh_indiccorp` (personal prior
collection), JNU's [BHLTR corpus](https://github.com/shashwatup9k/BHLTR),
and `HuggingFaceFW/fineweb-2:bho_Deva`. Six other HF datasets were
evaluated and rejected: CulturaX/sil-ai-bloom-lm (gated), Sangraha/Glot500
(zero Bhojpuri rows), MADLAD-400 (no `bho` config), CC-100 (unsupported
legacy format).

Bhojpuri's shortfall against the 500M target is expected and
spec-permitted for the low-resource model, not a gap in effort — all
known manual sources are confirmed exhausted, and this corpus is already
orders of magnitude larger than any previously published Bhojpuri
monolingual resource (the largest prior public resource, JNU's BHLTR, is
~45,000 sentences). Full justification in `report/phase1/dataset_statistics.md`.

Full per-source tables, cleaning drop-reason breakdowns, tokenizer sweep
tables, and all 10 required figures are in
[`report/phase1/phase1_report.md`](report/phase1/phase1_report.md).

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
  bhojpuri/                    # mirrors the hindi/ layout exactly
  scripts/                     # shared pipeline code (clean_lib.py, crawlers,
                                # validate_phase1.py) used by both languages
  report/
    phase1/                    # README.md (deliverables index), phase1_report.{md,tex,pdf},
                                # dataset_statistics.md, tokenizer_analysis.md,
                                # make_plots.py, figures/
    phase2/ phase3/
  nepali/                      # exploratory Model L candidate, NOT the graded
                                # submission — see "Nepali (exploratory)" below
```

## Reproduction steps

1. **Environment**

   ```
   python -m venv .venv
   .venv/Scripts/activate      # or: source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Per-language pipeline** (repeat for `hindi/` and `bhojpuri/`, scripts
   are not interchangeable between languages — each hardcodes its own
   language ISO code, marker lists, and directory tree):

   ```
   python <lang>/data/pipeline/download_<lang>.py       # public/HF corpora -> raw/downloaded/
   python <lang>/data/pipeline/ingest_manual.py          # scraped/self-extracted sources -> raw/manual/
   python <lang>/data/pipeline/ingest_wikipedia_dump.py  # self-extracted Wikipedia XML dump -> raw/manual/
   python <lang>/data/pipeline/clean_<lang>.py           # raw/ -> interim/ -> cleaned/ (full rebuild by default)
   python <lang>/data/pipeline/split_<lang>.py           # cleaned/ -> splits/{train,validation,test}/
   ```

   `clean_<lang>.py` always rebuilds `interim/`/`cleaned/` from `raw/` from
   scratch; pass `--resume` only to continue an interrupted run without
   reprocessing already-seen documents. Both `clean_<lang>.py` and
   `split_<lang>.py` take an OS-level single-instance lock
   (`data/metadata/.clean.lock` / `.split.lock`) — if a run is killed
   uncleanly, delete the stale lock file before rerunning (a lock older than
   15 minutes is reclaimed automatically).

3. **Tokenizer training** (per language, sweep over vocab sizes):

   ```
   python <lang>/tokenizer/train_tokenizer.py --vocab-size 8000
   python <lang>/tokenizer/train_tokenizer.py --vocab-size 10000
   ...
   ```

   Trained only on each language's `data/splits/train/` — never
   validation or test. Vocab size is capped at 5,000–10,000 per a
   course-TA ruling (see `report/phase1/tokenizer_analysis.md`).

4. **Validate Phase 1 deliverables**:

   ```
   python scripts/validate_phase1.py
   ```

   Checks (per language): raw manual+downloaded sources present, manual
   share ≥ 20% (computed from `source_type` provenance, not a hardcoded
   count), train/validation/test splits exist with no document overlap,
   and a tokenizer has been trained. Also checks no document (by ID or
   content hash) is shared between the two language corpora. Prints
   PASS/FAIL per check.

5. **Regenerate the report figures**:

   ```
   python report/phase1/make_plots.py
   ```

## Large artifacts (OneDrive)

Raw corpora, cleaned corpora, and tokenizer files are too large for git and
are hosted on OneDrive instead:

**[LMA DATA SET — shared OneDrive folder](https://iiithydstudents-my.sharepoint.com/personal/swaraj_kumar_students_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fswaraj%5Fkumar%5Fstudents%5Fiiit%5Fac%5Fin%2FDocuments%2FLMA%20DATA%20SET&ga=1)**
(requires an IIIT account with access)

| Artifact | Language | Status |
|---|---|---|
| Raw corpus (manual + downloaded) | Bhojpuri | NOT YET CONFIRMED at this link |
| Cleaned corpus | Bhojpuri | NOT YET CONFIRMED at this link |
| Tokenizer files | Bhojpuri | NOT YET CONFIRMED at this link |
| Raw corpus (manual + downloaded) | Hindi | NOT YET CONFIRMED at this link |
| Cleaned corpus | Hindi | NOT YET CONFIRMED at this link |
| Tokenizer files | Hindi | NOT YET CONFIRMED at this link |
| Pretrained checkpoint (Model H) | Hindi | TODO — Phase 2 |
| Pretrained checkpoint (Model L) | Bhojpuri | TODO — Phase 2 |
| Finetuned checkpoint (Model H) | Hindi | TODO — Phase 3 |
| Finetuned checkpoint (Model L) | Bhojpuri | TODO — Phase 3 |

## Hard constraints followed

No pretrained models, no pretrained tokenizers, no HuggingFace Transformer
model classes. Positional embeddings, multi-head attention, and causal
masking are student-implemented from primitive PyTorch layers (Phase 2). The
two language corpora, tokenizers, vocabularies, and (from Phase 2 onward)
model weights are fully independent — no sharing, no multilingual training.

---

## Nepali (exploratory, not graded)

Alongside the graded Model L (Bhojpuri), a second, independent low-resource
candidate — **Nepali**, also on the assignment's allowed list — was
attempted using a more elaborate, config-driven pipeline
(`nepali/scripts/run_pipeline.py`, stages S02–S14). This was exploratory:
Nepali has more public data available than Bhojpuri, so it was worth
checking whether it could clear the 20% manual-share floor with a stronger
overall corpus.

**What happened.** An early run of that pipeline measured only 6.3%
manual share — failing the floor — but this turned out to be a **stale
measurement**: most of the raw manual data (in particular a
200,804-document news-scrape source, `onlinekhabar`, ~72M words) had not
finished collecting when that run happened. After confirming ~84.4M raw
manual words were actually on disk (versus the ~15.4M that fed the stale
run), the pipeline was relaunched from a clean state against the complete
raw corpus.

**Progress on the rerun** (from the pipeline's own event log, `log.md`):
S02 token calibration, S03 normalize (1,658,065 docs, 0 dropped), S04
language-ID (99.77% retention), S07 holdout split (train 1,406,033 / val
124,042 / test 124,221), S05 heuristic filters (99.72% retention), S06
perplexity filter (97.04% retention) — all succeeded. It then **stalled at
S08, deduplication** — a MinHash/LSH pass over ~1.36M documents that, at
its measured pace, was on track to take 10+ hours to finish, before S09
through S14 (decontamination, lane balancing, final split, tokenizer
training/evaluation, corpus stats) could even start. That was not a safe
bet against the Phase 1 deadline.

Two real bugs were found and fixed along the way: `fasttext` had no
prebuilt wheel for this environment's Python 3.14, so the S04 stage alone
was routed through a separate Python 3.11 install (the language-ID logic
itself was not bypassed, only its runtime); and
`nepali/configs/tokenizer.yaml` was misconfigured to sweep a 16k–40k BPE
vocabulary, violating the same course-TA ruling covered above — corrected
to `[5000, 6000, 8000, 10000]` before the tokenizer stage could run.

**Bhojpuri is the final, graded Model L.** It was fully built, cleaned,
split, tokenized, and passing every `validate_phase1.py` check before the
Nepali rerun was even relaunched, and nothing in `report/phase1/` depends
on Nepali finishing. `nepali/scripts/` and `nepali/configs/` (the pipeline
code) are versioned in this repository for transparency into that
exploration; `nepali/data/` (~20GB of raw/interim pipeline output) is not,
for the same "code in git, corpus data elsewhere" reason Hindi and
Bhojpuri's own raw/cleaned corpora are not versioned either.

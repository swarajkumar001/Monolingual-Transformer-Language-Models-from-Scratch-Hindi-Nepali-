# Language Models and Agents — Individual Project

Two independent, from-scratch, decoder-only Transformer language models in
two Indian languages:

- **Model H** (higher-resource): Hindi
- **Model L** (lower-resource, allowed list): Bhojpuri

Both are Devanagari-script languages. Per the course spec, the two models do
**not** share data, tokenizer, vocabulary, or weights — each language's
pipeline lives in its own self-contained directory (`hindi/`, `bhojpuri/`)
and is run independently end to end.

**Start here for Phase 1:** [`report/phase1/README.md`](report/phase1/README.md)
walks through every deliverable (collection scripts, cleaning pipeline,
statistics report, splits, tokenizer code, vocab files, model files) with
exact numbers and file paths. [`report/phase1/phase1_report.md`](report/phase1/phase1_report.md)
is the full written report with all 10 required figures embedded.

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
    phase1/                    # README.md (start here), phase1_report.md,
                                # dataset_statistics.md, tokenizer_analysis.md,
                                # make_plots.py, figures/
    phase2/ phase3/
  nepali/                      # exploratory Model L candidate, NOT the graded
                                # submission — see "About the Nepali attempt" below
```

## Status (Phase 1, complete)

See [`report/phase1/README.md`](report/phase1/README.md) for the full
deliverables index, or [`report/phase1/phase1_report.md`](report/phase1/phase1_report.md)
for the complete written report with figures. `scripts/validate_phase1.py`
reports ALL CHECKS PASSED for both languages. In short:

- **Hindi**: 704,101 cleaned docs / 1,764,596,782 chars; 28.65% manual
  char-weighted (30.66% token-weighted), clears the 20% floor. Tokenizer
  swept at 5k/8k/10k, unigram vocab_size=8,000 selected.
  505,852,028 tokens (101.2% of the 500M target).
- **Bhojpuri**: 507,994 cleaned docs / 282,815,177 chars; 28.59% manual
  char-weighted (29.89% token-weighted), clears the 20% floor. All known
  manual sources confirmed exhausted. Tokenizer swept at 5k/6k/8k/10k,
  unigram vocab_size=8,000 selected. 84,217,032 tokens — a documented,
  spec-permitted shortfall against the 500M target for the low-resource
  language, not a gap in effort (see `report/phase1/dataset_statistics.md`
  for the exhausted-sources justification).

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

## Large artifacts (Google Drive)

Raw corpora, cleaned corpora, and tokenizer files are too large for git and
are hosted on Google Drive instead:

**[LMA DATASET — shared Drive folder](https://drive.google.com/drive/folders/1Te68Ys--Ak4Em946rI5h15NPYxew4DsS?usp=sharing)**

| Artifact | Language | Status |
|---|---|---|
| Raw corpus (manual + downloaded) | Bhojpuri | Uploaded (`BHOJPURI MODEL L/` in the folder above) |
| Cleaned corpus | Bhojpuri | Uploaded |
| Tokenizer files | Bhojpuri | Uploaded |
| Raw corpus (manual + downloaded) | Hindi | Upload pending |
| Cleaned corpus | Hindi | Upload pending |
| Tokenizer files | Hindi | Upload pending |
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

## About the Nepali attempt

Alongside the graded Model L (Bhojpuri), a second, independent low-resource
candidate — **Nepali**, also on the assignment's allowed list — was
attempted using a more elaborate, config-driven pipeline
(`nepali/scripts/run_pipeline.py`, stages S02–S14). This was exploratory:
Nepali has more public data available than Bhojpuri, so it was worth
checking whether it could clear the 20% manual-share floor with a stronger
overall corpus.

An early run of that pipeline measured only 6.3% manual share — failing the
floor — but this turned out to be a **stale measurement**: most of the raw
manual data (in particular a 200,804-document news-scrape source,
~72M words) had not finished collecting when that run happened. After
confirming ~84.4M raw manual words were actually on disk, the pipeline was
rerun from scratch against the complete raw corpus.

That rerun got through language-ID, quality filtering, and holdout
splitting cleanly, but stalled at the **deduplication stage (S08)** — a
MinHash/LSH pass over ~1.36M documents that, at its measured pace, would
have taken on the order of 10+ hours to complete, well past the Phase 1
deadline. Rather than risk the whole submission waiting on an unfinished
run, **Bhojpuri was kept as the final, graded Model L** — it was already
fully validated (all `validate_phase1.py` checks passing) before the Nepali
attempt began.

`nepali/` is left in the repository as-is for transparency into that
exploration, but it is **not part of the graded Phase 1 submission** — no
Nepali numbers appear in `report/phase1/`, and `scripts/validate_phase1.py`
does not check it.

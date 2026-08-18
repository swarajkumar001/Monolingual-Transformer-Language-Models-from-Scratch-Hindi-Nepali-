# Language Models and Agents — Individual Project

Two independent, from-scratch, decoder-only Transformer language models in
two Indian languages:

- **Model H** (higher-resource): Hindi
- **Model L** (lower-resource, allowed list): Bhojpuri

Both are Devanagari-script languages. Per the course spec, the two models do
**not** share data, tokenizer, vocabulary, or weights — each language's
pipeline lives in its own self-contained directory (`hindi/`, `bhojpuri/`)
and is run independently end to end.

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
    phase1/                    # dataset_statistics.md, tokenizer_analysis.md
    phase2/ phase3/
```

## Status (Phase 1, in progress)

See `report/phase1/dataset_statistics.md` for full, measured numbers. In
short:

- **Bhojpuri**: raw collection complete (all known manual sources
  confirmed exhausted); cleaning and manual-share validation done
  (28.59% manual, clears the 20% floor); tokenizer sweep in progress.
- **Hindi**: raw collection frozen at 1,840,357,147 chars (29.36% manual);
  full cleaning run in progress.

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
   python <lang>/tokenizer/train_tokenizer.py --vocab-size 16000
   ...
   ```

   Trained only on each language's `data/splits/train/` — never
   validation or test.

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

## Large artifacts (Google Drive)

Raw and cleaned corpora, and later, model checkpoints, are too large for git
and are hosted on Google Drive instead. Links below are filled in as each
artifact is finalized — do not treat an empty link as an oversight, it means
that stage is not complete yet.

| Artifact | Language | Link |
|---|---|---|
| Raw corpus (manual + downloaded) | Hindi | TODO |
| Raw corpus (manual + downloaded) | Bhojpuri | TODO |
| Cleaned corpus | Hindi | TODO |
| Cleaned corpus | Bhojpuri | TODO |
| Tokenizer files | Hindi | TODO |
| Tokenizer files | Bhojpuri | TODO |
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

# Personal Learning Guides — Phase 1 and Phase 2

> **Neither guide is a course deliverable.** They are not graded, not linked from
> the main `README.md`, and do not affect marks. They exist so that I (Swaraj) can
> explain every part of the actual work confidently in a TA evaluation.

## Phase 2 — models, training, evaluation

**[Phase2_Learning_Guide.pdf](Phase2_Learning_Guide.pdf)** (46 pages, compiled
from `Phase2_Learning_Guide.tex` with `xelatex`, run twice).

Covers the Transformer built piece by piece with diagrams, the parameter-count
arithmetic done by hand, training mechanics, all the evaluation metrics and why
each one is or is not trustworthy for Devanagari, the attention analysis, every
bug that broke a run and what it taught, a strict clause-by-clause check against
the assignment, and **28 expected viva questions with written-out answers**.

Every number in it was verified against the actual result files in
`report/phase2/` — nothing is invented, and the things that were *not* done are
stated plainly rather than glossed over.

Start with the last two chapters (cheat sheet and question bank) if short on time.

## Phase 1 — data and tokenizers

---

# Phase 1 — Personal Learning Guide

> **This folder is not a course deliverable.** It is not graded, not linked
> from the main `README.md`, and does not affect Phase 1 marks in any way.
> It exists purely so the author (Swaraj) can understand and explain every
> part of the actual Phase 1 work — code, data, decisions — confidently in
> a TA evaluation. Nothing here overrides or replaces
> [`report/phase1/phase1_report.md`](../../report/phase1/phase1_report.md),
> which is the real, graded report.

For the **full detailed version** — theory from scratch, every file
explained, diagrams, a TA question bank, a cheat sheet — see
**[`Phase1_Learning_Guide.pdf`](Phase1_Learning_Guide.pdf)** (compiled from
`Phase1_Learning_Guide.tex`). This README is the short, practical version.

---

## 1. What is this project, in one paragraph

The course (Language Models and Agents) asks each student to build **two
independent, from-scratch Transformer language models**, one for a
higher-resource Indian language ("Model H") and one for a lower-resource
language from a fixed allowed list ("Model L"). Phase 1 — the only phase
covered here — is just the **data and tokenizer** stage: collect text,
clean it, split it into train/validation/test, and train a subword
tokenizer. No model architecture, no training, no PyTorch yet — that's
Phase 2. **Model H is Hindi. Model L is Nepali.**

## 2. Why two different pipelines exist

Hindi and Nepali were built at different times, by different code, for a
concrete reason: Hindi's pipeline (`hindi/data/pipeline/` +
`scripts/clean_lib.py`) was written first and is a **single-pass
cleaner** — one script reads raw text and writes cleaned text in one
straight run. Nepali's pipeline (`nepali/scripts/`, its own `corpuslib`
package) is a **stage-based pipeline**, S00 through S14, where each stage
is its own script, writes its own checkpoint, and can be re-run
independently. This is a real architectural difference, not a naming
inconsistency — the two are documented separately throughout because they
genuinely work differently.

One detail worth knowing so it doesn't confuse you later: every Nepali
stage script's docstring shows its example command as
`python hindi/scripts/sXX.py` — but `hindi/scripts/` does not exist in
this repo. That's leftover template text from when this `corpuslib`
framework was designed to be shared between both languages (with Hindi as
the "source of truth" copy and Nepali as a synced copy — see
`run_pipeline.py`'s `sync` subcommand). In practice, only Nepali ended up
using this framework; Hindi kept its earlier, separate pipeline. Not a
bug — just something to know before you go looking for a `hindi/scripts/`
folder that isn't there.

## 3. The whole idea in one diagram

```
                    RAW TEXT (from the internet / typed / OCR)
                                    |
                    ┌───────────────┴───────────────┐
                    |                                |
              HINDI PIPELINE                  NEPALI PIPELINE
        (single-pass cleaner)           (stage-based, S00-S14)
                    |                                |
        download/ingest -> clean -> split      collect -> normalize -> langid ->
                    |                          holdout -> heuristics -> perplexity
                    |                          -> dedup -> decontaminate ->
                    |                          balance -> finalize splits
                    |                                |
                    └───────────────┬───────────────┘
                                    |
                        train/validation/test splits
                                    |
                        train a SentencePiece tokenizer
                     (Hindi: Unigram, 8,000  |  Nepali: BPE, 10,000)
                                    |
                     Phase 1 deliverable: cleaned corpus +
                     splits + tokenizer + statistics report
```

## 4. Hindi pipeline — file by file

| Step | File | Input | Output | What it actually does |
|---|---|---|---|---|
| Download public sources | `hindi/data/pipeline/download_hindi.py` | Hugging Face datasets (Wikipedia, Sangraha verified/unverified, FineWeb-2) | `data/raw/downloaded/{text/*.txt, metadata.jsonl}` | Streams each HF dataset (doesn't download the whole thing), stops at a character budget per source, tags every doc with a stable `document_id` and full provenance metadata |
| Ingest manual sources | `hindi/data/pipeline/ingest_manual.py` | Your own typed files / PDFs / scanned images / a list of URLs | `data/raw/manual/{text/*.txt, metadata.jsonl}` | Four subcommands (`typed`, `ocr`, `pdf`, `scrape`) covering the four ways this project's manual text was actually collected |
| Ingest Wikipedia dump | `hindi/data/pipeline/ingest_wikipedia_dump.py` | A self-downloaded + WikiExtractor-processed Wikipedia XML dump | `data/raw/manual/` (counts as **manual**, not downloaded) | This is the single biggest manual source (160,591 docs). Counts as manual only because *you* ran WikiExtractor yourself — the same content pulled from Hugging Face's mirror would not count |
| Clean | `hindi/data/pipeline/clean_hindi.py` (using `scripts/clean_lib.py`) | `data/raw/{manual,downloaded}/` | `data/interim/`, `data/cleaned/`, `data/metadata/clean_stats.json` | Normalize → strip Latin-only tokens → drop malformed/too-short docs → language-ID filter (langdetect + a Bhojpuri-lexical-marker check) → exact-hash dedup |
| Split | `hindi/data/pipeline/split_hindi.py` | `data/cleaned/` | `data/splits/{train,validation,test}/` + 3 manifest JSON files | Document-level split (never sentence-level, to avoid leakage), seed 42, ~96/2/2 |
| Train tokenizer | `hindi/tokenizer/train_tokenizer.py` | `data/splits/train/` only | `hindi_unigram_{5000,8000,10000}.model/.vocab` | SentencePiece **Unigram**, `byte_fallback=True` |
| Analyze tokenizer | `hindi/tokenizer/analyze_tokenizer.py` | A trained `.model` + `data/splits/validation/` | Printed metrics (fertility, chars/token, UNK rate, byte-fallback rate, vocab utilization) | Held-out evaluation — never touches train or test |

**Real numbers:** 773,606 raw docs (229,011 manual / 544,595 downloaded) →
704,101 cleaned docs (91.0% doc retention) → 675,937 / 14,082 / 14,082
train/val/test → 505,852,028 tokens at vocab_size=8,000 (101.2% of the
500M target), 28.65% manual by characters.

## 5. Nepali pipeline — file by file

Nepali's pipeline has 15 stages (S00–S14), each its own script under
`nepali/scripts/`, all sharing the `corpuslib` package for common logic
(config loading, JSONL I/O, dedup math, etc.). `run_pipeline.py` runs S02
through S14 in order (S01/collection is separate — it's long-running and
network-bound, driven by `collect.py` instead).

| Stage | File | What it does | Real result |
|---|---|---|---|
| S00 | `s00_doctor.py` | Environment self-test — checks every dependency actually works (this exists because a silent `fasttext`/NumPy incompatibility once deleted 94% of a corpus while logging "SUCCESS") | — |
| S01 | `s01_collect.py` (driven by `collect.py`) | Collects every configured source concurrently, rate-limited per host | 2,655,504 raw docs collected |
| S02 | `s02_calibrate.py` | Trains a throwaway tokenizer on a raw sample to measure a real tokens-per-word ratio (not guessed) | 1.5451 tokens/word |
| S03 | `s03_normalize.py` | NFC normalize, strip invisible/control characters, PII redaction. **Never drops a document** — only S04/S05 decide what's "good" | 2,655,504 → 2,655,504 (100%, no drops possible here) |
| S04 | `s04_langid.py` | fastText language ID, using a *margin* (not raw confidence) against the best competing Devanagari language, since Nepali/Hindi/Maithili raw confidence alone doesn't separate reliably | 99.78% retention |
| S07 | `s07_holdout.py` | **Runs before S05/S06 despite the file number.** Carves out val/test right after S04 (normalize + langid only), so later quality filters never touch — and therefore never bias — the evaluation sets | train-pool 2,251,378 / val 124,042-ish / test similar |
| S05 | `s05_heuristics.py` | Length, symbol ratio, repeated-line ratio, lexical diversity, etc. — reads S07's `train_pool`, not S04 directly | 99.73% retention |
| S06 | `s06_perplexity.py` | Trains a small n-gram model on reference "edited prose" sources, drops the worst-scoring 3% of everything else as the least fluent tail | 97.02% retention |
| S08 | `s08_dedup.py` | Exact-hash dedup, then MinHash+LSH near-duplicate dedup. **Manual-priority rule**: on a manual-vs-downloaded collision, the manual copy always wins | removed 52,084 docs (2,350 exact + 49,734 near) |
| S09 | `s09_decontaminate.py` | Same MinHash machinery as S08, but checks train against val+test — removes any train document that's a near-duplicate of something in eval, so the model can't "cheat" by memorizing eval text | removed 17,111 train docs; final rescan found 0 remaining leaks |
| S10 | `s10_balance_lanes.py` | Trims the **downloaded** lane down to an absolute token target (400M), never touches manual. This project deliberately does NOT chase a manual/downloaded *ratio* — see the guide's theory chapter for why that's a real design decision, not the obvious default | manual share raised from 23.27% → 27.42% |
| S11 | `s11_finalize_splits.py` | Writes the final `train.jsonl`/`val.jsonl`/`test.jsonl`, checks zero document-id overlap | 1,520,000 / 199,428 / 198,837 |
| S12 | `s12_train_tokenizer.py` | Sweeps BPE at [5000, 6000, 8000, 10000], selects by a fertility-elbow rule (not a manually-argued trade-off like Hindi's) | 10,000 selected |
| S13 | `s13_tokenizer_eval.py` | Evaluates the chosen tokenizer on the held-out test split | test fertility 1.4672 |
| S14 | `s14_corpus_stats.py` | Reads every stage's own manifest and produces one final `corpus_stats.json` — nothing is measured twice by two different definitions | 635,204,333 total tokens |

**Real numbers:** 2,655,504 raw docs → 1,918,265 final docs (train + val +
test) → 635,204,333 tokens at vocab_size=10,000 (127.0% of the 500M
target), 26.41% manual by tokens (Nepali's pipeline tracks tokens, not
characters, per lane).

## 6. Complete data flow, beginning to end

```
Hindi:   raw (773,606 docs)
           -> cleaned (704,101 docs, 91.0% kept)
           -> split 675,937 / 14,082 / 14,082
           -> tokenized: 505,852,028 tokens

Nepali:  raw (2,655,504 docs)
           -> S04 langid (2,649,643 docs, 99.78% kept)
           -> S05+S06 quality filters (2,178,418 docs after both)
           -> S08 dedup (2,126,334 docs)
           -> S09 decontaminate (2,109,223 docs)
           -> S10 balance lanes (1,520,000 train docs — val/test held out earlier)
           -> split 1,520,000 / 199,428 / 198,837
           -> tokenized: 635,204,333 tokens
```

## 7. Repository structure (Phase 1 relevant parts only)

```
lma-individual-project/
├── hindi/
│   ├── data/
│   │   ├── raw/{manual,downloaded}/     ← step 1 output
│   │   ├── interim/                     ← post-normalize snapshot
│   │   ├── cleaned/                     ← step 4 output
│   │   ├── splits/{train,validation,test}/  ← step 5 output
│   │   ├── metadata/clean_stats.json    ← the cleaner's own numbers
│   │   └── pipeline/                    ← the 5 scripts in section 4
│   └── tokenizer/                       ← trained .model/.vocab files
├── nepali/
│   ├── scripts/                         ← s00_doctor.py ... s14_corpus_stats.py
│   │   └── corpuslib/                   ← shared code all stages import
│   ├── data/
│   │   ├── raw/{manual,downloaded}/
│   │   ├── interim/                     ← one folder per stage (S03, S04, ...)
│   │   └── final/{train,val,test}.jsonl ← the actual Phase 1 deliverable data
│   ├── configs/                         ← language.yaml, sources.yaml, cleaning.yaml, tokenizer.yaml
│   └── tokenizer/                       ← nepali_bpe.model/.vocab + sweep/
├── scripts/                             ← shared across languages: clean_lib.py (Hindi's),
│                                            validate_phase1.py, validate_phase1_nepali.py,
│                                            check_nepali_hindi_overlap.py
└── report/phase1/                       ← the actual graded deliverables
```

## 8. Important files quick-reference

| File | One-line purpose |
|---|---|
| `scripts/clean_lib.py` | Hindi's shared cleaning toolkit: normalize, dedup hash, language-ok check, file locking |
| `nepali/scripts/corpuslib/config.py` | Loads `language.yaml`/`sources.yaml`, defines every path Nepali's pipeline uses |
| `nepali/scripts/corpuslib/dedup.py` | The actual MinHash/LSH implementation both S08 and S09 call |
| `nepali/scripts/corpuslib/langid.py` | Wraps fastText, implements the margin-based accept/reject decision |
| `nepali/scripts/auto_grow_and_tokenize.py` | The "keep going until both token targets are met" outer loop — trims raw data once, then loops: run S02–S14, check totals, grow whichever lane is short, re-collect, repeat (up to 4 rounds) |
| `scripts/validate_phase1.py` | Automated PASS/FAIL check for Hindi's 4 required properties (raw sources present, ≥20% manual, no split leakage, tokenizer trained) |
| `scripts/validate_phase1_nepali.py` | Same idea, Nepali's own layout |
| `scripts/check_nepali_hindi_overlap.py` | Cross-language check: hashes every Nepali document the same way Hindi's are hashed, checks for any match — found 0 |

## 9. How to run it from scratch

**Hindi:**
```bash
python hindi/data/pipeline/download_hindi.py
python hindi/data/pipeline/ingest_manual.py typed --input-dir <folder>
python hindi/data/pipeline/ingest_wikipedia_dump.py
python hindi/data/pipeline/clean_hindi.py
python hindi/data/pipeline/split_hindi.py
python hindi/tokenizer/train_tokenizer.py --vocab-size 8000
python scripts/validate_phase1.py
```

**Nepali:**
```bash
cd nepali/scripts
python s00_doctor.py                 # always run this first
python auto_grow_and_tokenize.py     # runs everything else, S02-S14, in a loop
python ../../scripts/validate_phase1_nepali.py
python ../../scripts/check_nepali_hindi_overlap.py
```

**Verify each stage worked:** every Nepali stage writes a manifest JSON in
its own `data/interim/<stage>/` folder with `docs_in`/`docs_out`/
`retention` — read that, don't just trust the console output. Hindi's
single-pass cleaner writes one summary, `data/metadata/clean_stats.json`,
after it finishes.

**Troubleshooting:**
- A stage says `BLOCKED: ... has no stage manifest` → the previous stage
  hasn't been run yet (Nepali stages check for this explicitly).
- A `.clean.lock` / `.split.lock` file blocks a rerun → a previous run
  didn't exit cleanly; the lock self-expires after 15 minutes, or delete
  it manually if you're sure nothing else is running.
- Retention outside the expected band → Nepali's S04/S05 stages **abort
  on purpose** rather than silently continue (this is the direct fix for
  the "fastText/NumPy silently deleted 94% of the corpus" incident
  documented in `s00_doctor.py`).

## 10. Deliverables and completion status

| # | Deliverable | Hindi | Nepali |
|---|---|---|---|
| 1 | Collection scripts | ✅ done | ✅ done |
| 2 | Preprocessing pipeline | ✅ done | ✅ done |
| 3 | Dataset statistics report | ✅ done | ✅ done |
| 4 | Train/val/test splits | ✅ exist locally + OneDrive (too large for git) | ✅ same |
| 5 | Tokenizer training code | ✅ done | ✅ done |
| 6 | Vocabulary files | ✅ committed to git | ✅ committed to git |
| 7 | Tokenizer model files | ✅ committed to git | ✅ committed to git |

Both languages pass their respective validators. Cross-language overlap
check (Nepali vs. Hindi): 0 shared documents. Bhojpuri was this project's
*original* Model L, later superseded by Nepali once Nepali's pipeline
cleared its targets — Bhojpuri's own tracked files have since been
removed from the repo (its numbers live on as history in
`report/phase1/phase1_report.md` §6).

## 11. Where to go next

- Full theory + file-by-file + diagrams + TA question bank: **[`Phase1_Learning_Guide.pdf`](Phase1_Learning_Guide.pdf)**
- The actual graded report: [`report/phase1/phase1_report.md`](../../report/phase1/phase1_report.md)
- Real measured numbers, one language at a time: [`report/phase1/dataset_statistics.md`](../../report/phase1/dataset_statistics.md), [`report/phase1/tokenizer_analysis.md`](../../report/phase1/tokenizer_analysis.md)

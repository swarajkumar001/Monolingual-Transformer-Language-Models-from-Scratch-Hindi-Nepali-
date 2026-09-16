<div align="center">

# Language Models and Agents — Individual Project

**Two independent, from-scratch, decoder-only Transformer language models, one per Indian language.**

![Phase 1](https://img.shields.io/badge/Phase%201-Complete-brightgreen)
![Phase 2](https://img.shields.io/badge/Phase%202-Complete-brightgreen)
![Phase 3](https://img.shields.io/badge/Phase%203-Complete-brightgreen)
![Model H](https://img.shields.io/badge/Model%20H-Hindi-1f6f8b)
![Model L](https://img.shields.io/badge/Model%20L-Nepali-7a4fa3)
![Script](https://img.shields.io/badge/Script-Devanagari-orange)
![Params](https://img.shields.io/badge/Params-23.3M%20%2F%2024.3M-blue)
![Built with](https://img.shields.io/badge/Built%20with-plain%20PyTorch-ee4c2c)

</div>

Corpus collection, tokenizer training, a Transformer written from primitive
PyTorch layers, pretraining, evaluation, and reasoning finetuning — done twice,
independently, for a higher-resource language (**Hindi**) and a lower-resource
one (**Nepali**). **All three phases complete.**

The two models share **no data, tokenizer, vocabulary or weights**. `hindi/` and
`nepali/` are separate Python packages; neither imports the other, and every
command names exactly one language.

## Where to look

**Start here: [the Phase 3 final report](report/phase3/phase3_report.md)** — it
consolidates the whole project and answers the four questions the brief asks.

| | Phase 1 — data & tokenizers | Phase 2 — models & evaluation | Phase 3 — reasoning & final report |
|---|---|---|---|
| **Report** | [phase1_report.md](report/phase1/phase1_report.md) ([PDF](report/phase1/phase1_report.pdf)) | [phase2_report.md](report/phase2/phase2_report.md) | **[phase3_report.md](report/phase3/phase3_report.md)** |
| **Deliverables index** | [report/phase1/README.md](report/phase1/README.md) | [report/phase2/README.md](report/phase2/README.md) | [report/phase3/README.md](report/phase3/README.md) |
| **All artifacts** | [report/phase1/](report/phase1/) | [report/phase2/](report/phase2/) | [report/phase3/](report/phase3/) |
| **Large files** | [corpora — OneDrive](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/swaraj_kumar_students_iiit_ac_in/IgCh3u8O2sr4QIJqmrUS1CllAV1C5HL0R9SOWUUVIzTgcdw?e=bgUHaG) | **[checkpoints — Google Drive](https://drive.google.com/drive/folders/15kYy6N9YaCx1wk2JMg8k8udf1szQfneT?usp=sharing)** | **[checkpoints — Google Drive](https://drive.google.com/drive/folders/15kYy6N9YaCx1wk2JMg8k8udf1szQfneT?usp=sharing)** |

---

## Project at a glance

Everything from raw text to a trained, evaluated model — both languages, side by side.

| | Model H — Hindi | Model L — Nepali |
|---|---:|---:|
| **Phase 1 — corpus** | | |
| Documents (final) | 704,101 | 1,918,265 |
| Corpus tokens | 505,852,028 | 635,204,333 |
| % of ~500M target | 101.2% | 127.0% |
| Manual share (token-weighted) | 30.66% | 26.41% |
| **Phase 1 — tokenizer** | | |
| Type / vocabulary | SentencePiece Unigram, **8,000** | SentencePiece BPE, **10,000** |
| Bytes per token | 8.88 | 11.44 |
| Unknown-token rate (test) | 0.000000 | 0.001894 |
| **Phase 2 — model** | | |
| Total parameters | 23,273,472 | 24,297,472 |
| Non-embedding parameters | **18,915,328** | **18,915,328** |
| Layers / heads / d_model / d_ff | 6 / 8 / 512 / 2048 | 6 / 8 / 512 / 2048 |
| Context length | 512 | 512 |
| **Phase 2 — training** | | |
| Steps · tokens | 14,520 · 475,791,360 | 14,440 · 473,169,920 |
| Wall clock (Tesla T4) | 11.0 h @ 12.0k tok/s | 11.0 h @ 12.0k tok/s |
| **Phase 2 — results** | | |
| Validation perplexity | **27.07** | 46.14 |
| Test perplexity | **27.41** | 46.68 |
| Test bits-per-byte | 0.5382 | **0.4846** |
| Mean attention entropy | 3.205 | 3.090 |
| **Phase 3 — reasoning** | | |
| Synthetic finetuning set | 20,000 / 2,000 / 2,000 | 20,000 / 2,000 / 2,000 |
| Unknown-token rate on it | 0.000000 | 0.000000 |
| Exact match, pretrained → finetuned | 0.00% → **35.15%** | 0.00% → 29.45% |
| Forced choice (chance **41.67%**) | 39.80% → 36.35% | 38.80% → 39.30% |
| Answer-token loss | 5.2439 → **0.9408** | 5.5367 → 1.4351 |
| Answers not in the prompt | **0.0–7.2%** | 19.2–28.8% |

<div align="center">
<img src="report/phase2/figures/loss_curves_both.png" width="88%" alt="Training and validation loss for both models against tokens consumed">
</div>

<div align="center">
<img src="report/phase2/figures/attention_summary_both.png" width="88%" alt="Per-head attention entropy and mean attention distance by layer, both models">
</div>

### The headline finding

**Perplexity and bits-per-byte disagree — and the disagreement is the result.**

Perplexity favours Model H by 41%. Bits-per-byte puts Model L *marginally
ahead*. Both are correct; they measure different things. Nepali's BPE-10000
packs **11.44 bytes per token** against Hindi's Unigram-8000 at **8.88**, so its
tokens are individually harder to predict but each carries proportionally more
text. Normalising by UTF-8 bytes removes the tokenizer from the denominator.

Reporting perplexity alone would have supported the conclusion that the
lower-resource model is substantially worse. It isn't — most of the apparent gap
is a **tokenizer artifact**. Full analysis in
[the report](report/phase2/phase2_report.md) §4.

A second result worth noting: both models — different languages, tokenizers and
seeds — **independently converged on the same attention organisation**, including
a previous-token head at layer 3 (entropy 0.06 at distance 1.3 for Model H, 0.14
at 1.1 for Model L), with mean entropy about 2 nats below the uniform-attention
baseline of 5.256.

### The Phase 3 finding

**Finetuning taught the output format, not the reasoning — and only a
chance-calibrated metric shows the difference.**

Exact match rose from 0.00% to 35.15% (Model H) and 29.45% (Model L), which
looks like success. It is not. On a forced choice between the entities the
question actually names, both models sit **at or below the 41.67% chance rate**
— 36.35% and 39.30%. Per task the agreement with chance is near-exact: Model H
scores 50.40% and 47.00% on the two two-candidate families (chance 50%) and
33.20% on three-candidate `numeric3` (chance 33.3%).

The two-hop transitive task is the one place the models depart from chance, and
they depart *downward*: 12.40% and 13.40% against 33.3%. The reason is
measurable rather than mysterious — on *"A > B. B > C. Who is largest?"*, Model
H answers with the **middle entity 72.6% of the time and the first entity 0.0%
of the time**, and the middle entity is never correct. It is selecting the name
mentioned twice instead of following the chain.

Reporting exact match alone would have supported the conclusion that finetuning
taught these models to reason. Full analysis in
[the report](report/phase3/phase3_report.md) §3–§4.

---

## Phase 3 — Reasoning Finetuning, Attention & Final Report ✅

Each model finetuned on a synthetic comparative-reasoning corpus **written in
its own language and generated programmatically**, so every ground-truth label
is known by construction. No existing benchmark was downloaded; the two corpora
share no sentence, entity name or template.

Four task families — one-step comparison, two-hop transitive chains, and numeric
comparison over two or three entities — with **both** leakage defences the brief
suggests applied together: entity names and sentence templates are each split
three ways with no overlap, asserted in code rather than assumed.

**Two defects caught during construction**, both of which would have invalidated
the results:

- A **45% gender-agreement leak**. Hindi adjectives inflect, so *"सबसे छोटी कौन
  है?"* asked about one woman and two men has one grammatically possible answer.
  181 of 400 examples were solvable without comparing anything. Fixed by drawing
  all entities in an example from one gender.
- **Model L's vocabulary contains no Devanagari digit at all** — not one of ०–९
  — and its BPE has no byte fallback, so every numeral encoded to `<unk>`, at a
  4.8% corpus rate. Nepali numerals are now written with Arabic digits, which
  the vocabulary covers completely and which modern Nepali prose uses anyway.

Both corpora now tokenize at a **0.000000** unknown rate.

### Deliverables

| # | Deliverable | Where |
|---|---|---|
| 1 | Data generation · finetune scripts · configs | [`hindi/reasoning/`](hindi/reasoning/) · [`nepali/reasoning/`](nepali/reasoning/) · [`finetune.yaml`](hindi/configs/finetune.yaml) ×2 |
| 2 | Finetuned checkpoints (H and L) | [Drive](https://drive.google.com/drive/folders/15kYy6N9YaCx1wk2JMg8k8udf1szQfneT?usp=sharing) — table below |
| 3 | Finetuning logs · metrics · qualitative examples | [`*_finetune_log.jsonl`](report/phase3/) · [`*_reasoning_*.json`](report/phase3/) · [report §3–4](report/phase3/phase3_report.md) |
| 4 | Pretrain vs. finetune attention comparison | [report §5](report/phase3/phase3_report.md) · [`figures/`](report/phase3/figures/) |
| 5 | Final comparison tables and error analysis | [report §3, §4, §6](report/phase3/phase3_report.md) |
| 6 | Complete final report and reproduction steps | [phase3_report.md](report/phase3/phase3_report.md) |

### Results

2,000 held-out test questions per model. Held-out names **and** held-out
templates, so a test question is about unseen people phrased in an unseen way.

| Model | Metric | Pretrained | Finetuned | Change |
|---|---|---:|---:|---:|
| **H** | Exact match | 0.00% | **35.15%** | +35.15 |
| | Forced choice | 39.80% | 36.35% | −3.45 |
| | Answer-token loss | 5.2439 | **0.9408** | −4.3031 |
| **L** | Exact match | 0.00% | **29.45%** | +29.45 |
| | Forced choice | 38.80% | 39.30% | +0.50 |
| | Answer-token loss | 5.5367 | **1.4351** | −4.1016 |
| | *chance rate* | *41.67%* | *41.67%* | — |

Exact match after greedy decoding is the honest end-to-end number, but it floors
at zero on a model that has never seen the prompt format — so a **forced choice
among the entities the question names** is reported alongside it, with a chance
rate computed per example. That is the metric that shows what was actually
learned. See [the Phase 3 finding](#the-phase-3-finding).

Both models overfit fast: validation loss bottoms at **step 60** and rises
monotonically after, while training loss falls to 0.44. With 40 training names
and 14 templates, the answer prior is memorisable long before copying from the
prompt becomes worthwhile. Saving on best validation loss is what kept the
evaluated checkpoints usable.

<div align="center">
<img src="report/phase3/figures/finetune_loss_curves_both.png" width="88%" alt="Finetuning training and validation loss for both models against optimiser step">
</div>

### Attention, pretrained vs. finetuned

Both checkpoints run on the **same** held-out reasoning prompts, all of one token
length — otherwise the text or the sequence length could masquerade as a
finetuning effect.

| Model | Checkpoint | Mean entropy | Mean distance | Uniform baseline |
|---|---|---:|---:|---:|
| H | pretrained | 1.9102 | 10.303 | 2.9545 (T=46) |
| H | finetuned | 1.8691 | 10.360 | |
| L | pretrained | 1.8327 | 9.415 | 2.8061 (T=39) |
| L | finetuned | 1.7833 | 10.158 | |

Finetuning made attention **sharper and longer-range at once** in both models,
most clearly in Model L, whose mean distance rises 0.74 positions on a 39-token
prompt. The heads that moved most are in the middle layers, where Phase 2 found
the previous-token heads. Layer 0 and layer 5 heatmaps for both checkpoints and
both models are in [`report/phase3/figures/`](report/phase3/figures/), with
Devanagari token labels.

### Reproduction

```bash
python hindi/reasoning/generate.py  --train 20000 --val 2000 --test 2000
python nepali/reasoning/generate.py --train 20000 --val 2000 --test 2000

# Score the pretrained baseline FIRST — it cannot be recovered afterwards
python -m hindi.reasoning.evaluate_reasoning     --checkpoint checkpoints/hindi/ckpt_best.pt --tag pretrained

python -m hindi.reasoning.finetune --config hindi/configs/finetune.yaml

python -m hindi.reasoning.evaluate_reasoning     --checkpoint checkpoints/hindi_reasoning/ckpt_best.pt --tag finetuned
```

Full steps, including the Kaggle path and the figure regeneration, in
[the report §9](report/phase3/phase3_report.md#9-reproduction).

---

## Phase 2 — Model Implementation, Pretraining & Evaluation ✅

Decoder-only Transformers built from `nn.Linear`, `nn.Embedding`, `nn.Dropout`
and raw tensor operations. No `nn.Transformer*`, no `nn.MultiheadAttention`, no
`F.scaled_dot_product_attention`, no HuggingFace model class — enforced by a
test that walks the module tree and fails if any appears.

**Causal masking verified empirically**, as the spec requires: perturbing every
token after position *t* changes the logits at positions ≤ *t* by
**exactly 0.000e+00**. Not merely within tolerance — `softmax(-inf)` is exactly
zero and `0 × v = 0` exactly. Both test suites pass **9/9** independently.

### Deliverables

| # | Deliverable | Where |
|---|---|---|
| 1 | Transformer (MHA, positional embeddings, causal mask) | [`hindi/model/transformer.py`](hindi/model/transformer.py) · [`nepali/model/transformer.py`](nepali/model/transformer.py) |
| 2 | Model configuration files | [`hindi/configs/model.yaml`](hindi/configs/model.yaml) · [`nepali/configs/model.yaml`](nepali/configs/model.yaml) |
| 3 | Parameter counts · training scripts · checkpoints | table above · [`hindi/train/train.py`](hindi/train/train.py), [`nepali/train/train.py`](nepali/train/train.py) · [Drive](https://drive.google.com/drive/folders/15kYy6N9YaCx1wk2JMg8k8udf1szQfneT?usp=sharing) |
| 4 | Training logs and loss curves | [`*_training_log.jsonl`](report/phase2/) · [`loss_curves_both.png`](report/phase2/figures/loss_curves_both.png) |
| 5 | PPL / BPB tables; BLEU, chrF, ROUGE-L | [report §4–5](report/phase2/phase2_report.md) · [`*_eval.json`](report/phase2/) |
| 6 | Generated samples; diversity / repetition | [report §5.2](report/phase2/phase2_report.md) · `*_eval.json` → `generation.samples` |
| 7 | Attention heatmaps; entropy / distance | [`figures/`](report/phase2/figures/) — layers 0 and 5, 4 heads each · [`*_attention.json`](report/phase2/) |
| 8 | Resource-level comparison write-up | [report §7](report/phase2/phase2_report.md) |

### Evaluation summary

Scored over 1,999,872 tokens per split for each model.

| Model | Decoding | BLEU-4 | chrF++ | ROUGE-L | Repetition | Distinct-1 |
|---|---|---:|---:|---:|---:|---:|
| H | greedy | 1.96 | 12.49 | 0.093 | 0.598 | 0.172 |
| H | T=1.0 | 1.01 | **16.29** | 0.100 | 0.002 | 0.520 |
| L | greedy | 2.37 | 14.48 | 0.081 | 0.577 | 0.179 |
| L | T=1.0 | 1.69 | **18.05** | 0.079 | 0.001 | 0.634 |

Greedy decoding **degenerates** in both models — ~58–60% of generated 4-grams
are repeats, and the samples visibly loop. Temperature 1.0 removes the
repetition *and* peaks chrF++, so diversity costs no fidelity up to that point.
All four decoding settings (greedy, T=0.5/1.0/1.5) are in the report and the
eval JSONs.

### Checkpoints (Google Drive)

**[LMA CHECKPOINT — shared folder](https://drive.google.com/drive/folders/15kYy6N9YaCx1wk2JMg8k8udf1szQfneT?usp=sharing)**

| Phase | Model | Bundle | Size | Contents |
|---|---|---|---:|---|
| 2 — pretrained | H | `model_H_hindi_checkpoints.zip` | 1.28 GB | `ckpt_best.pt` (step 14,500), `ckpt_final.pt`, `latest.pt`, steps 14,000 / 14,500 |
| 2 — pretrained | L | `model_L_nepali_checkpoints.zip` | 1.33 GB | `ckpt_best.pt` (step 14,000), `ckpt_final.pt`, `latest.pt`, steps 13,500 / 14,000 |
| 3 — finetuned | H | `model_H_hindi_reasoning_finetuned.zip` | 0.77 GB | `ckpt_best.pt` (step 60), `ckpt_final.pt`, `latest.pt`, `finetune_config.yaml`, `finetune_log.jsonl` |
| 3 — finetuned | L | `model_L_nepali_reasoning_finetuned.zip` | 0.80 GB | `ckpt_best.pt` (step 60), `ckpt_final.pt`, `latest.pt`, `finetune_config.yaml`, `finetune_log.jsonl` |

Every checkpoint — pretrained and finetuned alike — carries model weights,
optimizer state, scheduler state, the current step, best validation loss and the
full run configuration, so training resumes **exactly**, not approximately. The
pretraining checkpoints additionally carry tokens consumed and both RNG states;
resume was verified across three consecutive sessions before the real runs
began. The finetuned checkpoints keep the pretraining payload keys, which is
what lets the Phase 2 evaluator and attention toolkit load them unchanged.

### Reproduction

```
python hindi/data/pack_corpus.py            # Phase 1 splits -> uint16 token streams
python nepali/data/pack_corpus.py

python hindi/tests/test_model.py            # 9/9, incl. the causal-mask proof
python nepali/tests/test_model.py

python -m hindi.train.train  --config hindi/configs/model.yaml
python -m nepali.train.train --config nepali/configs/model.yaml

python -m hindi.eval.evaluate  --checkpoint checkpoints/hindi/ckpt_best.pt  --split test
python -m nepali.eval.evaluate --checkpoint checkpoints/nepali/ckpt_best.pt --split test

python scripts/make_phase2_figures.py \
    --hindi-checkpoint checkpoints/hindi/ckpt_best.pt \
    --nepali-checkpoint checkpoints/nepali/ckpt_best.pt
```

Pretraining ran on Kaggle (GPU T4) via
[`notebooks/kaggle_pretrain.ipynb`](notebooks/kaggle_pretrain.ipynb), once per
language, with [`notebooks/kaggle_evaluate.ipynb`](notebooks/kaggle_evaluate.ipynb)
scoring a finished checkpoint.

---

## Phase 1 — Dataset Collection & Tokenizer Construction ✅

`scripts/validate_phase1.py` reports **ALL CHECKS PASSED** for Hindi.
`scripts/validate_phase1_nepali.py` (Nepali's own parallel validator, since its
pipeline and data layout differ) reports **ALL CHECKS PASSED** for Nepali except
the cross-language check, which it deliberately flags `SKIPPED` rather than
running inline — at ~9.2GB it is too large for a validator side effect. That
check was run separately (`scripts/check_nepali_hindi_overlap.py`): **0 shared
documents**.

Hindi and Nepali ran on two structurally different pipelines — Hindi a
single-pass cleaner, Nepali a stage-based S00–S14 pipeline with MinHash/LSH
dedup and absolute per-lane token targets — so the report describes each
separately rather than forcing one shared table.

### Deliverables

| # | Deliverable | Where |
|---|---|---|
| 1 | Dataset collection scripts | `hindi/data/pipeline/{download_hindi,ingest_manual,ingest_wikipedia_dump}.py`; `nepali/scripts/{s01_collect,collect,auto_grow_and_tokenize}.py`, `corpuslib/` |
| 2 | Dataset preprocessing pipelines | `hindi/data/pipeline/clean_hindi.py`, `scripts/clean_lib.py`; `nepali/scripts/s03…s11` |
| 3 | Per-language dataset statistics | [`phase1_report.md`](report/phase1/phase1_report.md), `dataset_statistics.md`, `tokenizer_analysis.md` |
| 4 | Train / validation / test splits | `hindi/data/splits/` + manifests; `nepali/data/final/{train,val,test}.jsonl` + `split_stats.json` |
| 5 | Tokenizer training code | `hindi/tokenizer/train_tokenizer.py`; `nepali/scripts/{s12_train_tokenizer,s13_tokenizer_eval}.py` |
| 6 | Vocabulary files | `hindi/tokenizer/hindi_unigram_8000.vocab`; `nepali/tokenizer/nepali_bpe.vocab` (+ `sweep/`) |
| 7 | Tokenizer model files | `hindi/tokenizer/hindi_unigram_8000.model`; `nepali/tokenizer/nepali_bpe.model` |

### Corpus token breakdown

| | Hindi (Model H) | Nepali (Model L) |
|---|---:|---:|
| Train tokens | 485,280,741 | 519,906,359 |
| Validation tokens | 10,484,640 | 57,839,396 |
| Test tokens | 10,086,647 | 57,458,578 |
| **Total** | **505,852,028** | **635,204,333** |
| Manual tokens | 155,104,651 | 167,787,251 |
| Clears the 20% manual floor? | Yes (30.66%) | Yes (26.41%) |

Every number is exact and unsampled — every train/val/test document was actually
tokenized and counted, not estimated.

<details>
<summary><b>Sources — click to expand</b></summary>

**Hindi, manual:** `hi_wikipedia_dump` (self-extracted Wikipedia XML dump via
WikiExtractor — largest single manual source), `gadya_kosh` (prose wiki),
`patrika_hindi` (newspaper archive), `kavita_kosh` (poetry wiki), 67 misc.
`archive.org` items (debate transcripts, books).
**Hindi, downloaded:** `HuggingFaceFW/fineweb-2:hin_Deva`,
`ai4bharat/sangraha:unverified/hin`,
`ai4bharat/sangraha:verified/hin:{web,pdf,speech}`. Excluded: the HF-mirrored
`wikimedia/wikipedia:20231101.hi` (redundant with the self-extracted dump).

**Nepali** gates on two *absolute* token targets rather than a ratio: manual
≥ 100M, downloaded ≥ 400M — both cleared (manual 167.8M, downloaded 467.4M).
**Manual:** `news_onlinekhabar` (web scrape, 91.6% of manual tokens),
`wikipedia_api` (self-parsed XML dump), `news_bbc_nepali`, `wikibooks_api`,
`wiktionary_api`, `news_ekantipur`, `news_ratopati`, `news_nagarik`,
`ocr_archive_books`. **Downloaded:** `ai4bharat/sangraha` (verified +
unverified), `ai4bharat/IndicCorpV2`.

> **Provenance note:** Nepali's pipeline was run on a separate machine — the
> ~30+ hour S08 MinHash dedup pass was impractical on the primary machine within
> the deadline window. Only output artifacts were transferred, not live
> per-stage logs, so `log.md`'s S08(production)–S14 entries are reconstructed
> from those artifacts and flagged inline as such.

</details>

<details>
<summary><b>Phase 1 reproduction steps — click to expand</b></summary>

**1. Environment**

```
python -m venv .venv
.venv/Scripts/activate      # or: source .venv/bin/activate
pip install -r requirements.txt
```

**2. Hindi pipeline**

```
python hindi/data/pipeline/download_hindi.py        # public/HF corpora -> raw/downloaded/
python hindi/data/pipeline/ingest_manual.py         # scraped/self-extracted -> raw/manual/
python hindi/data/pipeline/ingest_wikipedia_dump.py # self-extracted Wikipedia XML dump
python hindi/data/pipeline/clean_hindi.py           # raw/ -> interim/ -> cleaned/
python hindi/data/pipeline/split_hindi.py           # cleaned/ -> splits/
python hindi/tokenizer/train_tokenizer.py --vocab-size 8000
```

`clean_hindi.py` rebuilds `interim/`/`cleaned/` from scratch by default; pass
`--resume` to continue an interrupted run. `clean_hindi.py`/`split_hindi.py`
take an OS-level single-instance lock (`.clean.lock` / `.split.lock`) — delete a
stale lock older than 15 minutes if a run was killed uncleanly.

**3. Nepali pipeline** (stage-based, config-driven)

```
cd nepali/scripts
python s00_doctor.py                # environment/config sanity check
python auto_grow_and_tokenize.py    # full S02-S14, growing whichever lane falls
                                    # short of its absolute token target
```

Or run stages individually (`s01_collect.py` … `s14_corpus_stats.py`) via
`python run_pipeline.py <stage>`. S08 (MinHash/LSH dedup over ~2.2M documents)
is the slow stage — budget several hours.

**4. Validate**

```
python scripts/validate_phase1.py            # Hindi
python scripts/validate_phase1_nepali.py     # Nepali
python scripts/check_nepali_hindi_overlap.py # cross-language overlap
python report/phase1/make_plots.py           # regenerate all figures
```

</details>

---

## Repository layout

<details>
<summary>Click to expand</summary>

```
<repo>/
  README.md                    # this file
  requirements.txt
  hindi/                       # Model H — self-contained package
    data/
      raw/{manual,downloaded}/ # source text + metadata.jsonl (provenance)
      interim/ cleaned/        # cleaning stages
      splits/                  # train/validation/test (document-level)
      packed/                  # uint16 token streams (gitignored; .json
                                # sidecars with exact token counts tracked)
      pipeline/                # Phase 1 download/ingest/clean/split
      dataset.py               # Phase 2 batch sampling over the memmaps
      pack_corpus.py           # Phase 2 splits -> uint16 streams
    tokenizer/                 # SentencePiece model + vocab
    model/                     # config.py, transformer.py
    train/                     # train.py
    eval/                      # evaluate.py, attention.py, viz.py
    reasoning/                 # Phase 3: generate.py (synthetic corpus),
                                # dataset.py, finetune.py,
                                # evaluate_reasoning.py, attention_compare.py
      data/                    # train/val/test.jsonl + dataset_stats.json
    tests/                     # test_model.py (incl. causal-mask proof)
    configs/                   # model.yaml, finetune.yaml
  nepali/                      # Model L — same shape, independent copy.
                                # Phase 1 pipeline differs (stage-based
                                # S00-S14 under scripts/ + corpuslib/).
  notebooks/                   # kaggle_pretrain.ipynb, kaggle_evaluate.ipynb,
                                # kaggle_finetune.ipynb
  scripts/                     # Phase 1: clean_lib.py, crawlers, validate_*.py
                                # Phase 2: kaggle_push.py, kaggle_run.py,
                                # collect_phase2_results.py, make_phase2_figures.py
                                # Phase 3: collect_phase3_results.py,
                                # make_phase3_figures.py,
                                # make_finetune_notebook.py
  report/
    phase1/                    # phase1_report.{md,tex,pdf}, figures/, indices
    phase2/                    # phase2_report.md, {hindi,nepali}_eval.json,
                                # *_attention.json, *_training_log.jsonl,
                                # configs, figures/
    phase3/                    # phase3_report.md, *_reasoning_{pre,fine}*.json,
                                # *_attention_comparison.json,
                                # *_finetune_log.jsonl, configs, figures/
  logs/                        # pipeline logs (gitignored)
  # bhojpuri/ removed — earlier Model L attempt, superseded by Nepali
```

</details>

## Final submission checklist

The ten items the specification requires in the final submission, and where each
one is in this repository.

| # | Required | Where | ✅ |
|---|---|---|:-:|
| 1 | Dataset and preprocessing code (both) | [`hindi/data/pipeline/`](hindi/data/pipeline/) · [`nepali/scripts/`](nepali/scripts/) · [`scripts/clean_lib.py`](scripts/clean_lib.py) | ✅ |
| 2 | Separate tokenizers and vocabularies | [`hindi/tokenizer/`](hindi/tokenizer/) — Unigram, 8,000 · [`nepali/tokenizer/`](nepali/tokenizer/) — BPE, 10,000 | ✅ |
| 3 | Transformer implementation | [`hindi/model/transformer.py`](hindi/model/transformer.py) · [`nepali/model/transformer.py`](nepali/model/transformer.py) | ✅ |
| 4 | Drive links — **pretrained** checkpoints (H and L) | [Checkpoints table](#checkpoints-google-drive), rows "2 — pretrained" | ✅ |
| 5 | Drive links — **finetuned** checkpoints (H and L) | [Checkpoints table](#checkpoints-google-drive), rows "3 — finetuned" | ✅ |
| 6 | Training and finetuning logs | [`report/phase2/*_training_log.jsonl`](report/phase2/) · [`report/phase3/*_finetune_log.jsonl`](report/phase3/) | ✅ |
| 7 | Evaluation scripts and results | [`hindi/eval/`](hindi/eval/) · [`nepali/eval/`](nepali/eval/) · [`*/reasoning/evaluate_reasoning.py`](hindi/reasoning/evaluate_reasoning.py) · [`report/phase2/*_eval.json`](report/phase2/) · [`report/phase3/*_reasoning_*.json`](report/phase3/) | ✅ |
| 8 | Attention visualizations | [`report/phase2/figures/`](report/phase2/figures/) — pretrained · [`report/phase3/figures/`](report/phase3/figures/) — pretrained vs. finetuned, layers 0 and 5 | ✅ |
| 9 | Final report | [`report/phase3/phase3_report.md`](report/phase3/phase3_report.md) — consolidates all three phases | ✅ |
| 10 | README with reproduction steps and all Drive links | this file — [Phase 3](#reproduction) · [Phase 2](#reproduction-1) · [Phase 1](#phase-1--dataset-collection--tokenizer-construction-) · [Drive](#checkpoints-google-drive) | ✅ |

---

## Hard constraints followed

No pretrained models, no pretrained tokenizers, no HuggingFace Transformer model
classes. Positional embeddings, multi-head attention and causal masking are
student-implemented from primitive PyTorch layers — verified by a test that
walks the module tree and fails on any pre-built attention block.

The two corpora, tokenizers, vocabularies and model weights are fully
independent: `hindi/` and `nepali/` are separate packages, neither importing the
other, and no multilingual training was performed.

Phase 3 keeps that separation. The reasoning finetuning data was generated here
rather than downloaded, one corpus per language, sharing no sentence, entity
name or template — verified programmatically. Each model was finetuned from its
own pretrained checkpoint with its own Phase 1 tokenizer held fixed; the
finetuner asserts the tokenizer's vocabulary size matches the checkpoint rather
than trusting the config.

---

## Project status

All three phases are complete and pushed to their branches: `phase-1`, `phase-2`
and `phase-3`. The `phase-3` branch is the full project snapshot and is the
branch submitted for grading.

**The optional bonus ablation was also run**, on a separate branch built on top
of this one: **`bonus-no-pos`**. Model H was retrained from scratch with
`pos_encoding: none` and put through the full Phase 2 evaluation suite. Removing
positional embeddings cost 5.8% test perplexity (27.41 → 29.01) — much less than
permutation invariance alone would predict, because causal masking leaks
position — while mean attention entropy rose from 3.2047 to 3.6005 against a
uniform baseline of 5.2565, and the previous-token head re-formed one layer
later. Setup, results, figures and limitations are in `report/bonus/README.md`
on that branch. No ablation results are claimed on this branch.

---

<details>
<summary><b>Bhojpuri (superseded) — click to expand</b></summary>

Bhojpuri was this project's original Model L: fully built, cleaned, split,
tokenized, and passing every `validate_phase1.py` check, with a documented,
spec-permitted 16.8%-of-target shortfall (84,217,032 tokens). It was superseded
by **Nepali** once Nepali's pipeline cleared both its absolute token targets —
see [`phase1_report.md` §6](report/phase1/phase1_report.md) for the full history
and Bhojpuri's numbers.

Its tracked pipeline code and tokenizer files have since been removed;
`bhojpuri/data/` was never tracked in git regardless. Its measured numbers
remain in `phase1_report.md` as the historical record.

</details>

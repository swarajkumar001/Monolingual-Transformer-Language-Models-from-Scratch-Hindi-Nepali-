# All reports and every reported number

Part of the LMA individual project viva pack. Line numbers below match the files in the repository exactly, so any line cited here can be opened directly in the IDE.

Files in this pack:

- `README.md`
- `report/phase1/phase1_report.md`
- `report/phase2/phase2_report.md`
- `report/phase3/phase3_report.md`
- `report/phase3/README.md`
- `hindi/reasoning/data/dataset_stats.json`
- `nepali/reasoning/data/dataset_stats.json`


## README.md

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
and `phase-3`. The `phase-3` branch is the full project snapshot.

Optional work not undertaken: the no-positional-embeddings bonus ablation. The
model already supports it — set `pos_encoding: none` in a model config — but it
was not run, and no results for it are claimed anywhere.

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



## report/phase1/phase1_report.md

# Phase 1 Report — Data Collection & Tokenizer Construction

**Model H: Hindi. Model L: Nepali.** This report covers Phase 1 only:
dataset collection, cleaning, splitting, and tokenizer construction for both
models. Every number below is read from a file on disk — Hindi from
`clean_stats.json`, `hindi_pipeline_manifest.json`, `validate_final.log`, the
trained tokenizer `.model` files re-measured directly; Nepali from
`nepali/data/final/corpus_stats.json`, `split_stats.json`, its own tokenizer
`.model` files re-measured the same way — or
`report/phase1/measured_tokenizer_metrics.json` (written by
`report/phase1/make_plots.py`, which also produced every figure below).
Where a number was not measured, it is written as **NOT YET MEASURED**,
never estimated. Figures are regenerated by running:

```
.venv/Scripts/python.exe report/phase1/make_plots.py
```

**Note on pipelines:** Hindi and Nepali are built by two genuinely different
pipelines, not one shared script parameterized by language — Hindi's is a
single-pass cleaner (`scripts/clean_lib.py`, `clean_hindi.py`); Nepali's is a
stage-based S00–S14 pipeline (`nepali/scripts/`, its own `corpuslib`). Both
were run independently against their own raw data with no shared code path
that could leak one language's decisions into the other's, but it does mean
some sections below describe two distinct mechanisms side by side rather
than one shared table.

---

## 1. Overview

The assignment requires two independent, from-scratch decoder-only
Transformer language models in two different Indian languages: Model H from
a higher-resource language, Model L from a fixed list of lower-resource
languages. Both must be built with fully separate data, tokenizer,
vocabulary, and (from Phase 2) weights — no sharing, no multilingual
training.

**Model H is Hindi.** Hindi is one of the highest-resource Indian languages
for NLP: Wikipedia, curated corpora like AI4Bharat's Sangraha, and
web-crawled corpora like Fineweb-2 all have large Hindi partitions. This
makes it realistic to reach the assignment's ~500M-token planning target
almost entirely from public sources, while still requiring genuine manual
collection to clear the 20% floor.

**Model L is Nepali**, chosen from the assignment's allowed low-resource
list (Assamese, Bhojpuri, Bodo, Dogri, Konkani, Maithili, Manipuri, Mizo,
Nepali, Sindhi). Nepali has considerably more public data available than
several other allowed languages (AI4Bharat's Sangraha and IndicCorpV2 both
carry sizeable Nepali partitions), which let this corpus not just clear the
20% manual floor but exceed the 500M-token planning target outright
(127.0%) — while still requiring genuine, sustained manual collection
(mainly news-site scraping) to hit the pipeline's own 100M-token absolute
manual-lane floor.

Both languages are Devanagari-script. Hindi's cleaning/tokenization code is
`scripts/clean_lib.py`; Nepali's is its own independent `corpuslib` package
under `nepali/scripts/` — not shared with Hindi's — keeping every byte of
data, every trained model file, and every decision fully separate per
language.

---

## 2. Data collection

### What counts as "manual" under the course ruling

A source counts as **manual** only if it required work beyond pointing a
downloader at an existing public dataset: a self-run scrape, a personally
extracted Wikipedia XML dump, OCR from a physical or scanned source, or
similarly hands-on collection. A dataset that is just *labeled* as
Wikipedia-derived on Hugging Face does not count — even though the
underlying text originates from Wikipedia, using the HF-hosted mirror is a
download, not manual collection.

Concretely: `hi_wikipedia_dump` is the **self-extracted** Hindi Wikipedia XML
dump, run through WikiExtractor by this project — it counts as manual.
`wikimedia/wikipedia:20231101.hi`, the HF-mirrored parquet of the same
underlying Wikipedia content, does **not** count as manual, and using both
would double-count Wikipedia in two different lanes with two different
`source_type` labels for the same underlying text. It was moved to
`hindi/data/raw/_quarantined_hf_hiwiki_mirror/` and excluded from cleaning
entirely.

### Hindi (Model H) raw sources

| | Docs | Characters |
|---|---:|---:|
| Manual | 229,011 | 540,367,035 |
| Downloaded | 544,595 | 1,299,990,112 |
| **Total raw** | **773,606** | **1,840,357,147** |

**Manual sources:**

| Source | Docs | Characters | Collection method |
|---|---:|---:|---|
| hi_wikipedia_dump | 160,591 | 234,096,553 | self-extracted Wikipedia XML dump (WikiExtractor) |
| gadya_kosh | 21,481 | 197,490,954 | scrape + clean (prose wiki) |
| patrika_hindi | 16,251 | 73,759,826 | scrape + clean (patrika.com newspaper archive) |
| kavita_kosh | 30,621 | 23,610,585 | scrape + clean (poetry wiki) |
| archive.org (misc, 67 items) | 67 | 11,409,117 | OCR/text extraction — Rajya Sabha/Lok Sabha debate transcripts, religious texts, misc books |

**Downloaded sources:**

| Source | Docs | Characters |
|---|---:|---:|
| HuggingFaceFW/fineweb-2:hin_Deva | 217,460 | 500,000,289 |
| ai4bharat/sangraha:unverified/hin | 191,987 | 500,000,700 |
| ai4bharat/sangraha:verified/hin:web | 118,785 | 230,941,855 |
| ai4bharat/sangraha:verified/hin:pdf | 16,120 | 67,642,616 |
| ai4bharat/sangraha:verified/hin:speech | 243 | 1,404,652 |

**Excluded:** `wikimedia/wikipedia:20231101.hi` (163,091 docs / 253.3M
chars) — the HF-hosted Hindi Wikipedia mirror, quarantined to avoid
double-counting against the self-extracted dump above, and because it would
not count as manual regardless.

*Note on the raw total:* `hindi/data/metadata/clean_stats.json`'s own
`chars_in` (what `clean_hindi.py` measured as its input) reads
1,840,356,619 — 528 characters (0.00003%) less than the 1,840,357,147 sum of
the per-source table above. Both figures are real, independently-collected
measurements (one at ingestion/tagging time, one at cleaning time); the gap
is negligible and not investigated further given its scale.

### Nepali (Model L) raw sources

Nepali's pipeline tracks docs/words/tokens per source, not characters (its
cleaner never kept a per-lane character tally past normalization). Two
lanes, each gated on its own **absolute** token target rather than a ratio:
manual ≥ 100,000,000 tokens, downloaded ≥ 400,000,000 tokens
(`nepali/configs/language.yaml`).

| | Docs | Words | Tokens |
|---|---:|---:|---:|
| Manual | 337,783 | 116,207,907 | 167,787,251 |
| Downloaded | 1,580,482 | 319,077,639 | 467,417,082 |
| **Total (final)** | **1,918,265** | **435,285,546** | **635,204,333** |

**Manual sources:**

| Source | Docs | Words | Tokens | Collection method |
|---|---:|---:|---:|---|
| news_onlinekhabar | 308,139 | 107,614,170 | 153,717,439 | web scrape |
| wikipedia_api | 16,445 | 4,622,944 | 8,052,774 | self-parsed XML dump |
| news_bbc_nepali | 12,377 | 3,555,706 | 5,277,060 | web scrape |
| wikibooks_api | 74 | 230,885 | 412,110 | self-parsed XML dump |
| wiktionary_api | 606 | 96,765 | 200,300 | self-parsed XML dump |
| news_ekantipur | 119 | 80,432 | 115,316 | web scrape |
| news_ratopati | 20 | 6,055 | 10,608 | web scrape |
| ocr_archive_books | 1 | 447 | 866 | OCR |
| news_nagarik | 2 | 503 | 778 | web scrape |

`news_onlinekhabar` alone supplies 91.6% of manual tokens — a real,
measured constraint: it is the manual source with the largest scrapable
archive, and `nepali/scripts/auto_grow_and_tokenize.py`'s growth loop
raised its collection target specifically because the other manual sources
(Wikipedia/Wiktionary/Wikibooks, smaller news sites, one OCR document) are
genuinely smaller pools for Nepali than their Hindi equivalents.

**Downloaded sources:**

| Source | Docs | Words | Tokens |
|---|---:|---:|---:|
| ai4bharat/sangraha:verified | 579,743 | 173,902,717 | 258,462,424 |
| ai4bharat/sangraha:unverified | 341,901 | 99,767,699 | 141,886,279 |
| ai4bharat/IndicCorpV2 | 658,838 | 45,407,223 | 67,068,379 |

**Growth mechanism.** Rather than collecting once and hoping the target was
hit, `auto_grow_and_tokenize.py` ran the full S02–S14 pipeline, checked both
absolute lane targets against the real measured output, and — for whichever
lane fell short — raised that lane's still-growable sources' collection
targets and re-ran, up to 4 rounds, with a per-source growth ceiling so the
loop cannot runaway-fill the disk. Both targets were met without ever
loosening a cleaning filter to manufacture volume.

---

## 3. Cleaning pipeline

Hindi and Nepali use two different cleaning implementations — see the note
at the top of this report. Each is described in its own subsection below.

### Hindi cleaning pipeline

`scripts/clean_lib.py`, run via `clean_hindi.py` against Hindi's own `raw/`
tree only. The measured stage order, confirmed below from
`clean_stats.json`'s own checkpoints (not assumed from documentation), is:

```
raw → malformed-Unicode removal → too-short removal → language-ID filter
    → exact/near-duplicate removal → cleaned
```

**Why this order is verified, not assumed:** `clean_stats.json` records a
mid-pipeline checkpoint, `docs_interim`, taken after malformed+short-doc
removal but before the language filter:
`docs_in (773,606) − dropped_malformed (17) − dropped_short (63,132) =
710,457 = docs_interim` exactly. Then
`docs_interim (710,457) − dropped_language (3,371) − dropped_duplicate
(2,983) = 704,103 = docs_cleaned` (pre-cross-language-correction figure —
see below).

**Normalization: NFC, not NFKC.** Unicode NFC (canonical composition) is
used, not NFKC (compatibility composition). NFKC would strip or alter
U+200C (ZWNJ) and U+200D (ZWJ) — zero-width joiner/non-joiner characters
that look like formatting noise but are **semantically meaningful** in
Devanagari: they control whether a consonant cluster forms a conjunct
ligature or stays visually separate. Stripping them would silently corrupt
real orthographic distinctions. NFC preserves them; only genuine
zero-width/control characters unrelated to Devanagari orthography are
stripped separately.

**Latin-token removal, digits kept.** 8,016,102 residual Latin-script tokens
(English words, URLs, markup fragments) were stripped during cleaning.
Digits are kept, since numerals are not script-specific noise.

**Language filter.** `langdetect` has **no `bho` label** — it cannot itself
identify Bhojpuri text, and Hindi/Bhojpuri share vocabulary and script
closely enough that mislabeled Bhojpuri leaking into Hindi's corpus was a
real risk during collection (Bhojpuri was this project's original Model L —
see Section 6 for why it was later superseded by Nepali; the contamination
risk and the fix below predate that switch and remain part of Hindi's own
cleaning history, run unconditionally regardless of which language is
currently Model L). Hindi's `clean_stats.json` (`bhojpuri_reject_caveat`)
records: *"langdetect has no 'bho' label, so it cannot itself catch Bhojpuri
text misclassified as Hindi. `dropped_language.likely_bhojpuri` is an active
reject filter: any document where Bhojpuri lexical markers
(बा/बाटे/बाड़ें/एगो/गइल/भइल/कइल/etc., ≥2 hits) outnumber Hindi markers
(है/हैं/एक/गया/हुआ/किया/etc.) is dropped before reaching `data/cleaned/`.
This is a coarse lexical heuristic, not a language model, so it will not
catch every case."* Concretely, 1,365 documents were dropped by this filter.

**Historical cross-language correction (Hindi ↔ Bhojpuri).**
`scripts/validate_phase1.py`'s cross-language check, run while Bhojpuri was
still Model L, caught 2 documents whose content hash was shared between the
Hindi and Bhojpuri cleaned corpora — genuinely Bhojpuri news text
(containing verb forms like "कहलन", "जइहें") that `ai4bharat/sangraha` had
sourced as `hin`, an upstream labeling error. Both were removed from
Hindi's cleaned corpus: `hi_downloaded_sangraha_e6bb7e97f6` and
`hi_downloaded_sangraha_unverified_7ca84f1c71`. This is why Hindi's final
cleaned count is **704,101**, not `clean_stats.json`'s own recorded
704,103. This correction is permanent and independent of which language is
currently Model L — it stays part of Hindi's own cleaned corpus regardless.

**Deduplication.** Exact-hash matching on normalized text (`dedup_key`,
`clean_lib.dedup_key`); on a manual/downloaded tie, the **manual copy
wins**. Near-duplicate detection beyond exact-hash matching (e.g.
MinHash/LSH) was **not** run on Hindi's pipeline — see Section 8.

**Measured drop counts (Hindi):**

| Reason | Docs |
|---|---:|
| Malformed Unicode | 17 |
| Too short (<200 chars) | 63,132 |
| Low Devanagari ratio | 228 |
| Likely Bhojpuri (active reject filter) | 1,365 |
| langdetect mismatch | 1,778 |
| Exact/near-duplicate | 2,983 |
| Cross-language duplicate (targeted correction) | 2 |
| **Total dropped** | **69,505** |
| **Docs in → cleaned** | 773,606 → 704,101 |

Too-short documents are by far the largest single loss (63,132 of 69,505
total, 90.8%).

### Nepali cleaning pipeline

Stage-based (`nepali/scripts/s00_doctor.py` through `s14_corpus_stats.py`),
sharing the same `corpuslib` code as Hindi's own Nepali-parallel pipeline
copy (byte-identical, hashed at CI time, but operating on entirely separate
data) — not shared with Hindi's `clean_lib.py`.

**Document retention through the pipeline:**

| Stage | Docs in | Docs out | Retention |
|---|---:|---:|---:|
| S03 normalise | 2,655,504 | 2,655,504 | 100.00% |
| S04 language ID | 2,655,504 | 2,649,643 | 99.78% |
| S07 train-pool split-off | 2,649,643 | 2,251,378 | 84.97% |
| S05 heuristic filters | 2,251,378 | 2,245,354 | 99.73% |
| S06 perplexity filter | 2,245,354 | 2,178,418 | 97.02% |
| S08 dedup (MinHash/LSH) | 2,178,418 | 2,126,334 | 97.61% |
| S09 decontaminate (vs. val/test) | 2,126,334 | 2,109,223 | 99.20% |
| S10 balance lanes (train only) | 2,109,223 | 1,520,000 | 72.06% |

**Language ID.** `narrow_margin_vs_hi` (600 of 2,655,504 docs) is the
language-ID model catching documents too ambiguous to confidently call
Nepali rather than Hindi given the shared Devanagari script and overlapping
vocabulary — the same underlying risk Hindi's `likely_bhojpuri` filter
addresses for its neighbor language, handled here via a dedicated
margin-vs-Hindi check rather than a lexical marker list.

**Deduplication is MinHash/LSH, not exact-hash** (`num_perm=100, bands=20,
rows=5, shingle_size=5, threshold=0.55`) — a near-duplicate method Hindi's
pipeline does not use (see Section 8). Removed 52,084 of 2,178,418
documents (2,350 exact, 49,734 near); 21,273 cross-lane duplicate pairs
found, all resolved to the manual copy.

**Decontamination (S09)** is a step Hindi's pipeline has no equivalent
of: 17,111 train documents that matched held-out (val+test) content above
the dedup threshold were removed, preventing train/eval leakage at the
content level, not just the document-id level. A rescan afterward found 0
remaining matches.

**Lane balancing (S10).** The train pool's raw manual share was 23.27%
going into S10. Rather than loosening a filter, S10 trimmed the
**downloaded** lane (589,223 documents dropped, manual untouched) to reach
27.42% — safely above the 20% floor.

**Measured drop reasons (S04 language-ID + S05 heuristic filters):**

| Reason | Docs |
|---|---:|
| wrong_language (S04) | 1,599 |
| low_confidence (S04) | 1,189 |
| narrow_margin_vs_hi (S04) | 600 |
| too_short (S04) | 251 |
| other script-ratio buckets, ≤5 docs each (S04) | 121 |
| repeated_lines (S05) | 1,858 |
| too_few_sentences (S05) | 1,226 |
| too_short (S05) | 382 |
| link_farm (S05) | 481 |
| low_lexical_diversity (S05) | 5 |
| too_few_words (S05) | 8 |
| mean_word_too_long (S05) | 1 |

See **Figure 3** (cleaning funnel, both languages) and **Figure 4** (drop
reasons ranked, both languages). The two subplots in each figure use
different stage/reason vocabularies since the two pipelines measure
different things — this is expected, not an inconsistency.

![Cleaning funnel](figures/03_cleaning_funnel.png)

![Drop reasons breakdown](figures/04_drop_reasons.png)

### Cross-language check (current: Hindi vs. Nepali)

Run via `scripts/check_nepali_hindi_overlap.py`: every one of Nepali's
1,918,265 final documents had its `dedup_key` (identical algorithm to
`clean_lib.dedup_key`, computed fresh since Nepali's own pipeline doesn't
store one) checked against the full set of 704,101 dedup_keys already
computed for Hindi's cleaned corpus. **Result: 0 shared documents.** Full
run took 467.7 seconds.

---

## 4. Splits

Hindi splits at the **document level**, seed 42, using deterministic
hashing of `document_id` so the same document can never land in two
different splits regardless of processing order.

| | Hindi (train/val/test) |
|---|---:|
| Documents | 675,937 / 14,082 / 14,082 |
| Ratio | ~95.9% / 2.0% / 2.0% |

Nepali's split happens earlier in its pipeline, at S07 — val/test are held
out from the full document pool *before* S08 (dedup), S09 (decontamination)
and S10 (lane balancing) run on what remains, specifically so those later
stages never see or touch documents that will end up in eval. This is why
Nepali's train fraction (79.2%) differs from a simple document-count ratio
of the raw pool — later stages (dedup, decontamination, balancing) only
ever remove documents from train, not from val/test.

| | Nepali (train/val/test) |
|---|---:|
| Documents | 1,520,000 / 199,428 / 198,837 |
| Ratio | ~79.2% / 10.4% / 10.4% |

`scripts/validate_phase1.py`'s leakage check reports **no overlap** for
Hindi (`validate_final.log`). `scripts/validate_phase1_nepali.py` and
`nepali/data/final/split_stats.json` (`zero_doc_id_overlap: true`,
`flagged_splits: []`) independently confirm the same for Nepali. See
**Figure 5**.

![Train/validation/test split sizes](figures/05_split_sizes.png)

---

## 5. Tokenizers

### Hindi: SentencePiece Unigram

Trained on Hindi's own `data/splits/train/` only — never validation or
test. BPE builds a vocabulary by greedily merging the most frequent
adjacent pair of symbols, one merge at a time, with no mechanism to
reconsider an earlier merge once made. Unigram instead treats the
*segmentation itself* as a latent variable: it starts from a large
candidate set of subword pieces, then uses EM to iteratively re-estimate
piece probabilities and prune the candidate set down to the target
vocabulary size.

This matters for Hindi specifically because it is a morphologically rich,
agglutinative-suffixing language layered on top of Devanagari's
conjunct-cluster orthography. BPE's greedy, order-dependent merging has no
way to revisit a bad early merge inside a conjunct cluster or a common
suffix chain; Unigram's probabilistic re-estimation can. This is a
structural argument, not empirically confirmed for Hindi specifically here
(see Section 8) — though Section 5's Nepali subsection below reports an
actual BPE-vs-Unigram comparison that did run, for the other language.

**The cost:** Unigram's suffix-array construction stage is memory-heavy. On
this project's 16GB training machine, Hindi's suffix-array stage was the
binding memory constraint, peaking near 5.9GB.

### Nepali: SentencePiece BPE

`nepali/configs/tokenizer.yaml` configures `model_type: bpe` as this
pipeline's default (not chosen per-language for a stated linguistic
reason — Hindi's Unigram choice above predates Nepali's pipeline and the
two were configured independently). The config does include a genuine
safety net: `compare_unigram.enabled: true`, set to automatically also
train Unigram at the chosen vocab size and keep whichever scores better
*if* BPE's fertility exceeded 2.10 tokens/word. It never triggered — BPE's
measured fertility (1.4825–1.6988 across the sweep) stayed well under that
threshold — so BPE was kept without needing the fallback, a real
comparison-that-wasn't-needed rather than an unconsidered default.

**`byte_fallback` is explicitly disabled** (`byte_fallback: false`),
unlike Hindi's `byte_fallback=True`. This is a real, meaningful
methodological difference for interpreting UNK rate (see below).

### Full sweep table

Vocabulary range: **5,000–10,000** for both languages, swept at 5k/8k/10k
(Hindi) and 5k/6k/8k/10k (Nepali), per a course-TA ruling capping vocab
size at this range for a ~25M-parameter model (see "Embedding budget"
below).

Every number below was **re-measured fresh** for this report by
`report/phase1/make_plots.py`, loading each trained `.model` file directly
and encoding the first 2,000 non-empty documents of each language's own
held-out validation split (Hindi: `data/splits/validation/text/*.txt`;
Nepali: `data/final/val.jsonl`). Hindi's fresh numbers match
`hindi_pipeline_manifest.json` exactly to 4+ decimal places. Nepali's fresh
numbers are close to but not bit-identical to `nepali/tokenizer/
selection_report.json`'s own pipeline-reported sweep (e.g. fertility
1.4825 here vs. 1.4868 there, at vocab=10,000) — both are real
measurements, on two different (but same-methodology) 2,000-document
validation samples, not a discrepancy to root-cause.

**Hindi:**

| Vocab | chars/token | fertility | UNK rate | byte-fallback rate | vocab coverage | unused ids |
|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 3.3230 | 1.4961 | 0.000000 | 0.018301 | 0.9654 | 173 |
| 8,000 | 3.5622 | 1.3956 | 0.000000 | 0.019613 | 0.9762 | 190 |
| 10,000 | 3.6601 | 1.3583 | 0.000000 | 0.020213 | 0.9767 | 233 |

**Nepali:**

| Vocab | chars/token | fertility | UNK rate | byte-fallback rate | vocab coverage | unused ids |
|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 3.7882 | 1.6988 | 0.000000 | 0.000000 | 0.9776 | 112 |
| 6,000 | 3.9410 | 1.6329 | 0.000000 | 0.000000 | 0.9727 | 164 |
| 8,000 | 4.1725 | 1.5424 | 0.000000 | 0.000000 | 0.9616 | 307 |
| 10,000 | 4.3410 | 1.4825 | 0.000000 | 0.000000 | 0.9458 | 542 |

See **Figures 6–8**.

![Vocab sweep: chars per token](figures/06_vocab_chars_per_token.png)

![Vocab sweep: fertility](figures/07_vocab_fertility.png)

![Vocab sweep: byte-fallback rate](figures/08_vocab_byte_fallback.png)

### Why UNK rate is meaningless for Hindi but real for Nepali

For **Hindi**, UNK rate reads exactly 0.0 at every vocab size, but this is
structural, not a sign the vocabulary covers everything: `byte_fallback=
True` means any character sequence the model cannot represent as a normal
subword piece falls back to raw-byte pieces (`<0xNN>` tokens) instead of
`<unk>`. The metric that actually tracks real out-of-vocabulary pressure
for Hindi is **byte-fallback rate** (1.83%–2.02% across the sweep).

For **Nepali**, `byte_fallback` is disabled, so a 0.0 UNK rate at every
vocab size is not masked by a fallback mechanism — it means the trained
vocabulary genuinely represented 100% of the validation sample's
characters directly, with zero out-of-vocabulary fallback of any kind.
This is a real, meaningful difference in what "UNK rate = 0" means between
the two languages' tokenizers, not the same finding twice.

### Vocab-size choice

**Hindi: 8,000, chosen on a fertility/embedding-budget trade-off.**
Fertility improves with vocab size but with diminishing returns, while
vocab utilization degrades. The *fertility* gain from 8k→10k
(1.3956→1.3583, 2.7% relative) is much smaller than the 5k→8k gain
(1.4961→1.3956, 6.7% relative), for meaningfully more embedding cost.
Measured directly: **8,000 produces more total tokens than 10,000 for
Hindi** (505,852,028 vs. 491,883,387), pushing Hindi to 101.2% of the 500M
target versus 10,000's 98.4%.

**Embedding parameter budget.** At a ~25M-parameter total model with tied
input/output embeddings, the embedding table costs `vocab_size ×
model_dim` parameters. The course-TA ruling that capped vocab at 5k–10k
was stated as "a 24k vocab makes the embedding table about 49% of the
model" — corresponding almost exactly to `model_dim=512`:
`24,000 × 512 = 12,288,000 = 49.15%` of 25,000,000.

| Vocab | Embed params (d=512) | % of 25M |
|---:|---:|---:|
| 8,000 | 4,096,000 | 16.38% |
| 10,000 | 5,120,000 | 20.48% |
| ~~24,000~~ (ruled out) | 12,288,000 | 49.15% |

**Nepali: 10,000, chosen by the pipeline's own fertility-elbow rule.**
Unlike Hindi's cost/benefit write-up above, Nepali's selection is
automated: `nepali/tokenizer/selection_report.json` records *"no elbow
found within the swept range; fertility was still improving by more than
3% at every step, so the largest candidate (10,000) was selected"* — the
sweep's own stopping rule, not a manually argued trade-off. At 10,000,
Nepali's embedding table is 5,120,000 params (20.48% of a 25M budget) —
still within the TA's 5k–10k cap, at the top of the range rather than the
middle Hindi landed on.

### Sample tokenizations (real model output)

**Hindi** (vocab_size=8,000, `hindi_analyze_8000_final.log`):

```
text:   अतुल माहेश्वरी के निधन के कई महीनों बाद अमर उजाला में...
pieces: ['▁अ', 'तुल', '▁माह', 'ेश्वर', 'ी', '▁के', '▁निधन', '▁के', ...]
ids:    [357, 4081, 1847, 1993, 279, 259, 3143, 259, ...]

text:   नई दिल्ली
pieces: ['▁नई', '▁दिल्ली', '<0x0A>', 'नर', 'ेंद्र', '▁मोदी', ...]
ids:    [648, 513, 13, 1796, 2061, 713, ...]
```

**Nepali** (vocab_size=10,000, `nepali/tokenizer/evaluation_report.json`):

```
text:   नेपाली पद वर्ग सङ्खेय नाम अर्थ कमजोर भएको अवस्था; गल्ती
pieces: ['▁नेपाली', '▁पद', '▁वर्ग', '▁सङ्', 'खे', 'य', '▁नाम', '▁अर्थ', ...]
(45 tokens total)

text:   नेपाली पद वर्ग नाम अर्थ मान्छे; मानिस
pieces: ['▁नेपाली', '▁पद', '▁वर्ग', '▁नाम', '▁अर्थ', '▁मान्छे', ';', '▁मानिस', ...]
(45 tokens total)
```

See **Figure 9** for how these individual pieces fit into the whole
vocabulary's usage distribution.

![Token frequency distribution](figures/09_token_frequency_loglog.png)

Both languages show the expected Zipf-shaped curve: a small number of
pieces (common postpositions, punctuation) account for a large share of
all tokens emitted, with a long tail of rare pieces — plotted at each
language's own selected vocab size (Hindi 8,000, Nepali 10,000, which
differ, unlike Hindi/Bhojpuri which happened to both choose 8,000).

### Training sample size — stated plainly

SentencePiece does not train on the entire train split. Hindi uses
`input_sentence_size=2,000,000` (reservoir-sampling up to 2,000,000
sentences); Nepali's config sets `input_sentence_size=3,000,000`.

| Language | Chars used for training | Train split total chars | Fraction sampled |
|---|---:|---:|---:|
| Hindi | 372,725,734 (5k) / 372,681,654 (8k) / 372,709,882 (10k) | 1,692,881,268 | 22.0% |

Nepali's exact sampled-character count was not separately logged by its
pipeline the way Hindi's was; `nepali/tokenizer/selection_report.json`
records only per-vocab-size timing (~55s per candidate), not a sampled-
character checkpoint. This is a real reporting gap on Nepali's side, noted
rather than backfilled with an estimate — see Section 8.

This is SentencePiece's standard, documented mechanism for bounding
training cost on a large corpus. The held-out evaluation in the sweep
table above is run separately, on the validation split, so it measures the
trained tokenizer's real generalization regardless of how much of train it
saw while fitting. The full-corpus token counts in Section 6 are exact and
unsampled for both languages.

---

## 6. Results

**Hindi:** 505,852,028 total tokens (train 485,280,741 + validation
10,484,640 + test 10,086,647) — **101.2% of the ~500M target**, exceeding
it. Manual share: **28.65% char-weighted** (505,509,566 /
1,764,596,782 cleaned characters), **30.66% token-weighted** (155,104,651 /
505,852,028 tokens). Both clear the 20% floor with a comfortable margin.

**Nepali:** 635,204,333 total tokens (train 519,906,359 + val 57,839,396 +
test 57,458,578) — **127.0% of the ~500M target**, exceeding it by a wider
margin than Hindi. Manual share: **26.41% token-weighted** (167,787,251 /
635,204,333 tokens — Nepali's pipeline never tracked per-lane characters,
so no char-weighted figure exists for it). Both of the pipeline's own
absolute lane targets are also cleared independently: manual 167,787,251 ≥
100,000,000; downloaded 467,417,082 ≥ 400,000,000.

See **Figure 10**.

![Token count vs. the 500M target](figures/10_token_count_vs_target.png)

**Nepali was not always the plan.** This project's original Model L was
**Bhojpuri**: fully built, cleaned, split, tokenized, and passing every
validation check, with a documented, spec-permitted 16.8%-of-target
shortfall (84,217,032 tokens) since Bhojpuri's available public data is
genuinely much smaller than Nepali's. Once Nepali's own pipeline cleared
both its absolute lane targets, Bhojpuri's tracked pipeline code and
tokenizer files were removed from the repository (`bhojpuri/`) — its
numbers above are the historical record of that work, not a live,
browsable directory. Nepali's pipeline was resumed
after a course-wide deadline extension made finishing it worthwhile;
`nepali/scripts/auto_grow_and_tokenize.py` then grew whichever lane fell
short of its absolute target and re-ran the full pipeline until both
targets were met, landing at 127.0% of the planning target rather than a
documented shortfall.

---

## 7. Reproducibility

| | Hindi | Nepali |
|---|---|---|
| Split mechanism | seed 42, deterministic `document_id` hash | S07 holdout, before S08–S10 run on the remaining pool |
| Tokenizer determinism | SentencePiece `shuffle_input_sentence=True`, `input_sentence_size=2,000,000` | SentencePiece `shuffle_input_sentence=True`, `input_sentence_size=3,000,000` |
| Git commit (per `hindi_pipeline_manifest.json`) | `0b0012ecb3deb08b61a81cc6e989c9bc60289a31` | not separately recorded (see Section 8) |
| Python | 3.14.0 | 3.14.0 |
| sentencepiece | 0.2.2 | 0.2.2 |
| Platform | Windows-11-10.0.26200-SP0 | Windows-11-10.0.26200-SP0 (pipeline executed on a separate machine — see note below) |

**To rerun Hindi:**

```
python hindi/data/pipeline/download_hindi.py
python hindi/data/pipeline/ingest_manual.py
python hindi/data/pipeline/ingest_wikipedia_dump.py
python hindi/data/pipeline/clean_hindi.py
python hindi/data/pipeline/split_hindi.py
python hindi/tokenizer/train_tokenizer.py --vocab-size 8000
python scripts/validate_phase1.py
```

**To rerun Nepali:**

```
cd nepali/scripts
python s00_doctor.py
python auto_grow_and_tokenize.py        # full S02-S14, grows lanes to target
python ../../scripts/validate_phase1_nepali.py
python ../../scripts/check_nepali_hindi_overlap.py
```

**Then, either language:**

```
python report/phase1/make_plots.py      # regenerate every figure in this report
```

**Provenance note (Nepali only):** the corpus/tokenizer artifacts this
report cites for Nepali were produced by running `nepali/scripts/` on a
separate machine (the ~30+ hour S08 MinHash dedup pass was impractical on
the primary machine within the deadline window); only the pipeline's
output files were transferred into this repo, not live per-stage console
logs. `log.md`'s Nepali S08(production)–S14 entries are reconstructed from
those output files rather than logged in real time — flagged inline in
`log.md` itself.

**Large artifacts** (raw corpora, cleaned corpora, tokenizer files — too
large for git) are hosted on OneDrive:
[LMA DATA SET — shared OneDrive folder](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/swaraj_kumar_students_iiit_ac_in/IgCh3u8O2sr4QIJqmrUS1CllAV1C5HL0R9SOWUUVIzTgcdw?e=bgUHaG)
(an IIIT-institutional link, used deliberately rather than Google Drive —
see the repository root `README.md`'s "Large artifacts" section) — see
that same table for current upload status.

---

## 8. Limitations and what I would do differently

**Hindi:**

- **The language-ID proxy is a real, acknowledged weak point**, though now
  largely historical since Bhojpuri is no longer this project's Model L.
  `langdetect` has no Bhojpuri label at all, so `likely_bhojpuri` is a
  coarse lexical-marker filter, not a trained classifier.
  `scripts/validate_phase1.py`'s old cross-language content-hash check
  caught 2 real misses this heuristic let through (Section 3) — there are
  almost certainly more a hash-based check cannot catch. This filter still
  runs unconditionally as part of Hindi's own cleaning, even though its
  original purpose (protecting against a since-superseded Model L) no
  longer applies day to day.
- **SentencePiece training samples the corpus** (22.0% of train
  characters). Standard practice, and it does not bias the held-out
  evaluation numbers in Section 5 (measured separately, on validation text
  the tokenizer never trained on), but the trained model saw under a
  quarter of Hindi's train split's characters.
- **Near-duplicate detection beyond exact-hash matching was cut for time
  on Hindi's pipeline specifically.** Hindi only dedups on an exact
  normalized-text hash (`dedup_key`); Nepali's pipeline, by contrast, does
  run MinHash/LSH near-duplicate detection (Section 3) — so this is now an
  asymmetry between the two languages' pipelines, not a shared gap.
- **The BPE-vs-Unigram argument for Hindi (Section 5) is structural, not
  empirically confirmed for Hindi specifically.** No side-by-side BPE
  sweep was run on Hindi's data. Nepali's pipeline, by contrast, does run
  this comparison automatically (`compare_unigram`, Section 5) — it just
  never triggered, since Nepali's BPE fertility stayed under the 2.10
  threshold.
- **A small raw-total discrepancy** (528 characters, 0.00003%, between
  the per-source breakdown sum and `clean_stats.json`'s own `chars_in`)
  was not root-caused. Too small to affect any pass/fail check.

**Nepali:**

- **Manual share is token-weighted, not char-weighted**, unlike Hindi's.
  Nepali's pipeline never added per-lane character tracking; token-weighted
  is the honest number available, not a like-for-like methodology with
  Hindi's.
- **Manual-lane concentration**: 91.6% of manual tokens come from one
  source (`news_onlinekhabar`). Disclosed in Section 2 rather than
  smoothed over.
- **SentencePiece's sampled-character count was not logged** by Nepali's
  pipeline the way Hindi's was (Section 5) — a real reporting gap, not
  backfilled with an estimate.
- **The Nepali pipeline ran on a separate machine** (Section 7's
  provenance note); its per-stage console logs for the later stages
  (S08–S14) were not captured live, only reconstructed from output
  artifacts in `log.md`.
- **Cross-language check covers Nepali vs. Hindi only** (Section 3), not
  Nepali vs. Bhojpuri, since Bhojpuri is not part of the final submission.



## report/phase2/phase2_report.md

# Phase 2 — Model Implementation, Pretraining, and Evaluation

**Model H = Hindi (higher-resource) · Model L = Nepali (lower-resource)**

Two independent ~25M-parameter decoder-only Transformers, written from scratch
in PyTorch, pretrained on separate monolingual corpora with separate tokenizers
and vocabularies, and evaluated side by side.

---

## 1. Headline results

| | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| Vocabulary (Phase 1) | 8,000 Unigram | 10,000 BPE |
| Total parameters | 23,273,472 | 24,297,472 |
| **Non-embedding parameters** | **18,915,328** | **18,915,328** |
| Training steps completed | 14,520 | 14,440 |
| Training tokens | 475,791,360 | 473,169,920 |
| Wall clock (Tesla T4) | 11.0 h @ 12.0k tok/s | 11.0 h @ 12.0k tok/s |
| Evaluated checkpoint | step 14,500 | step 14,000 |
| Validation cross-entropy | 3.2985 | 3.8317 |
| **Validation perplexity** | **27.07** | **46.14** |
| Test cross-entropy | 3.3109 | 3.8433 |
| **Test perplexity** | **27.41** | **46.68** |
| **Test bits-per-byte** | 0.5382 | **0.4846** |
| Mean attention entropy | 3.205 | 3.090 |
| Mean attention distance | 62.4 tokens | 54.5 tokens |

The two models are deliberately identical in depth, width, head count,
feed-forward size, context length and token budget. Their non-embedding
parameter counts are *exactly* equal and their token budgets differ by 0.55%,
so the differences below are attributable to data and tokenizer rather than to
capacity or compute.

---

## 2. Architecture

Built entirely from `nn.Linear`, `nn.Embedding`, `nn.Dropout` and raw tensor
operations. No `nn.Transformer*`, no `nn.MultiheadAttention`, no
`F.scaled_dot_product_attention`, no HuggingFace class. A test walks the module
tree and fails if any of those ever appear.

Implementation: [`hindi/model/transformer.py`](../../hindi/model/transformer.py),
[`nepali/model/transformer.py`](../../nepali/model/transformer.py).

**Input representation.** Token ids index a `(V, d)` embedding matrix. Position
is injected separately because self-attention is permutation invariant — it
treats its input as a set, so without a position signal "A is bigger than B"
and "B is bigger than A" would produce identical representations. The two are
added and passed through embedding dropout.

We use **learned absolute** positional embeddings. Chosen over sinusoidal
because the model is trained and evaluated at a single fixed context length,
where sinusoidal extrapolation buys nothing, and learned positions let the model
allocate resolution where the data needs it. **This makes `block_size = 512` a
hard ceiling**: there is no embedding row for position 512 or beyond, so the
model cannot be run on longer sequences without interpolating or retraining that
table. The forward pass raises rather than silently indexing out of range.

**Multi-head causal self-attention.** `W_q`, `W_k`, `W_v` are kept as three
separate `nn.Linear` modules rather than one fused `(d, 3d)` matmul, so they map
one-to-one onto the spec's notation. Shapes go
`(B, T, d) → (B, T, h, d_k) → (B, h, T, d_k)` via `view` then `transpose`, and
back via `transpose` then `contiguous().view` — `contiguous()` is required
because `transpose` only permutes strides while `view` needs a packed buffer.

Scores are `Q Kᵀ / √d_k`. **The scaling is needed** because `Q` and `K` entries
are roughly unit variance at initialisation, so a dot product over `d_k`
dimensions has variance on the order of `d_k`. Unscaled, the logits grow with
`d_k`, the softmax saturates toward one-hot, and gradients vanish. Dividing by
`√d_k` restores unit-scale logits.

**Causal masking** is an upper-triangular boolean buffer applied with
`masked_fill(-inf)` before the softmax, so future positions receive exactly zero
probability. Heads are concatenated and passed through `W_o`.

**Blocks.** Pre-norm: `x = x + Attn(LN(x))`, then `x = x + FFN(LN(x))`. Pre-norm
was chosen because it leaves a clean, unnormalised residual path from embeddings
to output head, so gradients reach early layers undistorted; post-norm puts a
LayerNorm on that path, which is what makes deep post-norm stacks sensitive to
warmup. The feed-forward network is `512 → 2048 → 512` with GELU — wider inside
than the model dimension, which is where most of the parameters and most of the
learned transformation capacity sit. Dropout is applied to attention
probabilities and to both sublayer outputs. Six blocks, then a final LayerNorm.

**Output head.** A linear map `d → V` produces next-token logits, **tied** to
the input embedding matrix. Tying saves `V × d` parameters — 4,096,000 for
Hindi (~18% of the model) and 5,120,000 for Nepali (~21%) — and forces the input
and output representations of a token to coincide. Training minimises
cross-entropy between the logits at position `t` and the token at position
`t+1`, averaged over all positions.

### 2.1 Causal masking, verified empirically

The spec asks for a demonstration that the model cannot see the future.
Perturbing every token after position `t` and measuring the change in logits at
positions `≤ t`, across every position of a full-length sequence:

```
max logit leakage = 0.000e+00
```

**Exactly zero, not merely within tolerance.** `softmax(-inf)` is exactly `0`
and `0 × v = 0` exactly in IEEE arithmetic, so masked positions contribute
nothing at all. The same test asserts the converse — that the perturbation *does*
change logits at positions `≥ t+1` — so it cannot pass on a model that simply
ignores its input.

Both suites pass 9/9 independently:
`python hindi/tests/test_model.py`, `python nepali/tests/test_model.py`.

### 2.2 Configuration and the depth/width trade-off

| | Both models |
|---|---|
| `n_layer` / `n_head` / `n_embd` / `d_ff` | 6 / 8 / 512 / 2048 |
| Context length | 512 |
| Dropout | 0.1 |
| Weight tying | yes |
| Positional encoding | learned absolute |

At `n_embd = 512` each block costs ~3.15M parameters, so six blocks plus the
embedding table lands just under the ~25M target; a seventh would overshoot to
26.4M. Depth was spent on six blocks rather than a wider four-block model
because composing multi-hop dependencies needs several sequential rounds of
attention, while widening grows the feed-forward cost quadratically for the same
budget. The attention analysis in §6 supports this: the useful specialisation
(sharp local heads at layers 2–4, long-range heads at layer 5) only emerges with
depth.

Configs: [`hindi/configs/model.yaml`](../../hindi/configs/model.yaml),
[`nepali/configs/model.yaml`](../../nepali/configs/model.yaml).

---

## 3. Pretraining

AdamW (β = 0.9/0.95, weight decay 0.1 on matrices only — not on biases,
LayerNorm gains or embeddings, which set scales rather than perform the linear
mixing decay is meant to regularise), gradient clipping at 1.0, and a cosine
schedule decaying from 6e-4 to 6e-5 after 500 warmup steps. Effective batch
32,768 tokens (32 × 2 × 512). Validation every 500 steps. Mixed precision
float16 with a gradient scaler — Kaggle's T4 is `sm_75` and has no bfloat16.

Every checkpoint contains model weights, optimizer state, scheduler state,
current step, tokens consumed, best validation loss, the full run configuration
and both RNG states. Resume was verified across three consecutive sessions
before the real runs began.

**Both runs stopped a little short of the planned 15,000 steps** — 14,520 and
14,440 — because measured throughput (12.0k tok/s) was lower than projected and
the notebook's 11-hour wall-clock budget fired first, exiting cleanly with a
resumable checkpoint. This is reported as-run rather than as-planned. The two
budgets remain within 0.55% of each other, so the comparison is unaffected.

![Loss curves against tokens](figures/loss_curves_both.png)

![Loss curves against steps](figures/loss_curves_both_by_step.png)

Validation loss decreases monotonically for both models to the end of training,
with no divergence between training and validation loss — at ~475M tokens
against ~23M parameters, neither model is near overfitting, and both would
likely continue improving with more compute.

---

## 4. Intrinsic evaluation

Scored over 1,999,872 tokens per split for each model, walking the split in
non-overlapping windows so every token is scored exactly once.

| | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| Validation cross-entropy (nats) | 3.2985 | 3.8317 |
| Validation perplexity | **27.07** | 46.14 |
| Validation bits-per-byte | 0.5357 | **0.4822** |
| Test cross-entropy (nats) | 3.3109 | 3.8433 |
| Test perplexity | **27.41** | 46.68 |
| Test bits-per-byte | 0.5382 | **0.4846** |
| Bytes per token | 8.88 | 11.44 |

### 4.1 The two metrics disagree, and that is the finding

By perplexity Model H looks 41% better. By bits-per-byte Model L is slightly
*better*. Both numbers are correct; they measure different things.

Perplexity is per **token**, and the two tokenizers do not produce comparable
tokens. Nepali's BPE-10000 packs **11.44 bytes per token** against Hindi's
Unigram-8000 at **8.88** — a 29% coarser segmentation. Coarser tokens are
individually harder to predict, which inflates perplexity, but each one carries
proportionally more text.

Bits-per-byte divides the same total log-likelihood by the UTF-8 byte length of
the underlying text, removing the tokenizer from the denominator:

```
Model H: 4.777 bits/token ÷ 8.88 bytes/token = 0.5382 BPB
Model L: 5.545 bits/token ÷ 11.44 bytes/token = 0.4846 BPB
```

**Interpretation.** This does *not* mean Model L is the better language model.
It means the headline perplexity gap is substantially a tokenizer artifact, and
on the tokenizer-neutral measure the two models are much closer than perplexity
suggests — with Model L marginally ahead. This is exactly why the spec asks for
BPB "to make the two languages more comparable when tokenizers differ".

Two honest caveats:

* **BPB is deflated for both models by Devanagari's UTF-8 encoding.** Devanagari
  costs three bytes per character, inflating the denominator. These numbers are
  comparable *Hindi ↔ Nepali* (both Devanagari) but must not be compared against
  a Latin-script model.
* **Model L's corpus carries a 0.17% unknown-token rate** (§7), which slightly
  perturbs its byte accounting. Model H's is exactly zero.

### 4.2 Discussing the gap

Contributing factors, in what we judge to be descending order of importance:

1. **Tokenizer granularity** (§4.1) — accounts for most of the apparent gap.
2. **Corpus composition.** Model L's corpus is nominally larger (521M vs 486M
   train tokens) but 91.6% of its manual lane is a single news source, and its
   downloaded lane leans on two AI4Bharat corpora. Model H draws on five
   distinct manual sources and five downloaded partitions. Diversity, not raw
   size, is the plausible difference — Model L saw *more* tokens of *narrower*
   text.
3. **Tokenizer quality.** Model L's BPE has no byte-fallback and drops 0.17% of
   tokens to `<unk>`; Model H's Unigram has byte-fallback and reaches exactly
   zero. Every `<unk>` is a position the model cannot represent, let alone
   predict.
4. **Script is not a factor here** — both are Devanagari, which is precisely why
   this pairing isolates data and tokenizer effects unusually cleanly.

---

## 5. Generation quality

Sixty-four held-out prefixes of 64 tokens each, continued for 64 tokens under
greedy decoding and temperatures 0.5 / 1.0 / 1.5, scored against the true
continuations. BLEU is corpus-level BLEU-4 with `intl` tokenization.

![Generation metrics](figures/generation_metrics_both.png)

| Model | Decoding | BLEU-4 | chrF | chrF++ | ROUGE-L | Repetition | Distinct-1 | Distinct-2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| H | greedy | 1.96 | 14.15 | 12.49 | 0.093 | 0.598 | 0.172 | 0.313 |
| H | T=0.5 | 1.80 | 16.05 | 14.33 | 0.096 | 0.212 | 0.275 | 0.602 |
| H | T=1.0 | 1.01 | 18.51 | 16.29 | **0.100** | 0.002 | 0.520 | 0.939 |
| H | T=1.5 | 0.16 | 17.90 | 14.67 | 0.053 | 0.000 | 0.702 | 0.992 |
| L | greedy | 2.37 | 16.36 | 14.48 | 0.081 | 0.577 | 0.179 | 0.263 |
| L | T=0.5 | 1.88 | 18.42 | 16.54 | 0.092 | 0.190 | 0.345 | 0.611 |
| L | T=1.0 | 1.69 | **20.02** | **18.05** | 0.079 | 0.001 | 0.634 | 0.948 |
| L | T=1.5 | 0.78 | 18.31 | 16.13 | 0.035 | 0.000 | 0.793 | 0.995 |

### 5.1 Why each metric is or is not informative here

* **BLEU-4** rewards exact 4-gram overlap against a *single* reference. Open-ended
  continuation has no single correct answer, so absolute values are near the
  floor (0.16–2.37) and should not be read as a quality score. It is only
  informative *relatively*, across decoding settings of the same model — where it
  falls monotonically with temperature, telling us sampling diverges from the
  reference, which we already know. It is the least useful of the three.
* **chrF / chrF++** operate on character n-grams, so they degrade gracefully with
  the rich inflectional morphology of both languages and need no word tokenizer.
  chrF++ peaks at T=1.0 for both models — the only metric here whose optimum
  corresponds to output a reader would judge best. **It is the most trustworthy
  of the three.**
* **ROUGE-L** measures longest-common-subsequence overlap and is somewhat more
  forgiving than BLEU about word order. We compute it with a local LCS
  implementation over whitespace tokens rather than via `rouge_score`, whose
  default tokenizer lowercases and strips non-ASCII-alphanumeric characters —
  which would delete Devanagari entirely and silently report zero.

### 5.2 Diversity, repetition, and qualitative notes

The clearest result in this section is the **degeneration of greedy decoding**.
Both models repeat roughly 58–60% of their generated 4-grams under greedy
decoding, with Distinct-1 collapsing to ~0.17. The samples show it directly:

**Model H, greedy** — locks into a loop:
> …इस सिद्धान्त के अनुसार, "अर्थात्, **अष्टांग, अष्टांग, अष्टांग**…

**Model H, T=0.5** — still degenerate, on a different token:
> …उस पर चिन्तन कर सकने वाला **साधक साधक साधक साधक**…

**Model L, greedy** — repeats at phrase rather than word level:
> …कृषि **अनुदानको व्यवस्था गर्न जरुरी छ । कृषि अनुदानको व्यवस्था गर्न जरुरी छ ।**…

**Model L, T=0.5** — coherent and topical:
> …, ब्याज, ब्याज, अनुदान, कृषि ऋण आदिको व्यवस्था गर्नु आवश्यक छ । तर, यो नीतिले कृषिको उत्पादकत्व…

Raising temperature to 1.0 essentially eliminates repetition (0.002 and 0.001)
and roughly triples Distinct-1, while chrF++ simultaneously *improves* — so for
these models the diversity gain is not bought at the cost of fidelity, up to
T=1.0. At T=1.5 chrF++ falls and the text loses grammatical coherence.

**Language correctness.** Both models produce well-formed Devanagari with
correct conjunct formation and plausible morphology at T ≤ 1.0. Model L's
outputs are noticeably more topically consistent within a sample — a likely
consequence of its narrower, news-dominated corpus (§4.2), which makes it
easier to stay on-register but is not evidence of a better model. Neither model
maintains factual or discourse coherence beyond roughly one sentence, which is
expected at 23M parameters and ~475M training tokens.

---

## 6. Attention analysis

Statistics computed over the causally valid region only — query position `i` has
exactly `i + 1` reachable keys, so averaging over the masked upper triangle
would wash the numbers out with structural zeros.

![Attention summary](figures/attention_summary_both.png)

| | Model H | Model L |
|---|---:|---:|
| Mean entropy (nats) | 3.205 | 3.090 |
| **Uniform-attention baseline** | **5.256** | **5.256** |
| Mean attention distance | 62.4 | 54.5 |

**Both models learned to select rather than average.** Mean entropy sits ~2
nats below the uniform baseline of 5.256 — the value a head that merely averaged
its context would show. (An untrained model measured 3.256 against a baseline of
3.257 on a short sequence, i.e. indistinguishable from uniform, which is the
control for this claim.)

### 6.1 Layer-by-layer structure

Sharpest head per layer, with its mean attention distance:

| Layer | Model H entropy @ distance | Model L entropy @ distance |
|---|---|---|
| 0 | 4.16 @ 45.8 | 3.90 @ 36.0 |
| 1 | 3.09 @ 21.9 | 3.17 @ 17.1 |
| 2 | 1.49 @ 7.6 | 1.81 @ 10.6 |
| 3 | **0.06 @ 1.3** | **0.14 @ 1.1** |
| 4 | 1.49 @ 22.4 | 1.32 @ 5.6 |
| 5 | 2.99 @ 151.4 | 2.48 @ 81.3 |

The two models, trained independently on different languages with different
tokenizers and different random seeds, **converged on the same organisation**:

* **Layer 0 is diffuse and long-range** (layer-mean distance 99.6 / 95.4) — with
  no useful representation yet, early heads pool broadly.
* **Layers 2–4 sharpen and localise.** Both models learned a
  **previous-token head at layer 3** — entropy 0.06 at distance 1.3 for Model H,
  0.14 at distance 1.1 for Model L. Entropy near zero means the distribution is
  almost one-hot; distance near one means it points at the immediately preceding
  token. These are positional/local heads, and their independent emergence in
  both runs is strong evidence the structure is driven by the language-modelling
  objective rather than by any particular corpus.
* **Layer 5 widens again** (mean distance 105.5 / 53.9, with individual heads out
  to 151 tokens) — late heads gather long-range context to make the final
  prediction. These are the content-based heads: high distance combined with
  moderate rather than minimal entropy indicates selection over many positions,
  not a fixed offset.

**How the models differ.** Model H attends ~15% further on average (62.4 vs 54.5
tokens) and has a smaller fraction of local heads (0.04 vs 0.10). Model L's
attention is more concentrated nearby. This is consistent with its coarser
tokenizer — at 11.4 bytes per token, a fixed 512-token window covers ~29% more
text, so a shorter *token* distance spans a comparable amount of *language*.
Measured in bytes rather than tokens the two mean distances are within ~10% of
each other, which suggests the apparent difference is largely another tokenizer
effect rather than a difference in what the models learned.

### 6.2 Heatmaps

Early layer (0) and late layer (5), four heads each, with real Devanagari token
labels. Example sentences are stored as text in `*_attention.json` so the
visualised example is recoverable independently of font availability.

**Model H** — example sentence:
> किए थे । २०० वर्षों तक जमे हुए ब्रिटिश शासन को 'भारत छोड़ो' के उद्घोष से भगा देना विश्व के इतिहास का गौरवपूर्ण परिवर्तन है ।

![Hindi layer 0](figures/hindi_attention_layer0.png)
![Hindi layer 5](figures/hindi_attention_layer5.png)

**Model L** — example sentence:
> अनुसार नयाँ वर्ष मनाउन चीन गएका उनीहरु त्यसपछि फर्किएका छैनन् । चीन गएकामध्ये एकजना परियोजना…

![Nepali layer 0](figures/nepali_attention_layer0.png)
![Nepali layer 5](figures/nepali_attention_layer5.png)

The layer-0 maps show broad horizontal banding — many queries attending to
similar broad regions. The layer-5 maps show structured off-diagonal attention,
including columns where many query positions concentrate on a few informative
keys.

---

## 7. Resource-level comparison

The two models were built to make this comparison meaningful: identical
architecture, identical non-embedding parameter count (18,915,328), token
budgets within 0.55%, and the same script. The only material differences are the
corpus and the tokenizer.

| Factor | Model H (Hindi) | Model L (Nepali) | Effect |
|---|---|---|---|
| Train tokens | 485.9M | 521.4M | L nominally larger |
| Manual share | 30.66% | 26.41% | comparable |
| Source diversity | 5 manual + 5 downloaded | 1 dominant manual (91.6%) + 2 downloaded | **H more diverse** |
| Tokenizer | Unigram 8,000 | BPE 10,000 | — |
| Bytes/token | 8.88 | 11.44 | L coarser by 29% |
| Unknown-token rate | 0.000000 | 0.001646 (train) | **H cleaner** |
| Test perplexity | 27.41 | 46.68 | H better (tokenizer-dependent) |
| Test bits-per-byte | 0.5382 | 0.4846 | **L marginally better** |

**Conclusion.** The lower-resource model is *not* meaningfully behind on
language-modelling quality once the tokenizer is factored out. What separates
them is not corpus size — Model L trained on 7% more tokens — but corpus
*diversity* and tokenizer *hygiene*. The single most consequential Phase 1
decision for Model L turned out to be training a BPE vocabulary without byte
fallback, which both coarsened its segmentation (inflating perplexity) and
introduced an irreducible 0.17% `<unk>` floor.

An important methodological point: had we reported perplexity alone, we would
have concluded that the lower-resource model was substantially worse. It is the
tokenizer-neutral metric that reveals the two are close. This is the clearest
practical lesson of the phase.

---

## 8. Honest limitations

* Both runs stopped at 14,520 / 14,440 of a planned 15,000 steps when the
  11-hour session budget fired. Reported as-run.
* The evaluated checkpoints are each language's best-validation checkpoint,
  which fell at step 14,500 for Model H and 14,000 for Model L.
* Intrinsic metrics are computed over 2M-token subsets of each split, not the
  full 10.1M / 57.7M test sets. Both models use the identical budget.
* Final evaluation ran on CPU for both models. Temperature sampling draws from
  device RNG, so evaluating one model on GPU and the other on CPU would not have
  been strictly comparable; both were therefore re-run on the same device with
  the same code.
* The optional bonus ablation (no positional embeddings) was not run: the two
  pretraining runs consumed the 30-hour weekly GPU quota.
* Model L's 0.17% unknown-token rate is inherited from a Phase 1 artifact and
  was deliberately not corrected, since the tokenizer is itself a graded Phase 1
  deliverable.

---

## 9. Reproduction

Every command names exactly one language; no entry point runs both.

```bash
# Pack Phase 1 splits into uint16 token streams
python hindi/data/pack_corpus.py
python nepali/data/pack_corpus.py

# Verify the model (causal mask, shapes, parameter count, weight tying)
python hindi/tests/test_model.py
python nepali/tests/test_model.py

# Pretrain
python -m hindi.train.train  --config hindi/configs/model.yaml
python -m nepali.train.train --config nepali/configs/model.yaml

# Evaluate
python -m hindi.eval.evaluate  --checkpoint checkpoints/hindi/ckpt_best.pt  --split test
python -m nepali.eval.evaluate --checkpoint checkpoints/nepali/ckpt_best.pt --split test

# Regenerate every figure for both models
python scripts/make_phase2_figures.py \
    --hindi-checkpoint checkpoints/hindi/ckpt_best.pt \
    --nepali-checkpoint checkpoints/nepali/ckpt_best.pt
```

On Kaggle: [`notebooks/kaggle_pretrain.ipynb`](../../notebooks/kaggle_pretrain.ipynb)
(once per language) and
[`notebooks/kaggle_evaluate.ipynb`](../../notebooks/kaggle_evaluate.ipynb).

## 10. Artifacts

In this directory: `{hindi,nepali}_eval.json`, `{hindi,nepali}_attention.json`,
`{hindi,nepali}_training_log.jsonl`, `{hindi,nepali}_config.yaml`,
`hindi_kernel.log.json`, and `figures/`.

Checkpoints are too large for git and are linked from the top-level README.



## report/phase3/phase3_report.md

# Phase 3 — Reasoning Finetuning, Attention Analysis, and Final Report

**Model H = Hindi** (higher-resource) · **Model L = Nepali** (lower-resource)
Two fully independent monolingual decoder-only Transformers, built from
primitive PyTorch layers, pretrained separately, and finetuned separately on
synthetic comparative-reasoning data generated in each model's own language.

---

## 0. Summary of findings

Finetuning produced a large, unambiguous improvement in one thing and no
improvement at all in another, and separating the two is the main result of this
phase.

**What finetuning taught.** Answer-token cross-entropy fell from 5.2439 to
0.9408 for Model H and from 5.5367 to 1.4351 for Model L. Before finetuning,
neither model produced a bare entity name as a continuation even once in 2,000
test questions — exact match was 0.00% for both. After finetuning, Model H
answers with a correctly-formatted name 35.15% of the time and Model L 29.45%.

**What finetuning did not teach.** Neither model learned the comparison. On a
forced choice between the entities the question actually names, Model H scores
36.35% and Model L 39.30% against a chance rate of **41.67%** — both at or below
chance. Per task the agreement with chance is close to exact: Model H's
forced-choice score is 50.40% and 47.00% on the two two-candidate families
(chance 50%) and 33.20% on three-candidate `numeric3` (chance 33.3%).

**Why the two-hop task scores below chance.** On transitive questions of the
form *"A is taller than B. B is taller than C. Who is tallest?"*, Model H
answers with the **middle entity B in 72.6% of cases and with A in 0.0%**. B is
never the correct answer. The model is selecting the name that appears twice
rather than following the chain. This is a systematic, measurable failure mode,
not noise, and it is why `transitive3` lands at 12.40% where random guessing
among three names would score 33.3%.

**Where the two resource tiers differ.** The clearest difference is not accuracy
but *grounding*: Model L answers with a name the question never mentions in
19.2–28.8% of cases, against 0.0–7.2% for Model H. The lower-resource model
falls back on memorised training names far more often. The held-out name pools
are what make this measurable.

The honest headline is that **a 25M-parameter model trained on ~500M tokens
learns the surface form of this task and not its content**, and that the
evaluation design — a forced-choice metric with a known chance rate, plus
held-out entity names — is what makes that distinguishable from success.

---

## 1. Synthetic reasoning dataset

Built programmatically so every ground-truth label is known by construction, as
the specification requires. No existing benchmark was downloaded.

Generation scripts: [`hindi/reasoning/generate.py`](../../hindi/reasoning/generate.py)
· [`nepali/reasoning/generate.py`](../../nepali/reasoning/generate.py)

### 1.1 Task families

| Family | Hops | Candidates | Example (Hindi) |
|---|---:|---:|---|
| `compare2` | 1 | 2 | विजय सोहन से धीमा है। दोनों में से तेज़ कौन है? |
| `transitive3` | 2 | 3 | मीना प्रीति से छोटी है। प्रीति सुनीता से छोटी है। सबसे बड़ी कौन है? |
| `numeric2` | 1 | 2 | संगीता की उम्र १३ साल है। रेखा की उम्र ६८ साल है। बड़ी कौन है? |
| `numeric3` | 1 | 3 | तीन लोगों में … सबसे भारी कौन है? |

Five comparable attributes — height, age, weight, speed, wealth — with the last
two available only for non-numeric comparison, since neither has a natural
numeric phrasing in either language.

Both the stated direction and the asked direction vary independently. If the
sentence says *A is taller than B* and the question asks who is **shorter**, the
answer is B. Without this the answer would simply be the first-named entity.
Measured: the answer is the first-named entity in 46% of examples.

### 1.2 Size and composition

Identical for both languages; the two corpora share no sentence, name or
template.

| Split | Examples | Templates | Entity names | Per family |
|---|---:|---:|---:|---:|
| train | 20,000 | 14 | 40 | 5,000 each |
| validation | 2,000 | 4 | 10 | 500 each |
| test | 2,000 | 4 | 10 | 500 each |

Tokenized length on the test split:

| Model | Prompt tokens (mean / max) | Answer tokens (mean / max) | Unknown-token rate |
|---|---:|---:|---:|
| H | 26.63 / 47 | 3.00 / 4 | **0.000000** |
| L | 24.80 / 39 | 2.81 / 4 | **0.000000** |

Answer token counts include the end-of-sequence token.

### 1.3 Leakage control

Both defences the specification suggests are applied **together**, and the
generator asserts the guarantee rather than assuming it:

* **Held-out entity names.** 40 / 10 / 10, disjoint across splits. A test
  question is always about people the model never saw in training.
* **Held-out templates.** 14 / 4 / 4, disjoint across splits. A test question is
  phrased in a way the model never saw either.

`generate.py` fails loudly if the name pools ever intersect. Verified
programmatically: zero overlap between any pair of splits, and zero overlap
between the Hindi and Nepali name pools and template pools.

This was not a formality. Section 4 shows both models answering test questions
with **training** names; without disjoint pools that behaviour would have been
scored as partial success.

### 1.4 Two defects found and fixed during construction

**A 45% gender-agreement leak.** Hindi comparative adjectives inflect for
gender. A question ending *"सबसे छोटी कौन है?"* asked about one woman and two
men has exactly one grammatically possible answer, so a model could score well
by matching agreement without performing any comparison. 181 of an initial 400
examples were solvable this way. Fixed by drawing every entity in an example
from a single gender, which makes the adjective carry no information about which
entity is meant. Verified: zero mixed-gender examples.

**Nepali numerals were entirely unrepresentable.** Model L's BPE vocabulary
contains **no Devanagari digit at all** — not one of ०–९ is a vocabulary piece —
and it was trained without byte fallback, so every Devanagari numeral encodes to
`<unk>`. A first draft using them ran a 4.8% unknown-token rate across the
corpus, which would have trained Model L on holes where the quantities belong.
Model H's Unigram vocabulary covers both digit systems at zero unknowns.

The Nepali generator therefore writes numerals with Arabic digits, which the
vocabulary covers completely and which modern Nepali prose overwhelmingly uses
anyway. This asymmetry between the two generators is not an inconsistency; it is
the low-resource tokenizer's coverage gap appearing in practice, and it is
reported in §7.3 rather than papered over.

### 1.5 Natural phrasing

The specification requires the target language's script *and* natural phrasing.
Nepali orthography attaches the comparative postposition and the genitive to the
preceding word — **रबिनको उचाइ**, **कल्पनाभन्दा** — and writing them detached is
the usual signature of Hindi-shaped Nepali. The templates attach them.

---

## 2. Finetuning protocol

Scripts: [`hindi/reasoning/finetune.py`](../../hindi/reasoning/finetune.py) ·
[`nepali/reasoning/finetune.py`](../../nepali/reasoning/finetune.py)
Configs: [`hindi/configs/finetune.yaml`](../../hindi/configs/finetune.yaml) ·
[`nepali/configs/finetune.yaml`](../../nepali/configs/finetune.yaml)

Every requirement of the specified protocol is met:

* each model starts from **its own** Phase 2 checkpoint — Model H from step
  14,500, Model L from step 14,000 — never from the other model and never from
  scratch;
* each keeps **its own** Phase 1 tokenizer and vocabulary, so the embedding and
  output matrices keep their pretrained rows. The finetuner asserts that the
  tokenizer's vocabulary size matches the checkpoint rather than trusting the
  config;
* checkpoints carry weights, optimizer state, scheduler state, step counter and
  the full config — the same resume-capable format as pretraining.

### 2.1 Hyperparameters

Identical for both models, so any difference in results is attributable to data
and tokenizer rather than to the recipe.

| Setting | Value | Why |
|---|---|---|
| Peak learning rate | 6e-5 | One tenth of the 6e-4 pretraining peak. The model already holds a language model worth preserving. |
| Min learning rate | 6e-6 | Cosine floor. |
| Warmup | 100 steps | See §3.3 — this interacts with the result. |
| Batch size | 32 examples | |
| Epochs | 2 (1,250 steps) | |
| Optimizer | AdamW, β=(0.9, 0.95), wd 0.1 | Decay on matrices only. |
| Gradient clip | 1.0 | |
| Precision | fp16 + GradScaler | The Kaggle T4 is sm_75 and has no bfloat16. |
| Evaluation | every 20 steps | |
| Loss | **answer tokens only** | |

Seeds differ (20260914 / 20260915) so the two runs are not correlated through
their data ordering.

### 2.2 Two design decisions worth stating

**Loss is computed on the answer only.** Prompt target positions are set to
`-1`, which the model's `cross_entropy(..., ignore_index=-1)` already skips.
Training on the prompt too would spend most of the gradient reproducing question
templates that are given for free at test time.

**Batches are bucketed by prompt length, not padded.** The models use *learned
absolute* positional embeddings, so a token's representation depends on its
index. Left-padding would move every real token to a position it never occupied
during pretraining; right-padding is fine for training but breaks generation,
which appends at the end. Grouping examples whose prompts tokenize to the same
length avoids both, with no padding in the prompt at all.

---

## 3. Results: pretrained vs. finetuned

Reports: [`hindi_reasoning_pretrained.json`](hindi_reasoning_pretrained.json) ·
[`hindi_reasoning_finetuned.json`](hindi_reasoning_finetuned.json) ·
[`nepali_reasoning_pretrained.json`](nepali_reasoning_pretrained.json) ·
[`nepali_reasoning_finetuned.json`](nepali_reasoning_finetuned.json)

### 3.1 Why two metrics

A pretrained language model has never seen this prompt format and will not
spontaneously answer with a bare name, so end-to-end exact match on it measures
"did it guess the output format" and bottoms out at zero. That makes the
pretrained column uninformative exactly where the specification asks for a
comparison. So two metrics are reported:

* **Exact match** after greedy decoding — the honest end-to-end number.
* **Forced choice** among the entities the prompt names: each candidate is
  scored by its mean log-probability per answer token and the highest wins.
  Chance is known (41.67%, computed per example from the candidate count), so
  the pretrained baseline is interpretable rather than a floor artefact.

Scores are length-normalised. Without that, short names win on total
log-probability for reasons unconnected to the question.

### 3.2 Headline table

2,000 held-out test questions per model.

| Model | Metric | Pretrained | Finetuned | Change |
|---|---|---:|---:|---:|
| **H** (Hindi) | Exact match | 0.00% | **35.15%** | +35.15 |
| | Forced choice | 39.80% | 36.35% | −3.45 |
| | Answer-token loss | 5.2439 | **0.9408** | −4.3031 |
| **L** (Nepali) | Exact match | 0.00% | **29.45%** | +29.45 |
| | Forced choice | 38.80% | 39.30% | +0.50 |
| | Answer-token loss | 5.5367 | **1.4351** | −4.1016 |
| | *forced-choice chance* | *41.67%* | *41.67%* | — |

Neither model's forced-choice score reaches chance after finetuning.

### 3.3 Per task family

Exact match, finetuned, with the chance rate for that family's candidate count:

| Model | Family | Candidates | Chance | Pretrained | Finetuned |
|---|---|---:|---:|---:|---:|
| H | `compare2` | 2 | 50.0% | 0.00% | 47.40% |
| H | `numeric2` | 2 | 50.0% | 0.00% | 47.40% |
| H | `numeric3` | 3 | 33.3% | 0.00% | 33.40% |
| H | `transitive3` | 3 | 33.3% | 0.00% | **12.40%** |
| L | `compare2` | 2 | 50.0% | 0.00% | 40.20% |
| L | `numeric2` | 2 | 50.0% | 0.00% | 38.00% |
| L | `numeric3` | 3 | 33.3% | 0.00% | 26.20% |
| L | `transitive3` | 3 | 33.3% | 0.00% | **13.40%** |

Model H's one-hop scores sit within 2.6 points of chance and its `numeric3`
score within 0.1 points. The two-hop family is the outlier, and it is *below*
chance for both models.

### 3.4 Training dynamics

![Finetuning loss curves](figures/finetune_loss_curves_both.png)

Logs: [`hindi_finetune_log.jsonl`](hindi_finetune_log.jsonl) ·
[`nepali_finetune_log.jsonl`](nepali_finetune_log.jsonl)

Both models reach minimum validation loss at **step 60** — Model H at 1.4966,
Model L at 1.3427 — and rise monotonically afterwards, ending at 3.5188 and
3.5911 by step 1,250 while training loss falls to 0.44 and 0.60. That is severe
overfitting, and it is expected: with 40 training names and 14 templates, the
answer prior is memorisable long before copying from the prompt becomes
worthwhile. Saving on best validation loss meant the evaluated checkpoints were
step 60, not the badly overfit final ones.

An earlier run evaluated every 100 steps and selected its checkpoint at step 100
— which is also exactly where warmup ends and the learning rate first reaches
its peak. Re-running with a 20-step evaluation grid, changing nothing else,
moved Model L from 14.90% to 29.45% exact match and Model H from 32.70% to
35.15%. The conclusions in §0 were unchanged by this, but the numbers were, and
the coarse grid would have understated both models.

---

## 4. Error analysis

The `error_analysis` block of each report records, per task family, which prompt
position the model's answer came from, alongside where the gold answer actually
sits. Accuracy alone cannot distinguish uniform guessing from a positional habit
from a model ignoring the question entirely; this separates them.

### 4.1 The two-hop failure is systematic

`transitive3`, finetuned, distribution over answer position:

| Model | Answered A (first) | Answered B (middle) | Answered C (last) | Off-prompt |
|---|---:|---:|---:|---:|
| Gold | 47.4% | **0.0%** | 52.6% | — |
| H | **0.0%** | **72.6%** | 25.4% | 2.0% |
| L | 3.2% | 41.8% | 26.2% | 28.8% |

The middle entity is never a correct answer, because the chain is stated as
*A rel B, B rel C* and the extremes are always A and C. Model H nevertheless
selects B in 72.6% of cases and selects A in **none**. The explanation is
simple: B is the only name mentioned twice, and the model is selecting on
salience rather than following the inequality. That is precisely why the family
scores below chance — random guessing would at least hit A and C sometimes.

### 4.2 One-hop answers show a recency bias

`compare2` and `numeric2`, Model H:

| Family | Answered first-named | Answered second-named | Gold split |
|---|---:|---:|---:|
| `compare2` | 30.8% | **62.0%** | 48 / 52 |
| `numeric2` | 35.0% | **65.0%** | 50 / 50 |

Against a balanced gold distribution, Model H prefers the second-named entity
roughly two to one. The qualitative misses show it plainly — every one of these
is the model answering with the most recently mentioned name:

> ध्यान दीजिए: नवीन वरुण से अमीर है। अमीर कौन है? → **वरुण** (gold नवीन)
> ध्यान दीजिए: अरुण नवीन से बड़ा है। बड़ा कौन है? → **नवीन** (gold अरुण)

Model L is flatter — 43.8% / 37.0% on `compare2` — but only because a fifth of
its answers leave the prompt entirely.

### 4.3 Model L is much less grounded in the prompt

Rate of answering with a name the question never mentions:

| Family | Model H | Model L |
|---|---:|---:|
| `compare2` | 7.2% | **19.2%** |
| `numeric2` | 0.0% | **25.0%** |
| `numeric3` | 1.0% | **21.2%** |
| `transitive3` | 2.0% | **28.8%** |

Model L emits a memorised *training* name — repeatedly सिर्जना, निरु — instead
of one of the candidates in front of it. Both pretrained models do this 100% of
the time (99.6% for Model H on `transitive3`), so finetuning moved Model H
almost entirely onto the prompt and Model L only about three-quarters of the
way. This is the single clearest resource-tier difference in the phase.

### 4.4 Successes

Both models do get questions right, including ones requiring the direction flip:

> ध्यान दीजिए: प्रवीण वरुण से हल्का है। भारी कौन है? → **वरुण** ✓
> ध्यान दीजिए: वरुण प्रवीण से गरीब है। अमीर कौन है? → **प्रवीण** ✓
> ध्यान दिनुहोस्: निरु कल्पनाभन्दा धनी छ। धनी को हो? → **निरु** ✓

Given §4.1–4.2, these are best read as the positional habit coinciding with the
correct answer rather than as evidence of comparison.

---

## 5. Attention analysis: pretrained vs. finetuned

Module: [`hindi/reasoning/attention_compare.py`](../../hindi/reasoning/attention_compare.py)
· reports [`hindi_attention_comparison.json`](hindi_attention_comparison.json) ·
[`nepali_attention_comparison.json`](nepali_attention_comparison.json)

Both checkpoints are run on the **same** held-out comparative-reasoning prompts,
all of one token length, so neither the choice of text nor the sequence length
can masquerade as a finetuning effect. Prompts come from the reasoning test
split, so neither checkpoint has memorised them.

### 5.1 Whole-model summary

| Model | Checkpoint | Mean entropy (nats) | Mean distance (positions) | Uniform baseline |
|---|---|---:|---:|---:|
| H | pretrained | 1.9102 | 10.303 | 2.9545 (T=46) |
| H | finetuned | 1.8691 | 10.360 | |
| L | pretrained | 1.8327 | 9.415 | 2.8061 (T=39) |
| L | finetuned | 1.7833 | 10.158 | |

Both models sit about 1 nat below the uniform baseline on these prompts, so
attention is structured rather than diffuse. Finetuning **sharpened** attention
slightly in both (−0.041 and −0.049 nats) and moved it **further** along the
sequence, dramatically so for Model L: mean distance rises 0.74 positions, from
9.42 to 10.16, on a 39-token prompt.

That is the expected direction for this task — the answer is a name that appears
earlier in the prompt, so answering at all requires reaching back — and it is
consistent with §4.3, where Model L is the model that most needed to learn to
look at the prompt rather than at its own priors.

### 5.2 Per-head shifts

Layer means hide what actually happened; finetuning a small model for a narrow
task moves a few heads sharply. The largest movements:

| Model | Head | Δ entropy (nats) | Δ distance (positions) |
|---|---|---:|---:|
| H | L1H5 | −0.2734 | +1.257 |
| H | L3H1 | −0.2306 | **+2.585** |
| H | L4H3 | −0.2285 | +0.869 |
| H | L4H1 | — | −1.367 |
| L | L1H3 | −0.3336 | +1.836 |
| L | L5H5 | −0.3323 | +2.247 |
| L | L2H6 | −0.3062 | +1.412 |
| L | L3H3 | — | **+3.711** |

The pattern is consistent across both models: the heads that changed most became
**sharper and longer-range simultaneously**. Model L's L3H3 extends its reach by
3.7 positions — nearly a tenth of the whole prompt. Middle layers (1–3) dominate
the list in both models, which is where Phase 2 found the previous-token heads;
finetuning appears to have repurposed part of that machinery to look further
back for a name.

### 5.3 Heatmaps

Early layer (0) and late layer (5), four heads each, both checkpoints, both
models, with Devanagari token labels:

| | Pretrained | Finetuned |
|---|---|---|
| **H**, layer 0 | [figure](figures/hindi_pretrained_reasoning_layer0.png) | [figure](figures/hindi_finetuned_reasoning_layer0.png) |
| **H**, layer 5 | [figure](figures/hindi_pretrained_reasoning_layer5.png) | [figure](figures/hindi_finetuned_reasoning_layer5.png) |
| **L**, layer 0 | [figure](figures/nepali_pretrained_reasoning_layer0.png) | [figure](figures/nepali_finetuned_reasoning_layer0.png) |
| **L**, layer 5 | [figure](figures/nepali_pretrained_reasoning_layer5.png) | [figure](figures/nepali_finetuned_reasoning_layer5.png) |

![Attention summary](figures/hindi_attention_pretrained_vs_finetuned.png)

Layer 0 is dominated by a strong attention sink at position 0 plus a tight
diagonal band — local, positional, and essentially unchanged by finetuning.
Layer 5 is visibly content-dependent, with off-diagonal stripes that align to
the repeated entity names in the question. The causal structure is exact in
every panel: the upper triangle is uniformly zero, which is the visual form of
the Phase 2 masking test.

**A caveat, stated plainly.** These shifts are real but small, and §3–4 show the
model's answers did not become more correct. The most defensible reading is that
finetuning adjusted *where* the model looks without giving it the mechanism to
use what it finds — it learned to reach back for a name, but not which name.

---

## 6. Model H vs. Model L

| Dimension | Model H (Hindi) | Model L (Nepali) | Advantage |
|---|---:|---:|---|
| Corpus tokens (train split) | 485,956,678 | 521,426,359 | L |
| Corpus bytes/token | 8.875 | 11.441 | H (finer) |
| Vocabulary | 8,000 (Unigram) | 10,000 (BPE) | — |
| Parameters | 23.3M | 24.3M | — |
| Non-embedding parameters | 18,915,328 | 18,915,328 | identical |
| Test perplexity | **27.41** | 46.68 | H |
| Test bits-per-byte | 0.5382 | **0.4846** | L |
| Reasoning exact match | **35.15%** | 29.45% | H |
| Reasoning forced choice | 36.35% | **39.30%** | L (both < chance) |
| Answer-token loss | **0.9408** | 1.4351 | H |
| Off-prompt answer rate | **0.0–7.2%** | 19.2–28.8% | H |

The two models have **identical** non-embedding parameter counts by design, so
every difference above is attributable to data and tokenizer rather than to
capacity.

The perplexity/BPB disagreement is the project's central measurement lesson and
is unchanged from Phase 2: Model L looks much worse on perplexity (46.68 vs
27.41) and slightly *better* on bits-per-byte (0.4846 vs 0.5382), because its
tokenizer packs 11.44 bytes into a token against Hindi's 8.88. Most of the
apparent gap is a tokenizer artefact, not a modelling one.

On reasoning, the two metrics also disagree — Model H wins on exact match, Model
L nominally on forced choice — but since both forced-choice figures are below
chance, that ordering carries no signal. The difference that does carry signal
is §4.3: **Model H attends to the prompt and Model L frequently does not.**

### 6.1 Key Phase 1–2 figures, consolidated

The four questions in §7 are answered from evidence gathered across all three
phases. The figures that evidence rests on are collected here so this report can
be read on its own.

| Phase | Figure | What it establishes |
|---|---|---|
| 1 | [Corpus size](../phase1/figures/01_corpus_size.png) | The resource gap in raw terms — the premise of the whole comparison |
| 1 | [Cleaning funnel](../phase1/figures/03_cleaning_funnel.png) · [drop reasons](../phase1/figures/04_drop_reasons.png) | What each filter removed, and why retention differs by language |
| 1 | [Split sizes](../phase1/figures/05_split_sizes.png) | Document-level train/val/test partition for both languages |
| 1 | [Vocabulary fertility](../phase1/figures/07_vocab_fertility.png) · [chars per token](../phase1/figures/06_vocab_chars_per_token.png) | The bytes/token difference that makes perplexity and BPB disagree (§7.3) |
| 1 | [Byte fallback](../phase1/figures/08_vocab_byte_fallback.png) | The coverage gap behind the Nepali numeral defect (§1.4) |
| 2 | [Pretraining loss curves](../phase2/figures/loss_curves_both.png) | Both models trained to convergence on equal token budgets |
| 2 | [Generation metrics](../phase2/figures/generation_metrics_both.png) | BLEU / chrF++ / ROUGE-L and the greedy-decoding degeneration |
| 2 | [Attention summary](../phase2/figures/attention_summary_both.png) | The pretrained entropy/distance baseline that §5 measures the shift against |

The Phase 1 and Phase 2 reports remain the authoritative write-ups for those
phases: [`report/phase1/`](../phase1/) · [`report/phase2/phase2_report.md`](../phase2/phase2_report.md).

---

## 7. The four required questions

### 7.1 How did data scale and quality differ?

By raw token count the two corpora are comparable, and Nepali is nominally
larger — 521.4M vs 486.0M tokens on the train split. Scale is therefore *not*
the axis on which these models differ, which is itself the useful finding: the
project holds token budget roughly constant and varies language, tokenizer and
source quality.

Quality differs in composition. Hindi's full corpus came from 704,101 documents
with 30.66% of tokens from manual collection; Nepali's from 1,918,265 documents
at 26.41% manual — so Nepali's documents are much shorter on average and drawn
more heavily from web crawl. Nepali also
carries measurable residue from that provenance: Phase 1 reported a
`test_unk_rate` of 0.0 for Nepali but the measured value is 0.0019, because its
BPE has no byte fallback and silently drops characters it cannot represent —
Latin characters in mixed text, and, as Phase 3 discovered, **every Devanagari
digit**.

### 7.2 How do language-modeling and reasoning results compare across tiers?

They point in opposite directions, and the resolution is instructive.

On language modelling, the tiers appear far apart on perplexity (27.41 vs 46.68)
and essentially equal — slightly favouring Nepali — on bits-per-byte (0.5382 vs
0.4846). Bits-per-byte is the comparable measure across different tokenizers,
and it says the two models compress their languages about equally well.

On reasoning, both models fail the actual task equally: forced choice is at or
below chance for both. The tiers separate only on *format acquisition* — Model H
reaches 35.15% exact match and 0.9408 answer loss against Model L's 29.45% and
1.4351 — and on grounding, where Model H is three to ten times less likely to
answer off-prompt.

So: the low-resource model is not worse at modelling its language, is equally
unable to reason, and is distinctly worse at attending to the input it is given.

### 7.3 What tokenizer / corpus factors most affected the lower-resource model?

Three, in order of how much they actually cost.

1. **No byte fallback.** This is the most consequential single decision in the
   whole project. Model L's BPE cannot represent characters absent from its
   training corpus, and unrepresentable input becomes `<unk>` rather than
   degrading gracefully. Phase 3 found the sharpest case: the vocabulary
   contains **no Devanagari digit whatsoever**, so a Nepali sentence written
   with numerals ०–९ — entirely natural Nepali — tokenizes to holes. Had this
   not been caught, 4.8% of the finetuning corpus would have been `<unk>`.

2. **Coarser tokenization.** 11.44 bytes/token against Hindi's 8.88. Each token
   covers more text, so the same 512-token context holds ~31% more Nepali, but
   each prediction is correspondingly harder — which is most of the perplexity
   gap and almost none of a real quality gap.

3. **Shorter, web-heavier documents.** On the train split, Nepali packs
   521,426,359 tokens into 1,520,000 documents — **343.0 tokens per document** —
   against Hindi's 485,956,678 tokens in 675,937 documents, **718.9 per
   document**, a factor of 2.1. Less long-range structure per document is a
   plausible contributor to Model L's weaker prompt grounding in §4.3, though
   this project does not isolate it.

### 7.4 What evidence explains the observed differences?

* **Forced choice against a computed chance rate** is what shows the models did
  not learn to reason. Exact match alone would have read as +35 points of
  improvement.
* **Held-out entity names** are what show memorisation: both models answer test
  questions with training names, Model L four to ten times more often.
* **Position-level error analysis** is what shows the two-hop failure is
  systematic — 72.6% middle-entity selection against a 0.0% gold rate — rather
  than random.
* **Bits-per-byte alongside perplexity** is what shows the Model H / Model L gap
  is mostly tokenizer fertility.
* **Direct vocabulary inspection** is what shows the Devanagari-digit gap; no
  aggregate metric surfaced it.
* **Identical non-embedding parameter counts** are what license attributing any
  of this to data and tokenizer rather than to model size.

---

## 8. Deliverables

| # | Deliverable | Where |
|---|---|---|
| 1 | Reasoning data generation scripts, finetune scripts, configs | [`hindi/reasoning/`](../../hindi/reasoning/) · [`nepali/reasoning/`](../../nepali/reasoning/) · [`finetune.yaml`](../../hindi/configs/finetune.yaml) ×2 |
| 2 | Finetuned checkpoints (H and L) | Google Drive links in the [README](../../README.md) |
| 3 | Finetuning logs, metrics, qualitative examples | [`*_finetune_log.jsonl`](.) · [`*_reasoning_*.json`](.) · §3, §4 |
| 4 | Pretrain vs. finetune attention comparison | §5 · [`figures/`](figures/) · [`*_attention_comparison.json`](.) |
| 5 | Final comparison tables and error analysis | §3, §4, §6 |
| 6 | Complete final report and reproduction instructions | this document · §9 · [README](../../README.md) |

---

## 9. Reproduction

```bash
# 1. Regenerate both reasoning corpora (deterministic, seeded)
python hindi/reasoning/generate.py  --train 20000 --val 2000 --test 2000
python nepali/reasoning/generate.py --train 20000 --val 2000 --test 2000

# 2. Score the pretrained baselines FIRST - once weights are updated this
#    number cannot be recovered without retraining
python -m hindi.reasoning.evaluate_reasoning \
    --checkpoint checkpoints/hindi/ckpt_best.pt --tag pretrained
python -m nepali.reasoning.evaluate_reasoning \
    --checkpoint checkpoints/nepali/ckpt_best.pt --tag pretrained

# 3. Finetune each model on its own language only
python -m hindi.reasoning.finetune  --config hindi/configs/finetune.yaml
python -m nepali.reasoning.finetune --config nepali/configs/finetune.yaml

# 4. Score the finetuned checkpoints through the same code path
python -m hindi.reasoning.evaluate_reasoning \
    --checkpoint checkpoints/hindi_reasoning/ckpt_best.pt --tag finetuned
python -m nepali.reasoning.evaluate_reasoning \
    --checkpoint checkpoints/nepali_reasoning/ckpt_best.pt --tag finetuned

# 5. Attention comparison and all figures
python scripts/make_phase3_figures.py \
    --hindi-pretrained  checkpoints/hindi/ckpt_best.pt \
    --hindi-finetuned   checkpoints/hindi_reasoning/ckpt_best.pt \
    --nepali-pretrained checkpoints/nepali/ckpt_best.pt \
    --nepali-finetuned  checkpoints/nepali_reasoning/ckpt_best.pt
```

On Kaggle, one kernel per model runs steps 2–5 end to end:

```bash
python scripts/kaggle_push.py --what code
python scripts/kaggle_run.py --lang hindi  --mode finetune
python scripts/kaggle_run.py --lang nepali --mode finetune
python scripts/collect_phase3_results.py --lang hindi nepali --summary
```

Both model test suites pass 9/9 (`python hindi/tests/test_model.py`,
`python nepali/tests/test_model.py`), including the causal-masking check that
perturbing tokens after position *t* changes logits at positions ≤ *t* by
exactly 0.000e+00.

---

## 10. Limitations and what I would do next

Stated so the results above are not over-read.

1. **The models do not perform the task.** Forced choice is at or below chance
   for both. Everything in §5 describes a model that learned an output format.

2. **40 training entity names is too few.** It makes memorising the answer prior
   easier than learning to copy from the prompt, and both models took that path
   — validation loss turns upward by step 60. The most promising single change
   would be 200+ training names, which would make the prior much less
   rewarding. This was not run for want of time, not because it was judged
   unnecessary.

3. **The positional habits in §4.1–4.2 are diagnosed, not fixed.** Balancing
   the training distribution against answer position would test whether the
   two-hop failure is a data artefact or a capacity limit.

4. **The attention analysis is correlational.** It shows heads that moved; it
   does not establish that those heads carry the comparison. Ablating L3H1 or
   L3H3 and re-scoring would be the direct test.

5. **`transitive3` is the only multi-hop family.** Four-entity chains would show
   whether the middle-entity bias is specifically about the repeated name or
   about chain length generally.

6. **Single seed per model.** The Model H / Model L differences in §6 are not
   bracketed by run-to-run variance.



## report/phase3/README.md

# Phase 3 artifacts — index

Everything Phase 3 produced, and where each required deliverable lives. The main
write-up is **[phase3_report.md](phase3_report.md)**.

## Deliverables

| # | Required deliverable | Where |
|---|---|---|
| 1 | Reasoning data generation scripts, finetune scripts, configs | [`hindi/reasoning/`](../../hindi/reasoning/) · [`nepali/reasoning/`](../../nepali/reasoning/) · [`hindi/configs/finetune.yaml`](../../hindi/configs/finetune.yaml) · [`nepali/configs/finetune.yaml`](../../nepali/configs/finetune.yaml) |
| 2 | Finetuned checkpoints (H and L) via Drive links | [main README → Checkpoints](../../README.md#checkpoints-google-drive) |
| 3 | Finetuning logs, metrics, qualitative examples | [`hindi_finetune_log.jsonl`](hindi_finetune_log.jsonl) · [`nepali_finetune_log.jsonl`](nepali_finetune_log.jsonl) · the four `*_reasoning_*.json` files · [report §3–§4](phase3_report.md) |
| 4 | Pretrain vs. finetune attention comparison | [report §5](phase3_report.md) · [`figures/`](figures/) · [`hindi_attention_comparison.json`](hindi_attention_comparison.json) · [`nepali_attention_comparison.json`](nepali_attention_comparison.json) |
| 5 | Final comparison tables and error analysis | [report §3, §4, §6](phase3_report.md) |
| 6 | Complete final report and reproduction instructions | [phase3_report.md](phase3_report.md) · [§9 Reproduction](phase3_report.md#9-reproduction) |

## Files

### Metric reports

| File | What it holds |
|---|---|
| `hindi_reasoning_pretrained.json` | Model H scored on the reasoning test set **before** finetuning |
| `hindi_reasoning_finetuned.json` | Model H after finetuning, same code path |
| `nepali_reasoning_pretrained.json` | Model L before finetuning |
| `nepali_reasoning_finetuned.json` | Model L after finetuning |

Each contains exact-match and forced-choice accuracy (overall, by task family, by
hop count, by attribute), the computed chance rate, answer-token loss and
perplexity, tokenized sequence statistics, the `error_analysis` block that
records which prompt position each answer came from, and worked qualitative
examples of both successes and failures.

### Attention

| File | What it holds |
|---|---|
| `hindi_attention_comparison.json` | Per-head entropy and mean distance for Model H, pretrained and finetuned, on identical prompts, plus per-head deltas ranked by magnitude |
| `nepali_attention_comparison.json` | The same for Model L |

### Training

| File | What it holds |
|---|---|
| `hindi_finetune_log.jsonl` | Model H's finetuning log — train loss, validation loss every 20 steps, learning rate |
| `nepali_finetune_log.jsonl` | Model L's finetuning log |
| `hindi_finetune_config.yaml` | The exact resolved config the run used |
| `nepali_finetune_config.yaml` | The same for Model L |

### Figures

| File | What it shows |
|---|---|
| `figures/finetune_loss_curves_both.png` | Both models' finetuning loss and validation perplexity on shared axes. The overfitting is the point: validation turns upward at step 60 for both. |
| `figures/hindi_pretrained_reasoning_layer0.png` | Model H, early layer, 4 heads, **before** finetuning |
| `figures/hindi_finetuned_reasoning_layer0.png` | Model H, early layer, **after** |
| `figures/hindi_pretrained_reasoning_layer5.png` | Model H, late layer, before |
| `figures/hindi_finetuned_reasoning_layer5.png` | Model H, late layer, after |
| `figures/nepali_*_reasoning_layer{0,5}.png` | The same four for Model L |
| `figures/hindi_attention_pretrained_vs_finetuned.png` | Per-layer entropy and distance summary, both checkpoints |
| `figures/nepali_attention_pretrained_vs_finetuned.png` | The same for Model L |

All heatmaps use Devanagari token labels rather than position indices, and show
query position against key position with a probability colour scale. Every panel
is strictly lower-triangular, which is the visual form of the causal mask.

### Superseded runs

`run_a_40names/` holds the first finetuning run, which selected its checkpoint on
a 100-step evaluation grid. It is kept because the report refers to it: moving to
a 20-step grid, with nothing else changed, raised Model L from 14.90% to 29.45%
exact match. The conclusions did not change, but the numbers did, and the coarse
grid understated both models.

## Reading order

1. **[phase3_report.md §0](phase3_report.md)** — what was and was not learned, in
   five paragraphs.
2. **§3** — the pretrained vs. finetuned tables, and why two metrics are needed.
3. **§4** — the error analysis. This is where the result is actually explained.
4. **§7** — the four questions the specification asks.



## hindi/reasoning/data/dataset_stats.json

```json
   1  {
   2    "language": "hindi",
   3    "splits": {
   4      "train": {
   5        "examples": 20000,
   6        "by_task": {
   7          "transitive3": 5000,
   8          "numeric2": 5000,
   9          "numeric3": 5000,
  10          "compare2": 5000
  11        },
  12        "by_attribute": {
  13          "speed": 2033,
  14          "height": 5239,
  15          "weight": 5357,
  16          "age": 5365,
  17          "wealth": 2006
  18        },
  19        "distinct_templates": 14,
  20        "distinct_answers": 40,
  21        "distinct_prompts": 20000,
  22        "mean_prompt_chars": 71.1
  23      },
  24      "val": {
  25        "examples": 2000,
  26        "by_task": {
  27          "compare2": 500,
  28          "numeric2": 500,
  29          "numeric3": 500,
  30          "transitive3": 500
  31        },
  32        "by_attribute": {
  33          "age": 548,
  34          "speed": 196,
  35          "weight": 550,
  36          "wealth": 208,
  37          "height": 498
  38        },
  39        "distinct_templates": 4,
  40        "distinct_answers": 10,
  41        "distinct_prompts": 2000,
  42        "mean_prompt_chars": 72.6
  43      },
  44      "test": {
  45        "examples": 2000,
  46        "by_task": {
  47          "numeric3": 500,
  48          "transitive3": 500,
  49          "compare2": 500,
  50          "numeric2": 500
  51        },
  52        "by_attribute": {
  53          "age": 527,
  54          "weight": 539,
  55          "wealth": 201,
  56          "height": 531,
  57          "speed": 202
  58        },
  59        "distinct_templates": 4,
  60        "distinct_answers": 10,
  61        "distinct_prompts": 2000,
  62        "mean_prompt_chars": 76.9
  63      }
  64    },
  65    "leakage_check": {
  66      "entity_names_disjoint_across_splits": true,
  67      "templates_disjoint_across_splits": true,
  68      "train_names": 40,
  69      "val_names": 10,
  70      "test_names": 10
  71    }
  72  }
```


## nepali/reasoning/data/dataset_stats.json

```json
   1  {
   2    "language": "nepali",
   3    "splits": {
   4      "train": {
   5        "examples": 20000,
   6        "by_task": {
   7          "compare2": 5000,
   8          "transitive3": 5000,
   9          "numeric2": 5000,
  10          "numeric3": 5000
  11        },
  12        "by_attribute": {
  13          "height": 5377,
  14          "wealth": 2057,
  15          "speed": 1962,
  16          "age": 5348,
  17          "weight": 5256
  18        },
  19        "distinct_templates": 14,
  20        "distinct_answers": 40,
  21        "distinct_prompts": 20000,
  22        "mean_prompt_chars": 74.5
  23      },
  24      "val": {
  25        "examples": 2000,
  26        "by_task": {
  27          "compare2": 500,
  28          "transitive3": 500,
  29          "numeric3": 500,
  30          "numeric2": 500
  31        },
  32        "by_attribute": {
  33          "age": 535,
  34          "height": 531,
  35          "weight": 547,
  36          "speed": 202,
  37          "wealth": 185
  38        },
  39        "distinct_templates": 4,
  40        "distinct_answers": 10,
  41        "distinct_prompts": 2000,
  42        "mean_prompt_chars": 74.2
  43      },
  44      "test": {
  45        "examples": 2000,
  46        "by_task": {
  47          "numeric2": 500,
  48          "compare2": 500,
  49          "transitive3": 500,
  50          "numeric3": 500
  51        },
  52        "by_attribute": {
  53          "age": 526,
  54          "speed": 219,
  55          "height": 524,
  56          "weight": 518,
  57          "wealth": 213
  58        },
  59        "distinct_templates": 4,
  60        "distinct_answers": 10,
  61        "distinct_prompts": 2000,
  62        "mean_prompt_chars": 79.9
  63      }
  64    },
  65    "leakage_check": {
  66      "entity_names_disjoint_across_splits": true,
  67      "templates_disjoint_across_splits": true,
  68      "train_names": 40,
  69      "val_names": 10,
  70      "test_names": 10
  71    }
  72  }
```

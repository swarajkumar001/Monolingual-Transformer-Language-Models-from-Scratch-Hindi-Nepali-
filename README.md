# Language Models and Agents — Individual Project

This is my individual project for the Language Models and Agents course. The task
was to build two language models from scratch, completely separately: one for a
higher-resource Indian language and one for a lower-resource one, each around 25M
parameters, each trained on its own data with its own tokenizer.

I picked **Hindi as Model H** and **Nepali as Model L**.

I picked them partly because both are written in Devanagari. That way, if the two
models end up performing differently, the script cannot be the reason, so
whatever is left has to come from the data and the tokenizer. That turned out to
matter more than I expected.

Nothing is shared between the two models. Separate corpora, separate tokenizers,
separate vocabularies, separate weights. `hindi/` and `nepali/` are two Python
packages and neither one imports the other. Every command I run names one
language.

All three phases are done.

## Where things are

| | Link |
|---|---|
| Whole project in one PDF (9 pages) | [report/final/final_report.pdf](report/final/final_report.pdf) |
| Phase 3 report (the consolidated one the brief asks for) | [report/phase3/phase3_report.md](report/phase3/phase3_report.md) |
| Phase 2 report | [report/phase2/phase2_report.md](report/phase2/phase2_report.md) |
| Phase 1 report | [report/phase1/phase1_report.md](report/phase1/phase1_report.md) ([PDF](report/phase1/phase1_report.pdf)) |
| File-by-file index per phase | [phase 1](report/phase1/README.md) · [phase 2](report/phase2/README.md) · [phase 3](report/phase3/README.md) |
| Checkpoints (too big for git) | [Google Drive](https://drive.google.com/drive/folders/15kYy6N9YaCx1wk2JMg8k8udf1szQfneT?usp=sharing) |
| Corpora (too big for git) | [OneDrive](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/swaraj_kumar_students_iiit_ac_in/IgCh3u8O2sr4QIJqmrUS1CllAV1C5HL0R9SOWUUVIzTgcdw?e=bgUHaG) |

## Both models, side by side

| | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| Documents | 704,101 | 1,918,265 |
| Corpus tokens | 505,852,028 | 635,204,333 |
| Against the 500M target | 101.2% | 127.0% |
| Collected by hand | 30.66% | 26.41% |
| Tokenizer | SentencePiece Unigram, 8,000 | SentencePiece BPE, 10,000 |
| Bytes per token | 8.88 | 11.44 |
| Unknown-token rate (test) | 0.000000 | 0.001894 |
| Total parameters | 23,273,472 | 24,297,472 |
| Parameters minus embeddings | 18,915,328 | 18,915,328 |
| Layers / heads / width / FF / context | 6 / 8 / 512 / 2048 / 512 | same |
| Steps and tokens trained | 14,520 and 475,791,360 | 14,440 and 473,169,920 |
| Training time (Tesla T4) | 11.0 h at 12.0k tokens/s | 11.0 h at 12.0k tokens/s |
| Validation perplexity | 27.07 | 46.14 |
| Test perplexity | 27.41 | 46.68 |
| Test bits-per-byte | 0.5382 | 0.4846 |
| Mean attention entropy | 3.205 | 3.090 |
| Reasoning set (train/val/test) | 20,000 / 2,000 / 2,000 | 20,000 / 2,000 / 2,000 |
| Exact match, before to after | 0.00% to 35.15% | 0.00% to 29.45% |
| Forced choice (chance 41.67%) | 39.80% to 36.35% | 38.80% to 39.30% |
| Answer-token loss | 5.2439 to 0.9408 | 5.5367 to 1.4351 |
| Answered with a name not in the prompt | 0.0–7.2% | 19.2–28.8% |

![Training and validation loss for both models against tokens consumed](report/phase2/figures/loss_curves_both.png)

## The two things I actually learned from this

**My two main metrics contradicted each other, and that turned out to be the
interesting part.** Perplexity says Hindi is 41% better. Bits-per-byte says
Nepali is slightly better. I spent a while assuming one of them had to be wrong
before realising they measure different units. Perplexity is per token, and a
token is not the same amount of text in the two models: Nepali's BPE holds 11.44
bytes per token where Hindi's Unigram holds 8.88. So Nepali's tokens are harder
to guess individually, but each one carries more text. Dividing by bytes instead
of tokens takes the tokenizer out of the comparison. If I had only reported
perplexity I would have written that the low-resource model is much worse, and
that would have been wrong. Most of the gap is the tokenizer, not the model.
(Details in the [Phase 2 report](report/phase2/phase2_report.md), section 4.)

**In Phase 3 I nearly made the same mistake in reverse.** After finetuning, exact
match went from 0.00% to 35.15% for Hindi and 0.00% to 29.45% for Nepali, which
looks like the models learned to reason. They did not. When I instead force the
model to choose between the names that appear in the question, both sit at or
below the chance rate of 41.67%, at 36.35% and 39.30%. Task by task the scores
land almost exactly on chance: Hindi gets 50.40% and 47.00% on the two-answer
tasks (chance 50%) and 33.20% on the three-answer one (chance 33.3%). So what
finetuning taught was the answer format, not the comparison.

The two-hop questions are the one place the models leave chance, and they leave
it downwards: 12.40% and 13.40% against 33.3%. I dug into why. On "A is bigger
than B, B is bigger than C, who is biggest?", Hindi answers with the middle name
72.6% of the time and the first name 0.0% of the time, and the middle name is
never correct. It is picking the name that gets mentioned twice instead of
following the chain. That is a specific, measurable failure, which I would rather
report than hide. (Details in the [Phase 3 report](report/phase3/phase3_report.md),
sections 3 and 4.)

Both models also ended up with almost the same attention structure even though
they saw different languages, tokenizers and seeds. Both grew a previous-token
head at layer 3 (entropy 0.06 at distance 1.3 for Hindi, 0.14 at 1.1 for Nepali).
I did not expect that from 23M-parameter models.

![Per-head attention entropy and mean attention distance by layer](report/phase2/figures/attention_summary_both.png)

---

# Phase 1 — Collecting the data and training the tokenizers

I needed about 500M tokens per language, with at least 20% of the training tokens
collected by hand rather than downloaded.

## Where I got the text

For **Hindi**, the manual side is a Hindi Wikipedia XML dump that I downloaded and
extracted myself with WikiExtractor (my biggest manual source), Gadya Kosh for
prose, Kavita Kosh for poetry, the patrika.com news archive, and 67 archive.org
items including Rajya Sabha and Lok Sabha transcripts and some books. The
downloaded side is FineWeb-2 `hin_Deva` and AI4Bharat Sangraha. I deliberately
left out the HuggingFace copy of Hindi Wikipedia, because it is the same text as
the dump I had already extracted and I did not want to count it twice.

Before scraping anything I checked robots.txt for each site and recorded the
result in `scripts/manual_sites.json`. bhaskar.com and amarujala.com disallow
automated access, so I did not scrape them.

For **Nepali**, the manual side is mostly an onlinekhabar scrape (91.6% of my
manual tokens), plus a Wikipedia dump I parsed myself, BBC Nepali, Wikibooks,
Wiktionary, Ekantipur, Ratopati, Nagarik, and OCR of some archive.org books. The
downloaded side is AI4Bharat Sangraha and IndicCorpV2. For Nepali I set two fixed
targets instead of a percentage — at least 100M manual tokens and at least 400M
downloaded — and hit both, with 167.8M and 467.4M.

## Cleaning

Normalisation is four steps (`scripts/clean_lib.py`): NFC, strip zero-width and
control characters, collapse runs of spaces, collapse runs of blank lines. I keep
ZWNJ and ZWJ on purpose, because in Devanagari they change what actually gets
displayed — they are part of the text, not junk. NFC matters for a less obvious
reason too: the same Devanagari string can be stored two different ways, and if I
do not normalise first, those two forms look like different documents and both
survive deduplication.

Then I drop a document if it is under 200 characters, if less than half its
characters are Devanagari, if more than 1% of it is broken Unicode, or if the
language check says it is not the right language. I made the language check
three-valued — yes, no, or cannot tell — so a document does not get thrown out
just because the detector was unsure about it. Hindi also gets a Bhojpuri filter,
which I needed because Bhojpuri uses the same script and a lot of the same words.

## Deduplication (the two pipelines are not the same, and I should say so)

**Hindi** deduplicates with a SHA-1 hash of the normalised text. That catches
exact copies and copies that differ only in spacing or punctuation. It does not
catch two documents that say the same thing in different words.

**Nepali** does it properly, with MinHash and LSH: 100 permutations, 20 bands of
5 rows, 5-character shingles, Jaccard threshold 0.55. It removed 2,350 exact and
49,734 near duplicates. When a duplicate pair had one copy in my manual set and
one in the downloaded set (21,273 pairs), I keep the manual copy, so
deduplication does not quietly eat my manual token share. Nepali also has a
decontamination stage that checks train against validation and test and removes
matches from train only; it took out 17,111 documents and a rescan found nothing
left.

Hindi has neither of those stages. That is a genuine weakness in the Hindi
pipeline and it is the first thing I would go back and fix.

## Splits

Both are split by document, never by chunk. If two pieces of the same article end
up on opposite sides of the split, the test score is partly measuring
memorisation.

Hindi is 96/2/2, seed 42, shuffle then slice. Nepali is 85/7.5/7.5, seed
20260817, using a hash of each document id. Nepali holds out validation and test
*before* the quality filters run, so those filters only shape the training pool
and cannot make the test set easier than it should be. The split file records
zero document-id overlap between splits.

The ratios differ because Nepali has far more documents, so 7.5% is still a
sensible size and leaves over 519M tokens to train on.

## Choosing the tokenizers

I trained both from scratch and picked the vocabulary size by trying several and
measuring on held-out text, rather than guessing. Fertility is tokens per word,
so lower is better.

| Language | Vocab | Chars/token | Fertility | Unknown rate | Byte fallback |
|---|---|---:|---:|---:|---:|
| Hindi | 5,000 | 3.3230 | 1.4961 | 0.000000 | 0.018301 |
| Hindi | **8,000** | **3.5622** | **1.3956** | **0.000000** | **0.019613** |
| Hindi | 10,000 | 3.6601 | 1.3583 | 0.000000 | 0.020213 |
| Nepali | 5,000 | 3.7882 | 1.6988 | 0.002956 | 0.000000 |
| Nepali | 6,000 | 3.9410 | 1.6329 | 0.003075 | 0.000000 |
| Nepali | 8,000 | 4.1725 | 1.5424 | 0.003256 | 0.000000 |
| Nepali | **10,000** | **4.3410** | **1.4825** | **0.003387** | **0.000000** |

I went with Unigram 8,000 for Hindi. Fertility does keep improving at 10,000, but
the jump from 8,000 to 10,000 is smaller than the one from 5,000 to 8,000, and
those 2,000 extra rows would cost me about 1M parameters out of a 25M budget. For
Nepali I took BPE 10,000, which had the best fertility in the sweep.

**The mistake is in the last column and I did not catch it here.** Hindi's
tokenizer has byte fallback, so a character it has never seen still gets stored
as bytes and nothing is lost. Nepali's has no byte fallback at any vocabulary
size, so anything outside the vocabulary becomes `<unk>` and is gone for good. I
looked at that column of zeros and read it as a good score. It is not a score, it
is a missing feature, and it came back to bite me in Phase 3.

## Final token counts

| | Hindi | Nepali |
|---|---:|---:|
| Train | 485,280,741 | 519,906,359 |
| Validation | 10,484,640 | 57,839,396 |
| Test | 10,086,647 | 57,458,578 |
| **Total** | **505,852,028** | **635,204,333** |
| Manual tokens | 155,104,651 | 167,787,251 |
| Over the 20% manual requirement? | yes, 30.66% | yes, 26.41% |

These are exact. I tokenized and counted every document in every split instead of
sampling and scaling up.

One thing that confused me for a bit: Phase 2 reports slightly higher numbers for
the same splits (485,956,678 for Hindi train). The difference is exactly the
number of documents, because packing adds one end-of-document token per document.

## Checks

`scripts/validate_phase1.py` says ALL CHECKS PASSED for Hindi.
`scripts/validate_phase1_nepali.py` says the same for Nepali except the
cross-language check, which it marks SKIPPED on purpose — at roughly 9.2GB it is
too heavy to run as a side effect of a validator. I ran it separately with
`scripts/check_nepali_hindi_overlap.py` and got 0 shared documents.

## What to look at for Phase 1

| # | Deliverable | Where |
|---|---|---|
| 1 | Collection scripts | `hindi/data/pipeline/{download_hindi,ingest_manual,ingest_wikipedia_dump}.py`; `nepali/scripts/{s01_collect,collect,auto_grow_and_tokenize}.py`, `corpuslib/` |
| 2 | Preprocessing | `hindi/data/pipeline/clean_hindi.py`, `scripts/clean_lib.py`; `nepali/scripts/s03` to `s11` |
| 3 | Dataset statistics | [phase1_report.md](report/phase1/phase1_report.md), [dataset_statistics.md](report/phase1/dataset_statistics.md), [tokenizer_analysis.md](report/phase1/tokenizer_analysis.md) |
| 4 | Splits | `hindi/data/splits/` with manifests; `nepali/data/final/{train,val,test}.jsonl` with `split_stats.json` |
| 5 | Tokenizer code | `hindi/tokenizer/train_tokenizer.py`; `nepali/scripts/{s12_train_tokenizer,s13_tokenizer_eval}.py` |
| 6 | Vocabularies | `hindi/tokenizer/hindi_unigram_8000.vocab`; `nepali/tokenizer/nepali_bpe.vocab` and `sweep/` |
| 7 | Tokenizer models | `hindi/tokenizer/hindi_unigram_8000.model`; `nepali/tokenizer/nepali_bpe.model` |

## Running it again

```
python -m venv .venv
.venv/Scripts/activate      # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Hindi:

```
python hindi/data/pipeline/download_hindi.py
python hindi/data/pipeline/ingest_manual.py
python hindi/data/pipeline/ingest_wikipedia_dump.py
python hindi/data/pipeline/clean_hindi.py           # raw -> interim -> cleaned
python hindi/data/pipeline/split_hindi.py
python hindi/tokenizer/train_tokenizer.py --vocab-size 8000
```

`clean_hindi.py` starts over by default; `--resume` continues an interrupted run.
Both it and `split_hindi.py` take a lock file, so if a run got killed, delete a
`.clean.lock` or `.split.lock` older than 15 minutes.

Nepali runs as numbered stages:

```
cd nepali/scripts
python s00_doctor.py                # sanity-checks the environment and config
python auto_grow_and_tokenize.py    # S02 to S14, growing whichever lane is short
```

Individual stages run with `python run_pipeline.py <stage>`. S08 is the slow one
(MinHash over about 2.2M documents) and takes hours.

Then:

```
python scripts/validate_phase1.py
python scripts/validate_phase1_nepali.py
python scripts/check_nepali_hindi_overlap.py
python report/phase1/make_plots.py
```

---

# Phase 2 — Writing the Transformer and pretraining it

## What I built

A decoder-only Transformer using only `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`,
`nn.Dropout` and ordinary tensor operations. No `nn.Transformer`, no
`nn.MultiheadAttention`, no `F.scaled_dot_product_attention`, no HuggingFace
model class. I wrote a test that walks the module tree and fails if any of those
show up, so I cannot break this rule by accident later.

Both models are 6 layers, 8 heads, width 512, feed-forward 2048, context 512,
dropout 0.1, pre-norm, GELU, with tied input and output embeddings and learned
absolute positions.

Keeping the architecture identical was deliberate. It means the non-embedding
parameter count is exactly the same for both (18,915,328), so nobody can explain
away a difference in results by saying one model was bigger.

Inside attention I kept W_q, W_k and W_v as three separate `nn.Linear` layers
instead of one fused matrix. It is slightly slower but the three roles are
visible in the code, and I wanted to be able to point at them. Each projection
takes (B, T, 512), a `view` makes it (B, T, 8, 64) and a `transpose` makes it
(B, 8, T, 64), which is 8 independent 64-wide heads. Scores are QKᵀ divided by
√64 = 8. The scaling is there because dot products of longer vectors get bigger
on average, and without it the softmax saturates and gradients through it shrink.

A few decisions I had to justify to myself:

- **6 layers instead of a wider 4-layer model.** Each block is about 3.15M
  parameters, so 6 blocks plus embeddings gives 23.3M and 7 would overshoot to
  26.4M. I spent the budget on depth because attention needs a few rounds to
  build up multi-step relationships.
- **Learned absolute positions.** I train and test at one fixed length, so the
  extrapolation advantage of sinusoidal encodings does not help me. The price is
  that 512 is a hard wall: there is no embedding row for position 512.
- **Tied embeddings**, which saves 8000 × 512 = 4,096,000 parameters for Hindi,
  around 15% of the model.

## Proving the model cannot see the future

This is the part I am most confident about. The test in
`hindi/tests/test_model.py` changes every token after position *t* and then
checks whether anything at positions up to *t* moved. The biggest change it finds
is **exactly 0.000e+00**. Not small — exactly zero, because `softmax(-inf)` is
exactly 0 and 0 × v is exactly 0. The test also checks the other direction, that
the outputs at the changed positions *do* move, otherwise a model that ignored
its input entirely would pass and I would never know. Both test files pass 9/9.

## Training

AdamW with betas 0.9 and 0.95, weight decay 0.1 on matrices only (decaying
LayerNorm gains and biases just shrinks them for no benefit), peak learning rate
6e-4 with 500 warmup steps then cosine decay to 6e-5, gradient clipping at 1.0
after unscaling, effective batch 32,768 tokens per step.

Every checkpoint stores weights, optimizer state, scheduler state, step, best
validation loss, tokens seen, both RNG states and the config, so a killed session
resumes exactly rather than approximately. I tested that across three sessions
before starting the real runs, because the whole thing runs on Kaggle and
sessions do die.

**Both runs stopped short.** I put an 11-hour wall-clock limit in the trainer so
a session could never die without leaving a usable checkpoint, and it fired at
step 14,520 for Hindi and 14,440 for Nepali out of a planned 15,000. Checkpoints
save on best validation loss, so what I evaluated is step 14,500 and step 14,000.
It cost me about 3% of the token budget and both loss curves were still going
down, so neither model is converged. I would rather say that than pretend the
runs finished.

## Results

Scored over 1,999,872 tokens per split for each model.

| | Hindi | Nepali |
|---|---:|---:|
| Test cross-entropy (nats) | 3.3109 | 3.8433 |
| Test perplexity | 27.41 | 46.68 |
| Bits per token | 4.7766 | 5.5447 |
| Bytes per token | 8.875 | 11.441 |
| Test bits-per-byte | 0.5382 | 0.4846 |

Generation, at greedy and three temperatures:

| Model | Decoding | BLEU-4 | chrF++ | ROUGE-L | Repetition | Distinct-1 |
|---|---|---:|---:|---:|---:|---:|
| H | greedy | 1.96 | 12.49 | 0.093 | 0.598 | 0.172 |
| H | T = 1.0 | 1.01 | 16.29 | 0.100 | 0.002 | 0.520 |
| L | greedy | 2.37 | 14.48 | 0.081 | 0.577 | 0.179 |
| L | T = 1.0 | 1.69 | 18.05 | 0.079 | 0.001 | 0.634 |

Greedy decoding falls apart in both models — around 58–60% of the 4-grams it
produces are repeats and you can watch the text loop. That is not a bug, it
follows from what greedy decoding is: it always takes the most likely token, so
the moment the model re-enters a state it has been in, it produces the same
continuation and has no way out. Temperature 1.0 kills the repetition and gives
the best chrF++ at the same time, so on these models more variety cost me nothing
in quality.

BLEU sits near the floor everywhere and I do not trust it here. Free-form
continuations have no particular reason to share 4-grams with one reference
continuation. chrF++ is the one I would defend, because it works on character
n-grams, and in a language where changing one suffix breaks a whole-word match
that matters. All four decoding settings are in the report and the eval JSONs.

## Attention

Mean entropy is 3.2047 for Hindi and 3.0898 for Nepali, against a uniform
baseline of 5.2565. I compute that baseline because otherwise "entropy 3.2" means
nothing on its own. Mean distance is 62.36 and 54.51 tokens.

Going layer by layer, the sharpest head gets steadily more local from layer 0 to
layer 3, where it turns into a previous-token head, then spreads out again by
layer 5. Both models did this on their own, which is the result I found most
surprising in this phase.

## What to look at for Phase 2

| # | Deliverable | Where |
|---|---|---|
| 1 | Transformer (attention, positions, causal mask) | [hindi/model/transformer.py](hindi/model/transformer.py), [nepali/model/transformer.py](nepali/model/transformer.py) |
| 2 | Configs | [hindi/configs/model.yaml](hindi/configs/model.yaml), [nepali/configs/model.yaml](nepali/configs/model.yaml) |
| 3 | Parameter counts, training scripts, checkpoints | table above, [hindi/train/train.py](hindi/train/train.py), [nepali/train/train.py](nepali/train/train.py), [Drive](https://drive.google.com/drive/folders/15kYy6N9YaCx1wk2JMg8k8udf1szQfneT?usp=sharing) |
| 4 | Training logs and loss curves | `report/phase2/*_training_log.jsonl`, [loss_curves_both.png](report/phase2/figures/loss_curves_both.png) |
| 5 | Perplexity, BPB, BLEU, chrF, ROUGE-L | [report sections 4–5](report/phase2/phase2_report.md), `report/phase2/*_eval.json` |
| 6 | Samples, repetition, diversity | [report section 5.2](report/phase2/phase2_report.md), `*_eval.json` under `generation.samples` |
| 7 | Attention heatmaps, entropy, distance | [report/phase2/figures/](report/phase2/figures/), `report/phase2/*_attention.json` |
| 8 | Comparison across the two resource levels | [report section 7](report/phase2/phase2_report.md) |

## Running it again

```
python hindi/data/pack_corpus.py
python nepali/data/pack_corpus.py

python hindi/tests/test_model.py            # 9/9, including the causal mask test
python nepali/tests/test_model.py

python -m hindi.train.train  --config hindi/configs/model.yaml
python -m nepali.train.train --config nepali/configs/model.yaml

python -m hindi.eval.evaluate  --checkpoint checkpoints/hindi/ckpt_best.pt  --split test
python -m nepali.eval.evaluate --checkpoint checkpoints/nepali/ckpt_best.pt --split test

python scripts/make_phase2_figures.py \
    --hindi-checkpoint checkpoints/hindi/ckpt_best.pt \
    --nepali-checkpoint checkpoints/nepali/ckpt_best.pt
```

I ran pretraining on Kaggle with a T4 using
[notebooks/kaggle_pretrain.ipynb](notebooks/kaggle_pretrain.ipynb), once per
language, and scored finished checkpoints with
[notebooks/kaggle_evaluate.ipynb](notebooks/kaggle_evaluate.ipynb).

---

# Phase 3 — Finetuning on reasoning questions

## The data I generated

I wrote a generator for each language instead of downloading a benchmark, so I
know the right answer to every question by construction. The Hindi and Nepali
sets share no sentence, no name and no template.

Four question types: a one-step comparison, a two-step chain, and numeric
comparison over two or three entities. 20,000 training, 2,000 validation and
2,000 test examples per language.

To stop the model from passing by memorising, I used both leakage defences at
once: the 40/10/10 entity names and the 14/4/4 sentence templates do not overlap
across splits. So a test question is about people the model has never seen,
phrased in a way it has never seen. I assert this in the generator rather than
assuming it.

**I found two bugs while building this, and either one would have made my results
meaningless.**

The first was a gender agreement leak. Hindi adjectives inflect, so a question
like "who is the smallest?" asked about one woman and two men only has one
grammatically possible answer — you can get it right without comparing anything.
181 out of 400 examples were solvable that way. I fixed it by drawing all
entities in an example from the same gender.

The second is the byte fallback problem coming back. The Nepali vocabulary
contains **no Devanagari digit at all**, not one of ०–९, and with no byte fallback
every number in my generated data turned into `<unk>` — 4.8% of the corpus. I
switched Nepali numbers to Arabic digits, which the vocabulary covers completely
and which modern Nepali writing uses anyway. Both sets now tokenize at 0.000000
unknown. This is the clearest example in the project of a Phase 1 decision
staying invisible for two phases and then causing a real problem.

## How I finetuned

Each model starts from its own pretrained checkpoint and keeps its own tokenizer.
The finetuning script asserts that the tokenizer vocabulary size matches the
checkpoint instead of trusting the config file, because that is exactly the kind
of mismatch you do not notice until the results are garbage.

Loss is on answer tokens only (`ignore_index=-1` on the prompt), so the model
gets no credit for memorising question text. I group batches by length instead of
padding them, which is not cosmetic: with learned absolute positions, padding at
the front would shift the position index of every real token.

Settings: 2 epochs, batch 32, peak learning rate 6e-5 (a tenth of pretraining,
because a pretraining-sized step wipes out the language model before the task is
learned), 100 warmup steps, evaluation every 20 steps, seed 20260914.

**I had to redo this run.** My first attempt evaluated every 100 steps and picked
its best checkpoint at step 100, which is exactly where warmup ends — the grid
was too coarse to find the real optimum. Moving to every 20 steps, with nothing
else changed, took Nepali from 14.90% to 29.45% exact match and Hindi from 32.70%
to 35.15%. The first run is still in `report/phase3/run_a_40names/`.

## Results

2,000 held-out test questions per model, unseen names and unseen templates.

| Model | Measure | Before | After | Change |
|---|---|---:|---:|---:|
| H | Exact match | 0.00% | 35.15% | +35.15 |
| H | Forced choice | 39.80% | 36.35% | -3.45 |
| H | Answer-token loss | 5.2439 | 0.9408 | -4.3031 |
| L | Exact match | 0.00% | 29.45% | +29.45 |
| L | Forced choice | 38.80% | 39.30% | +0.50 |
| L | Answer-token loss | 5.5367 | 1.4351 | -4.1016 |
| | chance rate, forced choice | 41.67% | 41.67% | |

I report two accuracy numbers because neither is enough alone. Exact match after
greedy decoding is the honest end-to-end result, but it is stuck at zero for a
model that has never seen this prompt format, so it cannot tell "learned nothing"
apart from "knows the answer but not how to say it". The forced choice between
the names in the question fixes that, with the chance rate computed per example
rather than assumed. It is the number that shows what was actually learned, and
it is the less flattering one.

Both models overfit fast. Validation loss bottoms out at step 60 and rises from
there while training loss keeps dropping to 0.44. With 40 names and 14 templates,
memorising which answer tends to be right is easier than learning to read the
question. Saving on best validation loss is the only reason the evaluated
checkpoints are usable at all — they are step 60, not the badly overfit end of
the run.

![Finetuning training and validation loss for both models](report/phase3/figures/finetune_loss_curves_both.png)

## Attention before and after

I ran both checkpoints on the same held-out reasoning prompts, all the same
token length, otherwise the text or the length could look like a finetuning
effect when it is not.

| Model | Checkpoint | Mean entropy | Mean distance | Uniform baseline |
|---|---|---:|---:|---:|
| H | pretrained | 1.9102 | 10.303 | 2.9545 (length 46) |
| H | finetuned | 1.8691 | 10.360 | |
| L | pretrained | 1.8327 | 9.415 | 2.8061 (length 39) |
| L | finetuned | 1.7833 | 10.157 | |

Finetuning made attention sharper and longer-range at the same time in both
models, most obviously in Nepali, where mean distance goes up 0.74 positions on a
39-token prompt. The heads that moved most are in the middle layers, which is
where Phase 2 found the previous-token heads. Layer 0 and layer 5 heatmaps for
both checkpoints and both models are in
[report/phase3/figures/](report/phase3/figures/) with Devanagari labels.

To be clear about what this does and does not show: it tells me which heads
changed, not that those heads are doing the comparison. Testing that properly
would mean disabling a head and re-scoring, and I did not get to it.

## What to look at for Phase 3

| # | Deliverable | Where |
|---|---|---|
| 1 | Generation and finetuning scripts and configs | [hindi/reasoning/](hindi/reasoning/), [nepali/reasoning/](nepali/reasoning/), [hindi/configs/finetune.yaml](hindi/configs/finetune.yaml), [nepali/configs/finetune.yaml](nepali/configs/finetune.yaml) |
| 2 | Finetuned checkpoints | [Drive](https://drive.google.com/drive/folders/15kYy6N9YaCx1wk2JMg8k8udf1szQfneT?usp=sharing), table below |
| 3 | Logs, metrics, examples | `report/phase3/*_finetune_log.jsonl`, `report/phase3/*_reasoning_*.json`, [report sections 3–4](report/phase3/phase3_report.md) |
| 4 | Attention before vs after | [report section 5](report/phase3/phase3_report.md), [report/phase3/figures/](report/phase3/figures/) |
| 5 | Comparison tables and error analysis | [report sections 3, 4 and 6](report/phase3/phase3_report.md) |
| 6 | Final report and reproduction steps | [phase3_report.md](report/phase3/phase3_report.md), [final_report.pdf](report/final/final_report.pdf) |

## Running it again

```
python hindi/reasoning/generate.py  --train 20000 --val 2000 --test 2000
python nepali/reasoning/generate.py --train 20000 --val 2000 --test 2000

# Score the pretrained model FIRST. Once the weights change this number is gone
# unless I retrain.
python -m hindi.reasoning.evaluate_reasoning --checkpoint checkpoints/hindi/ckpt_best.pt --tag pretrained

python -m hindi.reasoning.finetune --config hindi/configs/finetune.yaml

python -m hindi.reasoning.evaluate_reasoning --checkpoint checkpoints/hindi_reasoning/ckpt_best.pt --tag finetuned
```

Full steps including the Kaggle path and the figures are in
[the Phase 3 report, section 9](report/phase3/phase3_report.md#9-reproduction).

---

# Checkpoints

[LMA CHECKPOINT — shared Drive folder](https://drive.google.com/drive/folders/15kYy6N9YaCx1wk2JMg8k8udf1szQfneT?usp=sharing)

| Phase | Model | File | Size | Contents |
|---|---|---|---:|---|
| 2, pretrained | H | `model_H_hindi_checkpoints.zip` | 1.28 GB | `ckpt_best.pt` (step 14,500), `ckpt_final.pt`, `latest.pt`, steps 14,000 and 14,500 |
| 2, pretrained | L | `model_L_nepali_checkpoints.zip` | 1.33 GB | `ckpt_best.pt` (step 14,000), `ckpt_final.pt`, `latest.pt`, steps 13,500 and 14,000 |
| 3, finetuned | H | `model_H_hindi_reasoning_finetuned.zip` | 0.77 GB | `ckpt_best.pt` (step 60), `ckpt_final.pt`, `latest.pt`, config, log |
| 3, finetuned | L | `model_L_nepali_reasoning_finetuned.zip` | 0.80 GB | `ckpt_best.pt` (step 60), `ckpt_final.pt`, `latest.pt`, config, log |

Every one of them carries weights, optimizer state, scheduler state, step, best
validation loss and the config, so training resumes exactly. The pretraining ones
also carry tokens consumed and both RNG states. The finetuned ones keep the same
keys as the pretrained ones, which is what let me load them into the Phase 2
evaluation and attention code without changing a line of it.

# How the repository is laid out

```
<repo>/
  README.md
  requirements.txt
  hindi/                       # Model H, self-contained
    data/
      raw/{manual,downloaded}/ # source text plus metadata.jsonl
      interim/ cleaned/        # cleaning stages
      splits/                  # train/validation/test, by document
      packed/                  # token streams (not in git; the .json files
                               # with exact counts are)
      pipeline/                # Phase 1 download, ingest, clean, split
      dataset.py               # Phase 2 batch sampling
      pack_corpus.py           # splits -> token streams
    tokenizer/                 # SentencePiece model and vocabulary
    model/                     # config.py, transformer.py
    train/                     # train.py
    eval/                      # evaluate.py, attention.py, viz.py
    reasoning/                 # Phase 3: generate.py, dataset.py, finetune.py,
                               # evaluate_reasoning.py, attention_compare.py
      data/                    # train/val/test.jsonl, dataset_stats.json
    tests/                     # test_model.py including the causal mask test
    configs/                   # model.yaml, finetune.yaml
  nepali/                      # Model L, same shape, separate copy. Phase 1 is
                               # different: stages S00–S14 under scripts/
  notebooks/                   # kaggle_pretrain, kaggle_evaluate, kaggle_finetune
  scripts/                     # shared helpers across all three phases
  report/
    phase1/ phase2/ phase3/    # reports, results, configs, figures
    final/                     # final_report.tex and final_report.pdf
  logs/                        # pipeline logs (not in git)
```

# Submission checklist

The ten things the brief asks for, and where each one is.

| # | Required | Where |
|---|---|---|
| 1 | Dataset and preprocessing code, both languages | [hindi/data/pipeline/](hindi/data/pipeline/), [nepali/scripts/](nepali/scripts/), [scripts/clean_lib.py](scripts/clean_lib.py) |
| 2 | Separate tokenizers and vocabularies | [hindi/tokenizer/](hindi/tokenizer/) Unigram 8,000, [nepali/tokenizer/](nepali/tokenizer/) BPE 10,000 |
| 3 | Transformer implementation | [hindi/model/transformer.py](hindi/model/transformer.py), [nepali/model/transformer.py](nepali/model/transformer.py) |
| 4 | Drive links, pretrained checkpoints | [checkpoints table](#checkpoints), rows "2, pretrained" |
| 5 | Drive links, finetuned checkpoints | [checkpoints table](#checkpoints), rows "3, finetuned" |
| 6 | Training and finetuning logs | `report/phase2/*_training_log.jsonl`, `report/phase3/*_finetune_log.jsonl` |
| 7 | Evaluation scripts and results | [hindi/eval/](hindi/eval/), [nepali/eval/](nepali/eval/), `*/reasoning/evaluate_reasoning.py`, `report/phase2/*_eval.json`, `report/phase3/*_reasoning_*.json` |
| 8 | Attention visualisations | [report/phase2/figures/](report/phase2/figures/), [report/phase3/figures/](report/phase3/figures/) |
| 9 | Final report | [report/phase3/phase3_report.md](report/phase3/phase3_report.md), [report/final/final_report.pdf](report/final/final_report.pdf) |
| 10 | README with reproduction steps and Drive links | this file |

# Constraints I had to respect

No pretrained models, no pretrained tokenizers, no HuggingFace Transformer
classes. I wrote the positional embeddings, multi-head attention and causal
masking myself out of basic PyTorch layers, and there is a test that fails if a
pre-built attention block ever appears in the module tree.

The two corpora, tokenizers, vocabularies and weights are fully separate, and I
never trained a multilingual model. Phase 3 keeps that separation: the reasoning
data was generated per language and shares no sentence, name or template, and
each model was finetuned from its own checkpoint with its own tokenizer.

# Where the project stands

All three phases are finished and pushed to `phase-1`, `phase-2` and `phase-3`.
`phase-3` is the complete project and the branch I am submitting.

I also did the optional bonus ablation, on a branch called `bonus-no-pos`. I
retrained Hindi from scratch with positional embeddings removed and put it
through the whole Phase 2 evaluation. Removing positions cost 5.8% test
perplexity (27.41 to 29.01), which is much less damage than I expected — if
position information were really gone, a model that cannot tell word order should
do far worse. Attention entropy rose from 3.2047 to 3.6005 against the same 5.2565
baseline, and the previous-token head reappeared one layer later than usual.

My explanation is that causal masking already leaks some position information: a
token at position *t* can only attend to *t* earlier tokens, so how many tokens
are visible tells you roughly where you are. That fits what I measured, but this
experiment does not prove it — proving it needs a probe I did not run. I should
also note the ablated run got a faster GPU and finished all 15,000 steps, so it
was actually the better-trained of the two, and no timing comparison between them
would be fair. Setup, results and limitations are in `report/bonus/README.md` on
that branch. I am not claiming any ablation results on this branch.

# About Bhojpuri

Bhojpuri was my original Model L. I built the whole pipeline for it — collected,
cleaned, split, tokenized, and it passed every check in `validate_phase1.py` —
but public Bhojpuri text ran out at 84,217,032 tokens, which is 16.8% of the
target. The brief allows a documented shortfall, so I could have kept it, but
once the Nepali pipeline cleared both of its token targets I switched Model L to
Nepali rather than submit a corpus a sixth of the required size.

The Bhojpuri pipeline code and tokenizer files are gone now, and `bhojpuri/data/`
was never in git anyway. The numbers I measured are still in
[phase1_report.md](report/phase1/phase1_report.md) section 6 as a record of what
I did.

The lesson I took from it is to check how much text actually exists for a
language before building a pipeline around it, not after.

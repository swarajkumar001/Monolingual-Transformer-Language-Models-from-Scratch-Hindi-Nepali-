# Viva Preparation Guide

**Project:** Two from-scratch monolingual Transformer LMs — Hindi (Model H) and Nepali (Model L)

**Typeset version:** [`docs/VIVA_GUIDE.pdf`](docs/VIVA_GUIDE.pdf) (51 pages, colour-coded answer boxes, TikZ diagrams) — built from [`docs/VIVA_GUIDE.tex`](docs/VIVA_GUIDE.tex) with `bash docs/build_viva_guide.sh`. This Markdown file is the quick-search version; the two carry the same content.

**Graded report:** `report/final/final_report.pdf` (10 pages). Every number in all three documents is checked against the result artifacts by `report/final/check_report_facts.py`, which fails if any of them disagree.

How to use this guide: read Part A and Part M first, then work through C–E. Parts I, J and K are the ones examiners actually spend time on. Do not memorise — the "Think of it like this" boxes exist so you can rebuild an answer if you forget the wording.

---

## PART A — 2-Minute Project Introduction

### 30-second version

> I built two separate decoder-only Transformers from scratch in PyTorch, one for Hindi and one for Nepali. Nothing is shared between them except the source code — separate corpora, separate tokenizers, separate weights. Both are about 23 million parameters. The interesting result is that Hindi looks 41% better on perplexity, but that advantage mostly disappears when you measure in bits-per-byte, because the two tokenizers chop text into different-sized pieces.

### 1-minute version

> The task was to build two monolingual language models of about 25M parameters each, with no pretrained weights and no pretrained tokenizers. I picked Hindi as the higher-resource language and Nepali as the lower-resource one, and I picked that pair deliberately because both use Devanagari — if the script is the same and the architecture is the same, then any difference between the models has to come from the data or the tokenizer, not from the writing system or model size.
>
> Phase 1 was collecting and cleaning the corpora and training the tokenizers: 506 million Hindi tokens and 635 million Nepali, both with over 20% manually collected. Phase 2 was implementing the Transformer myself — attention, causal masking, positional embeddings — and pretraining both models for about 475 million tokens on a Kaggle T4. Phase 3 was generating synthetic comparative-reasoning data in each language and finetuning on it.
>
> The headline results are that Hindi gets test perplexity 27.41 and Nepali 46.68, but on bits-per-byte it is 0.5382 against 0.4846 — the ranking flips. And in Phase 3, exact match went from 0% to 35%, but when I forced the models to choose between the names in the question they were at chance, so they learned the answer format rather than the comparison.

### 2-minute version

Say the 1-minute version, then add:

> The part I would most want to talk about is that this project gave me the same trap twice, in opposite directions. In Phase 2, if I had only reported perplexity I would have concluded the low-resource model was far worse — bits-per-byte showed most of that gap was the tokenizer. In Phase 3, if I had only reported exact match I would have concluded the models learned to reason — forced choice showed they had not. Both times a second metric measuring the same thing in different units is what caught it.
>
> I also traced two Phase 1 decisions forward. Nepali's tokenizer has no byte fallback, which showed up as a column of zeros that I misread as a good score; two phases later it meant the Nepali vocabulary had no Devanagari digit at all, so 4.8% of my generated reasoning data turned into unknown tokens. And Hindi only deduplicates on exact SHA-1 hashes while Nepali does MinHash near-duplicate removal plus train/test decontamination — which is a real weakness in the Hindi pipeline and a limit on how strongly I can interpret the comparison.

**KEY POINT:** Lead with the controlled-comparison design (same script, same architecture). That is the one sentence that makes the whole project sound deliberate rather than assembled.

---

## PART B — Complete Project Pipeline

```
        ┌──────────────────────────┐
        │  Raw corpora             │  H: 773,606 docs   L: 2,655,504 docs
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │  Cleaning & Unicode      │  NFC, strip zero-width/control,
        │  normalisation           │  collapse whitespace
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │  Filtering               │  length ≥200, Devanagari ≥0.5,
        │                          │  malformed ≤1%, language ID
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │  Deduplication /         │  H: SHA-1 exact only
        │  decontamination         │  L: MinHash+LSH + decontamination
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │  Train / val / test      │  H: 96/2/2   L: 85/7.5/7.5
        │  split (by document)     │  by document, never by chunk
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │  Tokenizer training      │  H: Unigram 8,000
        │                          │  L: BPE 10,000
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │  Pack to token stream    │  uint16, +1 EOS per document
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │  Transformer LM          │  6 layers, 8 heads, d=512,
        │  (written from scratch)  │  d_ff=2048, context 512
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │  Pretraining             │  AdamW, cosine, ~475M tokens,
        │                          │  11 h on a Kaggle T4
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │  LM evaluation           │  PPL 27.41 / 46.68
        │                          │  BPB 0.5382 / 0.4846
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │  Generation + attention  │  BLEU/chrF++/ROUGE-L,
        │  analysis                │  entropy, distance
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │  Synthetic reasoning     │  20k/2k/2k, 4 task families,
        │  dataset                 │  disjoint names + templates
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │  Finetuning              │  from each model's OWN checkpoint,
        │                          │  answer-token loss only
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │  Reasoning evaluation    │  exact match + forced choice
        └────────────┬─────────────┘
                     ↓
        ┌──────────────────────────┐
        │  Error + attention       │  answer-position analysis,
        │  analysis                │  pretrained vs finetuned
        └──────────────────────────┘
```

### Every stage: WHAT / WHY / HOW / OUTPUT

| Stage | WHAT | WHY | HOW | OUTPUT |
|---|---|---|---|---|
| Collection | Gather raw text per language | Need ~500M tokens, ≥20% manual | Scraping, Wikipedia dumps, OCR, public corpora | 773,606 + 2,655,504 raw docs |
| Normalisation | Make text canonical | Same string can be stored two ways; unnormalised text defeats dedup | `normalize_text()` in `scripts/clean_lib.py` — 4 operations | Byte-canonical text |
| Filtering | Drop junk | Short/wrong-language/corrupt docs teach the model nothing | Length, `devanagari_ratio()`, `malformed_ratio()`, `langdetect_verdict()` | H: 704,103 docs kept |
| Deduplication | Remove repeats | Duplicates inflate apparent data and leak into test | H: `dedup_key()` SHA-1; L: `MinHasher` + LSH | H: 2,983 removed; L: 2,350 + 49,734 |
| Splitting | Train/val/test | Need held-out data to measure generalisation | Document-level, seeded | H 96/2/2, L 85/7.5/7.5 |
| Tokenizer | Text → integer IDs | Model works on IDs, not characters | SentencePiece trained on train split only | Unigram 8k, BPE 10k |
| Packing | Flatten to token stream | Training reads contiguous windows | `CorpusPacker.pack_split()`, uint16 + EOS | 485,956,678 / 521,426,359 train tokens |
| Transformer | The model | The thing being built | `GPT` in `hindi/model/transformer.py` | 23.27M / 24.30M params |
| Pretraining | Learn the language | Next-token prediction | `Trainer.train()`, AdamW + cosine | ~475M tokens each |
| LM evaluation | Measure quality | PPL/BPB on held-out text | `Evaluator.perplexity_and_bpb()` | PPL 27.41 / 46.68 |
| Generation | Produce text | Check fluency, not just loss | `GPT.generate()`, greedy + T∈{0.5,1,1.5} | chrF++ peaks 16.29 / 18.05 |
| Attention analysis | Inspect heads | I implemented attention, so I should inspect it | `summarise_model_attention()` | entropy 3.2047 / 3.0898 |
| Reasoning data | Synthetic task | Need known ground truth and controlled leakage | `ReasoningGenerator` | 20k/2k/2k per language |
| Finetuning | Adapt to the task | Teach the task without destroying the LM | `FineTuner.train()`, LR 6e-5 | best val at step 60 |
| Reasoning eval | Measure it | Two metrics because one is not enough | `ReasoningEvaluator` | EM 35.15/29.45%, FC at chance |
| Error analysis | Explain failures | Accuracy alone hides the mechanism | `position_analysis()` | middle-entity bias 72.6% |

---

## PART C — Phase 1 Viva

### C1. Why these two languages?

**SHORT ANSWER:** Hindi as the higher-resource language because it has abundant public text, and Nepali from the allowed lower-resource list. I chose that specific pair because both are written in Devanagari — if the script is held constant and I also hold the architecture constant, then any gap between the models has to come from data or tokenizer, not from the writing system or capacity.

**IF EXAMINER ASKS MORE:** Bhojpuri was my original Model L. I built the complete pipeline for it and it passed validation, but public Bhojpuri text ran out at 84,217,032 tokens — only 16.8% of the 500M target. The brief allows a documented shortfall, but once the Nepali pipeline cleared its targets I switched rather than submit a corpus one-sixth of the required size.

**COMMON FOLLOW-UP:** *"Isn't it a problem that they share a script?"* No — it is the point. If they used different scripts I could never separate "script is harder" from "data is worse."

### C2. Manual vs downloaded data

**SHORT ANSWER:** The brief required at least 20% of final training tokens from manual collection. Hindi came out at 30.66% and Nepali at 26.41%.

| | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| Manual tokens | 155,104,651 (30.66%) | 167,787,251 (26.41%) |
| Downloaded tokens | 350,747,377 | 467,417,082 |
| **Total** | **505,852,028** | **635,204,333** |

**Hindi manual sources:** self-extracted Wikipedia XML dump (66.7M), Gadya Kosh prose (59.9M), patrika.com archive (19.5M), Kavita Kosh poetry (8.1M), 15 surviving archive.org items (0.85M).
**Hindi downloaded:** FineWeb-2 `hin_Deva`, four AI4Bharat Sangraha partitions.
**Nepali manual:** onlinekhabar scrape (91.6% of the manual lane), self-parsed Wikipedia dump, BBC Nepali, Wikibooks, Wiktionary, Ekantipur, Ratopati, Nagarik, OCR of archive.org books.
**Nepali downloaded:** AI4Bharat Sangraha, IndicCorpV2.

**IF ASKED MORE:** I also quarantined the HuggingFace mirror of Hindi Wikipedia, because it is the same text as the dump I had already extracted — counting both would have double-counted. And I checked `robots.txt` per site before scraping and recorded the verdict in `scripts/manual_sites.json`; bhaskar.com and amarujala.com disallow automated access, so I did not scrape them.

**⚠ LIKELY CHALLENGE:** *"Is downloading a Wikipedia XML dump really 'manual collection'?"* Be honest: **"I counted it as manual because I downloaded the raw dump and ran the extraction and cleaning myself, rather than pulling a prepared corpus. I can see the argument that it is closer to a download. If you exclude it, my Hindi manual share drops from 30.66% to 17.47%, which would be below the floor — so it is the single assumption my Phase 1 compliance rests on."** Saying this before they say it is much stronger than being caught by it.

### C3. Cleaning and normalisation

```
RAW TEXT
   ↓  NFC normalise
   ↓  strip zero-width + C0/C1 control characters
   ↓  collapse inline whitespace runs
   ↓  collapse blank-line runs
   ↓  strip
NORMALISED TEXT
   ↓  length filter (<200 chars → drop)
   ↓  Devanagari ratio (<0.5 → drop)
   ↓  malformed ratio (U+FFFD >1% → drop)
   ↓  language ID (three-valued)
   ↓  [Hindi only] Bhojpuri lexical reject
CLEAN TEXT
```

All of this is `normalize_text()` and friends in `scripts/clean_lib.py`.

**Tiny example:**

```
Raw:        "नमस्ते   दुनिया\n\n\n"   (with a stray U+200E)
            ↓ NFC
            ↓ strip U+200E
            ↓ collapse "   " → " "
            ↓ collapse "\n\n\n" → "\n\n"
Normalised: "नमस्ते दुनिया"
```

What changed: the invisible left-to-right mark is gone (it carries no meaning here and would make two otherwise identical documents hash differently), the triple space became one, trailing blank lines collapsed.

### C4. Why NFC and not NFKC?

**SHORT ANSWER:** NFC is canonical composition — it picks one byte representation for strings that are semantically identical. NFKC goes further and does *compatibility* mappings, which actually change characters. I only want to canonicalise, not to rewrite the text.

**IF ASKED MORE:** NFKC would, for example, fold ligatures and change some presentation forms. In Devanagari that risks destroying information I want the model to learn. NFC is the safe choice: same visible text, one canonical encoding.

**WHY IT MATTERS HERE (the good answer):** The same Devanagari string can be stored two different ways. If I do not normalise before hashing, those two forms produce different hashes, both survive deduplication, and I end up with duplicates I think I removed. So NFC is not cosmetic — it is a *precondition for deduplication working at all*.

**KEY POINT:** NFC = same text, one encoding. NFKC = possibly different text. I needed the first.

### C5. Zero-width characters, ZWNJ and ZWJ

**SHORT ANSWER:** I strip U+200E, U+200F, U+FEFF and the C0/C1 control range, but I deliberately **keep** ZWNJ (U+200C) and ZWJ (U+200D).

**IF ASKED MORE:** In Devanagari, ZWNJ and ZWJ change what is actually rendered — they control whether consonants form a conjunct ligature or stay separate. Removing them would silently change the word. The others (direction marks, byte-order mark) carry no linguistic content in this corpus and only create spurious hash differences.

**Think of it like this:** a BOM is packaging; a ZWNJ is spelling.

**COMMON FOLLOW-UP:** *"How do you know you kept the right ones?"* The split is explicit in the stripping regex in `clean_lib.py`, and it was a deliberate decision rather than a default.

### C6. Language identification

**SHORT ANSWER:** `langdetect_verdict()` returns **three** values — True, False, or `None` for "cannot tell" — with a minimum probability of 0.85. A document is only dropped on a confident False, not on uncertainty.

**IF ASKED MORE:** Treating "undetermined" as "wrong language" would throw away short or unusual-but-valid documents. Nepali uses a fastText model (`lid.176.bin`) with its own protections in `nepali/scripts/corpuslib/langid.py`: it calls the raw binding directly, self-tests on known Hindi/Nepali/English strings at construction, and raises if the error rate crosses a threshold.

**⚠ WORTH VOLUNTEERING:** An earlier Nepali run was destroyed by this exact area. The `fasttext` wheel calls `np.array(probs, copy=False)`, which raises on NumPy ≥2; every `predict()` threw, the caller swallowed it as "not our language", and millions of documents were deleted while the stage logged SUCCESS. That is why the current code has a startup self-test and counts errors separately from rejections. This is a strong story — it shows you debugged a silent failure.

### C7. Devanagari ratio filter

**SHORT ANSWER:** `devanagari_ratio()` computes the fraction of characters in the Devanagari Unicode block; documents below 0.5 are dropped. For Hindi that removed 228 documents.

**IF ASKED MORE:** It catches pages that are mostly English, mostly navigation chrome, or mostly numbers. 0.5 is deliberately permissive — real Hindi text contains Latin-script names, numerals and punctuation, so a stricter threshold would drop valid documents.

### C8. Deduplication — exact vs near (⚠ know this cold)

**The two pipelines are different, and you must say so.**

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| Method | SHA-1 over normalised text (`dedup_key()`) | exact `doc_id`, then MinHash + LSH |
| Catches | exact copies, whitespace/punctuation variants | the above **plus** near duplicates |
| Removed | 2,983 (0.4%) | 2,350 exact + 49,734 near |
| Decontamination | **none** | 17,111 train docs removed, rescan clean |

**SHORT ANSWER:** Hindi deduplicates with a SHA-1 hash of the normalised, punctuation-stripped, case-folded text. That catches exact copies and formatting variants but by construction cannot catch two documents that say the same thing in different words. Nepali uses MinHash with LSH and additionally decontaminates train against validation and test.

**⚠ DO NOT SAY** "I used MinHash" as a project-wide statement. It is false for Hindi and an examiner who opens `scripts/clean_lib.py` will find out.

**THE HONEST ADMISSION (prepare it):** *"Hindi having no near-duplicate removal and no decontamination is a real weakness. It means Hindi's test set may overlap its training data in ways Nepali's cannot, and since the direction of that bias flatters Hindi, part of the perplexity gap I attribute to tokenizer granularity could be unmeasured leakage. I did not quantify it. It is the first thing I would fix."*

### C9. Shingles, Jaccard, MinHash and LSH

Work through this example until you can reproduce it.

```
Document A: "the cat is sitting on the mat"
Document B: "the cat is sitting on a mat"
```

Using **word** shingles of size 3 (my code uses 5-*character* shingles; words are easier to show):

```
A: {the cat is, cat is sitting, is sitting on, sitting on the, on the mat}
B: {the cat is, cat is sitting, is sitting on, sitting on a,   on a mat}

Intersection = 3   {the cat is, cat is sitting, is sitting on}
Union        = 7
Jaccard      = 3/7 = 0.43
```

**1. Why exact hashing is not enough.** Change one word and the hash changes completely. A and B above have *different* SHA-1 hashes but are 43% similar — a near duplicate that exact matching silently keeps.

**2. What Jaccard means.** |A ∩ B| / |A ∪ B| — the fraction of distinct pieces the two documents share. 0 = nothing in common, 1 = identical sets.

**3. Why MinHash.** Comparing Jaccard directly means storing every shingle of every document and intersecting sets — far too slow for 2.2 million documents. MinHash replaces each document with a short fixed-length signature (mine is **100** numbers) with a useful property: **the probability that two signatures agree in any one position equals their Jaccard similarity.** So I can estimate similarity by comparing 100 integers instead of thousands of shingles.

**4. What LSH does.** Even with signatures, comparing all pairs is O(n²) — about 2.4 × 10¹² pairs at my scale. LSH splits each 100-number signature into **20 bands of 5 rows**, hashes each band, and only compares documents that collide in *at least one* band. Similar documents collide with high probability; dissimilar ones almost never do. It turns an all-pairs problem into a bucket lookup.

**Think of it like this:** MinHash is a fingerprint. LSH is filing the fingerprints into drawers so you only compare the ones already in the same drawer.

**My actual configuration** (`nepali/scripts/corpuslib/dedup.py`): `DEFAULT_NUM_PERM = 100`, `DEFAULT_BANDS = 20`, `DEFAULT_ROWS = 5`, `DEFAULT_SHINGLE = 5` (characters), Jaccard threshold **0.55**, seed 20260817. The relevant classes are `MinHasher` (`.signature()`), `build_lsh_buckets()`, `remove_near_duplicates()` and `find_contaminated()`.

**COMMON FOLLOW-UP:** *"Why 20 bands of 5?"* Bands × rows must equal the number of permutations (20 × 5 = 100). The split sets the sharpness of the threshold: more bands makes the filter more permissive (catches more, more false positives), fewer bands makes it stricter. 20 × 5 pairs with a 0.55 Jaccard threshold.

**COMMON FOLLOW-UP:** *"Why threshold 0.55 and not 0.8?"* 0.55 is aggressive — it removes documents that merely overlap substantially. For a web-scraped corpus with heavy boilerplate repetition that is what I wanted; the cost is some false positives, which at 2.2M documents I can afford.

### C10. Cross-lane duplicate resolution (a good detail to volunteer)

When a duplicate pair had one copy in the manual lane and one in the downloaded lane — **21,273 pairs** — the resolver keeps the **manual** copy. Otherwise deduplication would quietly eat the manual token share I was required to hit.

### C11. Splits and why document-level

**SHORT ANSWER:** Both corpora are split **by document, never by chunk**. If you split by chunk, two halves of the same article can land on opposite sides, and your test score is partly measuring memorisation.

| | Model H | Model L |
|---|---|---|
| Ratio | 96 / 2 / 2 | 85 / 7.5 / 7.5 |
| Seed | 42 | 20260817 |
| Method | shuffle then slice | deterministic hash of `doc_id` |
| Held out before quality filtering | no | **yes** (stage S07) |
| Train tokens | 485,280,741 | 519,906,359 |
| Val tokens | 10,484,640 | 57,839,396 |
| Test tokens | 10,086,647 | 57,458,578 |

**IF ASKED MORE:** Nepali holds out validation and test at stage S07, *before* the quality filters in S05/S06 run. That means the filters only shape the training pool and cannot make the evaluation sets artificially easy. `split_stats.json` records `zero_doc_id_overlap: true`. A separate check (`scripts/check_nepali_hindi_overlap.py`) found 0 documents shared between the two languages' corpora.

**COMMON FOLLOW-UP:** *"Why different ratios?"* Nepali has far more documents, so 7.5% is still a very large evaluation set (57M tokens) while leaving 520M to train on. Hindi has fewer documents, so 2% is the right trade-off there. I would acknowledge this is an inconsistency between the two pipelines.

### C12. Tokenizers — BPE and Unigram

**What is BPE?** Start from characters, repeatedly find the most frequent adjacent pair, and merge it into a new symbol. The vocabulary is the sequence of merges. It is greedy and bottom-up.

**What is Unigram?** Start from a large candidate vocabulary and *prune* it: assume tokens are drawn independently, and iteratively remove the tokens whose removal costs the least likelihood. It is probabilistic and top-down, and at encoding time it picks the segmentation with the highest probability.

**Why did I use each?** I chose them per language from a measured sweep rather than by preference. Unigram won for Hindi, BPE for Nepali, on fertility and unknown rate on held-out text.

**The pipeline:**

```
raw text → normalise → SentencePiece → token IDs → embedding lookup
"नमस्ते"                                [3012, 877]      (2, 512) vectors
```

**Small example of the concept:**

```
"internationalization"  →  ["inter", "nation", "al", "ization"]
```

One word, four tokens. A different tokenizer might give three or six. **This is why perplexity is not comparable across tokenizers** — see Part H.

### C13. Vocabulary size, fertility, unknown rate, byte fallback

The measured sweep (`report/phase1/measured_tokenizer_metrics.json`):

| Language | Vocab | Chars/token | Fertility | Unk rate | Byte fallback |
|---|---:|---:|---:|---:|---:|
| Hindi | 5,000 | 3.3230 | 1.4961 | 0.000000 | 0.018301 |
| Hindi | **8,000** | **3.5622** | **1.3956** | **0.000000** | **0.019613** |
| Hindi | 10,000 | 3.6601 | 1.3583 | 0.000000 | 0.020213 |
| Nepali | 5,000 | 3.7882 | 1.6988 | 0.002956 | 0.000000 |
| Nepali | 6,000 | 3.9410 | 1.6329 | 0.003075 | 0.000000 |
| Nepali | 8,000 | 4.1725 | 1.5424 | 0.003256 | 0.000000 |
| Nepali | **10,000** | **4.3410** | **1.4825** | **0.003387** | **0.000000** |

**Fertility** = tokens per word. Lower is better: it means each token covers more text, so a fixed context window holds more content and the model takes fewer steps per sentence.

**Unknown-token rate** = fraction of tokens that come out as `<unk>`. Every `<unk>` is text the model cannot even represent, let alone predict.

**Byte fallback** = when the tokenizer meets something not in its vocabulary, it encodes the raw UTF-8 bytes instead of emitting `<unk>`. Nothing is ever lost.

**Why vocabulary 8,000 for Hindi?** Fertility keeps improving with size, so the choice is a trade-off against embedding parameters. Going 8,000 → 10,000 buys less fertility than 5,000 → 8,000 did, and costs about 1M parameters out of a 25M budget. So I stopped at 8,000. Nepali took 10,000, the best fertility in its sweep.

**⚠ THE BYTE-FALLBACK STORY (know this — it is your best Phase 1 answer):**

> Nepali's byte-fallback column is 0.000000 at *every* vocabulary size. At the time I read a column of zeros as a good score. It is not a score — it records that the feature is **absent**. That is why Nepali's unknown rate is non-zero and Hindi's is exactly zero. And it came back in Phase 3: the Nepali vocabulary contains no Devanagari digit at all, not one of ०–९, so with no byte fallback every number in my generated reasoning data became `<unk>` — 4.8% of the corpus. I caught it by tokenizing the generated set and checking its unknown rate before training, and fixed it by switching Nepali numerals to Arabic digits, which the vocabulary covers completely and which modern Nepali writing uses anyway.

**COMMON FOLLOW-UP:** *"Why does unknown rate go UP with vocabulary size for Nepali?"* Larger vocabularies spend their budget on longer merged units, so rarer characters are less likely to survive as standalone tokens. With byte fallback this would be invisible; without it, it surfaces as `<unk>`.

### C14. Why separate tokenizers and no shared vocabulary?

**SHORT ANSWER:** The brief requires it — the two models must not share data, tokenizer, vocabulary or weights. But it is also methodologically right: a shared vocabulary would make this a multilingual model, and the whole point was two independent monolingual models.

**IF ASKED MORE:** A shared Devanagari vocabulary would have been technically easy since both languages use the same script, which is exactly why the rule exists. It would also have confounded the comparison — I could no longer attribute a difference to the tokenizer if both models used the same one.

### C15. Phase 1 — what I would improve

1. Run MinHash near-duplicate removal and decontamination on Hindi too.
2. Enable byte fallback on the Nepali tokenizer.
3. Use the same manual-share methodology for both (Hindi's is character-weighted, Nepali's token-weighted — not strictly like-for-like).
4. Check how much public text actually exists for a language *before* building a pipeline around it (the Bhojpuri lesson).

---

## PART D — Phase 2 Viva

### D1. Architecture at a glance

```
                  Input token IDs  (B, T),  T ≤ 512
                         │
                         ↓
                 Token embedding  (V, 512)        V = 8,000 (H) / 10,000 (L)
                         +
              Positional embedding  (512, 512)    learned, absolute
                         ↓
                 Embedding dropout 0.1
                         ↓
         ┌───────────────────────────────────┐
         │  Transformer block (pre-norm)     │
         │    LayerNorm                      │
         │      ↓                            │
         │    Multi-head causal self-attn    │   h = 8, d_k = 64
         │      ↓                            │
         │    + residual                     │
         │      ↓                            │
         │    LayerNorm                      │
         │      ↓                            │
         │    FFN 512 → 2048 → 512, GELU     │
         │      ↓                            │
         │    + residual                     │
         └───────────────┬───────────────────┘
                         │   × 6 layers
                         ↓
                  Final LayerNorm
                         ↓
                  LM head (512, V)   ← tied to token embedding
                         ↓
                  Logits (B, T, V)
                         ↓
                  Next-token distribution
```

Classes: `GPT`, `TransformerBlock`, `MultiHeadCausalSelfAttention`, `FeedForward`, `LayerNorm` — all in `hindi/model/transformer.py` (and the mirrored `nepali/model/transformer.py`).

### D2. Why decoder-only?

**SHORT ANSWER:** The task is causal language modelling — predict the next token given everything before it. A decoder-only stack with causal masking is exactly that. An encoder would let every position see every other position, which makes next-token prediction trivial and useless.

**COMMON FOLLOW-UP:** *"Why not encoder-decoder?"* That architecture is for mapping one sequence to another — translation, summarisation. I have one sequence and one objective.

### D3. Q, K, V — the analogy, then the maths

**Think of it like this:** it is a lookup in a library. Each token writes a **query** ("what am I looking for?"), every token also advertises a **key** ("what am I about?"), and carries a **value** ("here is my actual content"). You match your query against all keys to get a relevance score, turn those scores into weights, and take a weighted blend of the values.

**The formula:**

```
Attention(Q, K, V) = softmax( QKᵀ / √d_k  +  M ) V
```

| Symbol | Meaning | Shape in my model |
|---|---|---|
| X | input to the block | (B, T, 512) |
| Q = X·W_Q | queries | (B, 8, T, 64) |
| K = X·W_K | keys | (B, 8, T, 64) |
| V = X·W_V | values | (B, 8, T, 64) |
| QKᵀ | raw scores | (B, 8, T, T) |
| d_k | per-head dimension | 64 |
| M | additive causal mask | (1, 1, T, T) |
| output | after ×V and concat | (B, T, 512) |

**The reshape, in words:** each projection produces (B, T, 512). A `view` turns it into (B, T, 8, 64) and a `transpose` into (B, 8, T, 64) — eight independent 64-dimensional heads. After attention, transpose and reshape back to (B, T, 512) and apply W_O.

**Implementation detail worth volunteering:** I kept W_Q, W_K and W_V as **three separate `nn.Linear` layers** instead of one fused matrix. It is marginally slower, but the three roles stay visible in the code and I can point at them.

```
Q ──┐
    ├── QKᵀ ── ÷√d_k ── + mask ── softmax ── dropout ── × V ── concat ── W_O ──→ out
K ──┘                                                     │
V ────────────────────────────────────────────────────────┘
```

### D4. Why divide by √d_k?

**SHORT ANSWER:** Dot products of longer vectors grow in magnitude. With d_k = 64, if the components are roughly independent with unit variance, the dot product has variance about 64 and so standard deviation about 8. Feeding numbers that large into softmax saturates it — one entry goes to ~1 and the rest to ~0 — and the gradient through a saturated softmax is almost zero. Dividing by √64 = 8 brings the variance back to about 1.

**EXAMPLE:** scores `[2, 1, 0]` → softmax ≈ `[0.67, 0.24, 0.09]` (informative). Scale them up to `[16, 8, 0]` → softmax ≈ `[0.9997, 0.0003, ~0]` (saturated, gradient ≈ 0).

**KEY POINT:** Without the scaling the model does not fail loudly — it just trains badly, because gradients through attention vanish.

### D5. Causal masking (⚠ know this cold)

**Simple statement:** the model must not be allowed to look at the answer while predicting it.

For the sequence `A B C D`, the allowed attention pattern is:

```
        key:  A    B    C    D
query A:      ✓    ✗    ✗    ✗
query B:      ✓    ✓    ✗    ✗
query C:      ✓    ✓    ✓    ✗
query D:      ✓    ✓    ✓    ✓        ← lower triangular
```

and **not** every row full. When predicting what follows C, the model may use A, B and C, but not D — D is the thing being predicted.

**How it is implemented:** an upper-triangular mask built with `torch.triu(..., diagonal=1)`, registered as a non-persistent buffer, applied with `masked_fill(-inf)` **before** the softmax.

**Why before softmax, and why −∞?** Because softmax exponentiates: `exp(−∞) = 0` exactly, and softmax then renormalises over the remaining positions so the row still sums to 1. If you masked *after* softmax you would have to renormalise yourself, and the masked probability mass would already have been mixed in.

**Why exactly zero and not just small?** `softmax(−inf)` is exactly 0, and `0 × v = 0`. There is no leakage at all, not even at floating-point precision.

**How my test verifies it** — `test_causal_mask_blocks_the_future()` in `hindi/tests/test_model.py`:

1. Run the model, record the logits.
2. Perturb **every** token after position *t*.
3. Re-run and measure the largest change in the logits at positions ≤ *t*.
4. Result: **exactly 0.000e+00**.
5. The test also asserts the **converse** — that outputs at the perturbed positions *do* change.

**Why step 5 matters (say this):** without it, a model that ignored its input entirely would also pass the test. Checking only that nothing leaks is a one-sided test.

Both test suites pass 9/9.

### D6. Pre-norm vs post-norm

**SHORT ANSWER:** I used pre-norm — LayerNorm is applied to the input of each sublayer, before attention and before the FFN, rather than after the residual addition.

**IF ASKED MORE:** With pre-norm the residual path is a clean identity from input to output, so gradients flow to early layers without passing through a normalisation at every step. Post-norm needs careful warmup to stay stable in deep stacks. At 6 layers either would probably work; pre-norm is the safer default and is what modern GPT-style models use.

### D7. Feed-forward network and GELU

**SHORT ANSWER:** Two linear layers, 512 → 2048 → 512, with GELU in between. The inner dimension is 4× the model dimension, which is the standard ratio.

**Why have an FFN at all?** Attention mixes information *across positions* but is linear in the values. The FFN is where per-position non-linear transformation happens. Without it, stacking attention layers would collapse toward a linear operation and the model could not represent much.

**Why GELU over ReLU?** It is smooth, so it has a non-zero gradient for small negative inputs instead of ReLU's hard zero; it is the standard choice in GPT-style models.

### D8. Residual connections

**SHORT ANSWER:** Each sublayer computes `x + sublayer(norm(x))`. The addition gives gradients a direct path back through the network.

**What breaks without them:** gradients have to propagate through every transformation in series, and in a 6-layer stack they shrink enough that the early layers barely learn. Residuals also let a block learn an *adjustment* to its input rather than having to reproduce the whole representation.

### D9. Parameter count — derive it live (⚠ practise this)

Config: `V` = 8,000 (H) / 10,000 (L), d = 512, n_layer = 6, n_head = 8, d_ff = 2048, block = 512, bias = true, weight tying on.

```
EMBEDDINGS
  token embedding     V × 512        H: 8,000 × 512  = 4,096,000
                                     L: 10,000 × 512 = 5,120,000
  positional          512 × 512      =   262,144
                                     ─────────────
                      H embeddings   = 4,358,144
                      L embeddings   = 5,382,144

PER TRANSFORMER BLOCK
  attention: 4 projections (Q, K, V, O), each 512×512 + 512 bias
             4 × (262,144 + 512)          = 1,050,624
  FFN:       512×2048 + 2048              = 1,050,624
             2048×512 + 512               = 1,049,088
                                          ─────────────
                                          = 2,099,712
  2 LayerNorms: 2 × (512 weight + 512 bias) =     2,048
                                          ─────────────
  block total                             = 3,152,384

STACK
  6 blocks × 3,152,384        = 18,914,304
  final LayerNorm (512 + 512) =      1,024
                               ─────────────
  NON-EMBEDDING               = 18,915,328      ← identical for BOTH models

LM HEAD
  tied to the token embedding = 0 extra parameters

TOTAL
  Model H: 4,358,144 + 18,915,328 = 23,273,472
  Model L: 5,382,144 + 18,915,328 = 24,297,472
```

**The one sentence that makes this impressive:** *"The non-embedding count is identical at 18,915,328 for both models, which is deliberate — it means nobody can explain a difference in results by saying one model was bigger."*

**This is also a test:** `test_parameter_count_matches_analytic_estimate()` asserts the analytic formula against the loaded model, and `GPT.parameter_count()` returns the breakdown.

### D10. Weight tying

**SHORT ANSWER:** The output projection shares its matrix with the input token embedding. For Model H that saves 8,000 × 512 = **4,096,000** parameters, roughly 15% of the model.

**Why it works:** both matrices relate the same vocabulary to the same 512-dimensional space — one maps ID → vector, the other vector → score per ID. Having two independent representations of the same vocabulary is redundant at this scale, and tying acts as a regulariser.

**COMMON FOLLOW-UP:** *"Any downside?"* It constrains input and output representations to live in the same space, which at very large scale can cost a little quality. At 23M parameters with a 25M budget, spending 4M on a second copy of the vocabulary was clearly the worse trade.

**Test:** `test_weight_tying_shares_storage()` checks they are the same storage, not just equal values.

### D11. Positional embeddings

**SHORT ANSWER:** Learned absolute positional embeddings — a (512, 512) table added to the token embeddings.

**Why positions at all?** Self-attention is permutation invariant: shuffle the input tokens and, without position information, the attention output set is unchanged. "Ram beat Shyam" and "Shyam beat Ram" would be identical to the model.

**Why learned rather than sinusoidal?** The model is trained and evaluated at one fixed context length, so the extrapolation advantage of sinusoidal encodings does not apply, and learned positions let the model allocate resolution where the data actually needs it.

**What it costs:** 512 is a hard ceiling. There is no embedding row for position 512, so the model cannot be run on a longer sequence without retraining the table. `test_sequence_longer_than_block_size_is_rejected()` asserts this fails loudly rather than silently misbehaving.

**Both alternatives are implemented** — `sinusoidal_positional_encoding()` exists, and `pos_encoding: none` is what the bonus ablation uses.

### D12. Training setup

| | |
|---|---|
| Optimiser | AdamW, β = (0.9, 0.95) |
| Weight decay | 0.1, applied only where `param.dim() >= 2` |
| Peak / min LR | 6×10⁻⁴ → 6×10⁻⁵ |
| Schedule | 500 warmup steps, then cosine decay (`CosineWarmupSchedule`) |
| Gradient clipping | 1.0, applied **after** `unscale_` |
| Micro-batch × accum | 32 × 2 × 512 = **32,768 tokens per step** |
| Precision | bfloat16 |
| Seed | 20260902 |
| Planned steps | 15,000 |
| Reached | H 14,520 / L 14,440 (11-hour wall-clock guard) |
| Tokens consumed | H 475,791,360 / L 473,169,920 |
| Hardware | Kaggle Tesla T4, 10.99 h each, ~12,024 / 11,958 tok s⁻¹ |

**Why AdamW rather than Adam?** AdamW decouples weight decay from the gradient update. In plain Adam, L2 regularisation gets scaled by the adaptive learning rate, so it is applied inconsistently across parameters. AdamW applies it directly to the weights.

**Why β₂ = 0.95 instead of 0.999?** A shorter second-moment window adapts faster, which is the common choice for language-model pretraining at this scale.

**Why decay only matrices?** LayerNorm gains and biases are not weights in the same sense — decaying them just shrinks them toward zero for no regularisation benefit. The filter is `param.dim() >= 2` in `configure_optimizers()`.

**Why warmup?** At step 0 Adam's moment estimates are unreliable and the parameters are random. A full-size step then can push the model somewhere it takes a long time to recover from. 500 steps of linear ramp lets the moment estimates stabilise first.

**Why cosine decay?** A large LR early explores; a small LR late refines. Cosine is a smooth schedule that ends near zero without a hand-tuned staircase.

**Why gradient accumulation?** The effective batch I wanted (32,768 tokens) does not fit in T4 memory as one micro-batch. Accumulating 2 micro-batches of 32 decouples the statistical batch size from the memory limit. To run on a smaller GPU you lower `batch_size` and raise `grad_accum_steps` to keep the product the same.

**Why clip after unscale?** With mixed precision the loss is scaled up to avoid underflow, so the gradients are scaled too. Clipping before unscaling would clip the *scaled* gradients and the effective threshold would depend on the scale factor. You must unscale first, then clip to 1.0.

### D13. Checkpointing and resume

```
     training step
          │
          ├──► every 500 steps: evaluate on validation
          │          │
          │          └──► if best so far ──► save ckpt_best.pt
          │
          └──► save latest.pt

   checkpoint payload (_checkpoint_payload()):
       model weights
       optimizer state
       scheduler state
       training step
       best validation loss
       tokens seen
       RNG states (python + torch)
       full config

   resume (_maybe_resume()):  load all of the above, continue exactly
```

**Why all of it?** Restoring only the weights restarts the optimiser's moment estimates and the LR schedule from scratch, which is a different training run. Restoring the RNG states means the data order continues rather than restarting.

**Evidence:** I verified resume across three deliberately killed sessions before starting the real runs, because the whole thing runs on Kaggle and sessions do die.

**Why the 11-hour wall-clock guard?** So a dying session could never leave an unusable state. It fired at step 14,520 and 14,440 of a planned 15,000 — costing about 3% of the token budget.

### D14. Intrinsic results

| | Model H | Model L |
|---|---:|---:|
| Test cross-entropy (nats) | 3.3109 | 3.8433 |
| **Test perplexity** | **27.41** | **46.68** |
| Bits per token | 4.7766 | 5.5447 |
| Bytes per token | 8.875 | 11.441 |
| **Test bits-per-byte** | **0.5382** | **0.4846** |
| Tokens scored | 1,999,872 | 1,999,872 |
| Bytes scored | 17,749,798 | 22,880,653 |
| Best val loss / ppl | 3.3402 / 28.22 @ step 14,500 | 3.5953 / 36.43 @ step 14,000 |

See Part H for the full perplexity-vs-BPB argument — that is the key section.

### D15. Generation

Setup: 64 held-out prefixes of 64 tokens each, continued for 64 tokens, scored against the true continuations.

| Model | Decoding | BLEU-4 | chrF++ | ROUGE-L | Repetition | Distinct-1 | Distinct-2 |
|---|---|---:|---:|---:|---:|---:|---:|
| H | greedy | 1.96 | 12.49 | 0.093 | 0.598 | 0.172 | 0.313 |
| H | T=0.5 | 1.80 | 14.33 | 0.096 | 0.212 | 0.275 | 0.602 |
| H | T=1.0 | 1.01 | **16.29** | 0.100 | 0.002 | 0.520 | 0.939 |
| H | T=1.5 | 0.16 | 14.67 | 0.053 | 0.000 | 0.702 | 0.992 |
| L | greedy | 2.37 | 14.48 | 0.081 | 0.577 | 0.179 | 0.263 |
| L | T=0.5 | 1.88 | 16.54 | 0.092 | 0.190 | 0.345 | 0.611 |
| L | T=1.0 | 1.69 | **18.05** | 0.079 | 0.001 | 0.634 | 0.948 |
| L | T=1.5 | 0.78 | 16.13 | 0.035 | 0.000 | 0.793 | 0.995 |

**Temperature, explained with numbers.** Temperature divides the logits before softmax. Starting from A = 0.7, B = 0.2, C = 0.1:

- **T < 1** (say 0.5) sharpens: the gap widens, A gets closer to 1. More confident, more repetitive.
- **T = 1** leaves the distribution unchanged.
- **T > 1** (say 1.5) flattens: the three move toward equal. More diverse, more errors.
- **T → 0** is greedy decoding: always take the argmax.

**Why greedy degenerates.** At greedy, 58–60% of generated 4-grams are repeats. That is not a bug in the model — it follows from the rule. Greedy always takes the argmax, so the moment the model re-enters a state it has been in, it produces the same continuation, and there is no mechanism to escape the loop.

**The clean finding:** T = 1.0 gives the best chrF++ for both models *and* kills repetition. The usual quality–diversity trade-off does not bind anywhere in this range.

**Why is BLEU so low?** BLEU counts exact 4-gram overlap with a reference. An open-ended continuation has no particular reason to reproduce the one true continuation's 4-grams — a perfectly fluent, plausible continuation can score near zero. With a single reference the estimate is also very noisy. **chrF++ is the metric I would defend here**, because it works on character n-grams: in morphologically rich Devanagari, changing one suffix destroys a whole-word match but leaves most character n-grams intact. ROUGE-L is computed over whitespace tokens, which under-segment Devanagari, making it the least reliable of the three.

**Why report all three then?** Because the specification requires them. I report them and say which one I trust.

### D16. Attention analysis

**Attention entropy** — how spread out a head's attention is.

```
Low entropy  (≈0):  ▁▁▁█▁▁▁▁   one key gets almost all the weight → sharp, focused
High entropy (≈max): ▃▃▃▃▃▃▃▃   weight spread evenly → diffuse
```

Entropy of a probability distribution: `H = −Σ p log p`. For a uniform distribution over the allowed keys it is maximal.

**Mean attention distance** — how far back a head looks, on average, from query position to the key positions it attends to. Small = local, large = long-range.

| | Model H | Model L |
|---|---:|---:|
| Mean entropy (nats) | 3.2047 | 3.0898 |
| Uniform baseline | 5.2565 | 5.2565 |
| Mean distance (tokens) | 62.36 | 54.51 |
| Local-head fraction | 0.0625 | 0.1250 |

**Why compute the uniform baseline?** Because "entropy 3.2" means nothing on its own. Against a uniform baseline of 5.2565 you can say the heads are meaningfully more focused than chance. Without the baseline the number is uninterpretable. (`uniform_baseline_entropy()` in `hindi/eval/attention.py`.)

**Sharpest head per layer** (entropy @ mean distance):

| Layer | Model H | Model L |
|---|---|---|
| 0 | 4.16 @ 45.8 | 3.90 @ 36.0 |
| 1 | 3.09 @ 21.9 | 3.17 @ 17.1 |
| 2 | 1.49 @ 7.6 | 1.81 @ 10.6 |
| **3** | **0.06 @ 1.3** | **0.14 @ 1.1** |
| 4 | 1.49 @ 22.4 | 1.32 @ 5.6 |
| 5 | 2.99 @ 151.4 | 2.48 @ 81.3 |

**The result worth talking about:** both models independently grew a **previous-token head at layer 3** — entropy 0.06 at distance 1.3 for Hindi, 0.14 at distance 1.1 for Nepali. Different languages, different tokenizers, different seeds, same structure. Attention is diffuse at layer 0, sharpens to layer 3, then broadens again at layer 5.

**Why would a previous-token head emerge?** Language is locally predictable — the immediately preceding token is the single most informative context for the next one, especially for morphology and agreement. A head that just copies "what came immediately before" is cheap to learn and immediately useful, so gradient descent finds it early.

**⚠ THE TRAP — say this before they ask:** *"This is a consistent pattern, not evidence of a functional role. I measured which heads are sharp and how far they look. I did not establish what any head contributes to the prediction — that would need ablating a head and re-scoring, which I did not run."*

---

## PART E — Phase 3 Viva

### E1. Why synthetic data?

**SHORT ANSWER:** So that I know the ground-truth answer by construction and can control the train/test relationship exactly. The brief also requires it — the primary finetuning set must not be a downloaded benchmark.

**IF ASKED MORE:** With a generator I can guarantee that test questions use entity names and sentence templates the model has never seen. With a downloaded dataset I would have no idea what overlaps what, and no way to assert it.

### E2. The four task families

Real examples from `hindi/reasoning/data/test.jsonl` and `nepali/reasoning/data/test.jsonl`:

**`compare2`** — one-step comparison, 2 candidates:
> ध्यान दीजिए: प्रवीण वरुण से हल्का है। भारी कौन है? → **वरुण**
> *(Praveen is lighter than Varun. Who is heavier?)*
> Note the direction flip: "lighter than" in the premise, "heavier" in the question.

**`numeric2`** — explicit quantities, 2 candidates:
> दिया गया है कि चंदा का वज़न ६७ किलो है और मालती का वज़न ४८ किलो है। हल्की कौन है? → **मालती**

**`numeric3`** — explicit quantities, 3 candidates:
> यहाँ प्रवीण की उम्र ५९ साल, नवीन की उम्र ६७ साल, तथा अरुण की उम्र २४ साल है। सबसे बड़ा कौन है? → **नवीन**

**`transitive3`** — two-hop chain, 3 candidates:
> शारदा सविता से भारी है। इसी तरह सविता मालती से भारी है। सबसे भारी कौन है? → **शारदा**
> *(Sharda > Savita, Savita > Malti. Who is heaviest?)*

Nepali equivalents use Arabic digits — note `दिइएको छ कि रबिनको उमेर 22 वर्ष...` — which is the byte-fallback fix from Part C13.

### E3. Why `transitive3` is the real test

```
Premise 1:   A  >  B
Premise 2:        B  >  C
                  ───────
Question:    who is largest?        Answer: A
```

**What information is needed?** You must chain the two inequalities. Neither premise alone identifies A as the maximum.

**Why is this transitive reasoning?** It requires the inference `A > B ∧ B > C ⟹ A > C`, which is not stated anywhere in the prompt.

**What would a shortcut-based model do?** Notice that **B is the only name mentioned twice** and pick it on salience. That is exactly what my models do — see E8.

### E4. Dataset construction and leakage control

```
   templates 14 / 4 / 4            entity names 40 / 10 / 10
            │                               │
            └──────────┬────────────────────┘
                       ↓
             seeded ReasoningGenerator
                       ↓
          assert names disjoint across splits
          assert templates disjoint across splits
                       ↓
             train 20,000 / val 2,000 / test 2,000
             4 task families × 5,000 train each
```

**SHORT ANSWER:** Leakage is controlled on two axes at once. The 40/10/10 entity names and the 14/4/4 sentence templates are both disjoint across splits, so a test question is about people the model has never seen, phrased in a way it has never seen.

**IF ASKED MORE:** Both properties are **asserted inside `main()` of `generate.py`**, not assumed, and recorded in `dataset_stats.json` as `entity_names_disjoint_across_splits: true` and `templates_disjoint_across_splits: true`. The Hindi and Nepali sets share no sentence, no entity name and no template.

**COMMON FOLLOW-UP:** *"Why hold out both names and templates?"* Holding out only names would let the model pass by memorising which template implies which answer position. Holding out only templates would let it memorise per-name answer priors. You need both axes closed.

### E5. The gender-agreement leak (⚠ good story)

**What happened:** Hindi adjectives inflect for gender. A question like "who is smallest?" asked about one woman and two men has only one grammatically well-formed answer — so you can get it right without comparing anything.

**How I detected it:** inspecting generated examples looking for shortcuts. **181 of 400** sampled examples were solvable that way.

**Why it happened:** the generator drew entity names independently of gender, and the adjective had to agree with *some* gender.

**How I fixed it:** `inflect()` plus drawing all entities in an example from one gender, so agreement carries no information about the answer.

**Result:** the shortcut is closed. Had it survived, my accuracy numbers would have been meaningless.

### E6. Finetuning protocol

| | |
|---|---|
| Starting point | each model's **own** Phase 2 checkpoint |
| Tokenizer | held fixed from Phase 1 |
| Loss | answer tokens only (`ignore_index=-1` over the prompt) |
| Batching | grouped by length, **not** padded |
| Epochs / batch | 2 / 32 |
| Peak LR | 6×10⁻⁵ (one tenth of pretraining), min 6×10⁻⁶ |
| Warmup | 100 steps |
| Eval interval | every **20** steps |
| Seed | 20260914 |
| Precision | float16 + `GradScaler` |

**Why answer-token-only loss?** If I computed loss over the whole sequence, most of it would be the question text, and the model would be rewarded for memorising question phrasing rather than producing the right answer. Masking the prompt with `ignore_index=-1` means the gradient only ever comes from the answer.

**Why length-grouped batches instead of padding?** This is *necessary, not cosmetic*. With learned absolute positional embeddings, left-padding would shift the position index of every real token — token "A" would be at position 3 in one batch and position 7 in another, and the model would see a different positional embedding for the same content. Grouping by length avoids padding entirely.

**Why LR one tenth of pretraining?** The model already holds a language model worth keeping. A pretraining-sized step overwrites it before the task is learned; the usual failure mode is a model that answers in the right format but has forgotten how to read the question.

**Why float16 and not bfloat16?** The Kaggle T4 is compute capability 7.5 and has no bfloat16 support, so the pretraining config could not be reused. float16 needs a `GradScaler` to avoid gradient underflow.

**A guard worth mentioning:** the finetuning script **asserts** that the tokenizer vocabulary size matches the checkpoint, rather than trusting the config file. That mismatch is exactly the kind you do not notice until the results are garbage.

### E7. Exact match vs forced choice (⚠ your single most important Phase 3 section)

```
  EXACT MATCH                           FORCED CHOICE
  ───────────                           ─────────────
  Greedy-decode an answer.              Take only the entity names the
  Does the string equal the gold?       question actually contains.
                                        Score each one's length-normalised
                                        mean log-probability.
                                        Does the highest-scoring one match?

  Honest end-to-end measure.            Removes the formatting problem.
  But pinned at 0 before finetuning,    Chance rate computed per example
  so it cannot distinguish              from the number of candidates.
  "learned nothing" from
  "knows it but can't say it".
```

**Why both?** Exact match after greedy decoding is the honest end-to-end result, but a model that has never seen this prompt format scores zero no matter what it knows. Forced choice fixes that — it asks which of the candidate names the model assigns the highest probability to, so formatting is out of the picture.

**The chance rate.** `compare2` and `numeric2` have 2 candidates (chance 50%); `numeric3` and `transitive3` have 3 (chance 33.3%). Averaged over the equal-sized families: (0.5 + 0.5 + 1/3 + 1/3) / 4 = **0.4167**. It is computed per example from the actual candidate count, not assumed.

**The results:**

| | H pretrained | H finetuned | L pretrained | L finetuned |
|---|---:|---:|---:|---:|
| Exact match | 0.00% | **35.15%** | 0.00% | **29.45%** |
| Forced choice | 39.80% | 36.35% | 38.80% | 39.30% |
| Chance | 41.67% | 41.67% | 41.67% | 41.67% |
| Answer-token loss | 5.2439 | 0.9408 | 5.5367 | 1.4351 |
| Answer-token perplexity | 189.40 | 2.56 | 253.84 | 4.20 |

**THE ANSWER TO GIVE:**

> **Observation:** exact match rose from zero to 35.15% and 29.45%, and answer-token loss fell by more than four nats in both models.
> **Interpretation:** finetuning taught the models to emit a candidate name in the expected position and format. The loss collapse is consistent with that and with nothing stronger.
> **What it does not show:** under forced choice both models sit at or below the 41.67% chance rate, and per task the scores land close to their individual chance rates. So the result does not establish that either model learned the comparison operation.

**Per task, finetuned:**

| Task | Chance | H exact | L exact | H forced | L forced |
|---|---:|---:|---:|---:|---:|
| `compare2` | 50.0% | 47.40% | 40.20% | 50.40% | 49.40% |
| `numeric2` | 50.0% | 47.40% | 38.00% | 47.00% | 51.80% |
| `numeric3` | 33.3% | 33.40% | 26.20% | 33.20% | 32.40% |
| `transitive3` | 33.3% | 12.40% | 13.40% | **14.80%** | **23.60%** |

Three families sit on their chance line. The two-hop family falls **below** it.

### E8. Error analysis — the two-hop failure

`transitive3`, finetuned, distribution over which prompt position the answer came from:

| | answered A (first) | answered B (middle) | answered C (last) | off-prompt |
|---|---:|---:|---:|---:|
| **Gold** | 47.4% | **0.0%** | 52.6% | — |
| Model H | **0.0%** | **72.6%** | 25.4% | 2.0% |
| Model L | 3.2% | 41.8% | 26.2% | 28.8% |

**The argument, in one breath:** the chain is stated as *A rel B, B rel C*, so the extremes are always A and C and the middle entity is **never** the correct answer. Model H nevertheless picks the middle entity 72.6% of the time and the first entity in **none**. B is the only name mentioned twice. So the model is selecting on surface salience, not following the inequality — and that is precisely why this family scores *below* chance, because uniform guessing would at least hit A and C sometimes.

**One-hop recency bias**, Model H:

| Family | answered first-named | answered second-named | gold split |
|---|---:|---:|---:|
| `compare2` | 30.8% | **62.0%** | 48 / 52 |
| `numeric2` | 35.0% | **65.0%** | 50 / 50 |

Against a balanced gold distribution the model prefers the most recently mentioned name about two to one.

**Groundedness — the clearest resource-tier difference in the project.** Rate of answering with a name the question never mentions:

| Family | Model H | Model L |
|---|---:|---:|
| `compare2` | 7.2% | 19.2% |
| `numeric2` | 0.0% | 25.0% |
| `numeric3` | 1.0% | 21.2% |
| `transitive3` | 2.0% | 28.8% |

Model L emits memorised *training* names (सिर्जना, निरु) instead of a candidate in front of it. Before finetuning both models do this essentially 100% of the time. So finetuning moved Model H almost entirely onto the prompt and Model L only about three quarters of the way.

**Say this:** *"The resource tier shows up not as reasoning accuracy — both are at chance there — but as groundedness. The better-pretrained model is better at using its input. That is a weaker and more defensible claim than 'better reasoning'."*

### E9. Overfitting and checkpoint selection

```
  loss
   │  ╲
   │   ╲___ train loss keeps falling → 0.4441 (H) / 0.6048 (L)
   │    ╲____________________________
   │   ╱
   │  ╱  validation loss rises → 3.5188 (H) / 3.5911 (L)
   │ ╱
   │╱  ← minimum at step 60
   └─────────────────────────────────── step
      0   60            ...        1,250
```

**What does this indicate?** The model stops learning the task and starts memorising the training set.

**Why did it happen here?** With only 40 entity names and 14 templates, memorising which answer tends to be right is easier than learning to read the question. The training distribution is too narrow in two independent ways: too few entities, and every example of a template is identical apart from slot fillers, so the surface form carries almost no entropy.

**Why is checkpoint selection critical?** Because the final checkpoint is badly overfit. I save on **best validation loss**, so the evaluated checkpoints are **step 60** — best val 1.4966 (H) and 1.3427 (L). The overfitting visible in the curve never reached my reported numbers.

**⚠ THE RUN A / RUN B STORY (volunteer this):**

> My first finetuning run evaluated every **100** steps and selected its best checkpoint at step 100 — which is exactly where warmup ends and the LR first reaches its peak — with validation loss rising monotonically after that. The grid was too coarse to locate the real optimum. I moved to a **20**-step grid with nothing else changed, and Model L went from 14.90% to 29.45% exact match and Model H from 32.70% to 35.15%. The superseded run is kept in `report/phase3/run_a_40names/` rather than deleted.

That is a one-variable change with a measured effect, and keeping the old run is good practice.

### E10. Attention before vs after finetuning

Measured on the same held-out reasoning prompts, all of identical token length, so neither prompt content nor sequence length could masquerade as a finetuning effect.

| Model | Checkpoint | Mean entropy | Mean distance | Uniform baseline |
|---|---|---:|---:|---|
| H | pretrained (step 14,500) | 1.9102 | 10.303 | 2.9545 (length 46) |
| H | finetuned (step 60) | 1.8691 | 10.360 | |
| L | pretrained (step 14,000) | 1.8327 | 9.415 | 2.8061 (length 39) |
| L | finetuned (step 60) | 1.7833 | 10.157 | |

**Observation:** attention became slightly sharper *and* slightly longer-range at the same time in both models, most visibly in Model L where mean distance rises 0.74 positions on a 39-token prompt. The largest per-head shifts concentrate in the middle layers — where Phase 2 found the previous-token heads.

**Interpretation:** consistent with a shift toward attending to the entity names scattered across the prompt rather than to local context.

**Limitation:** *"This identifies which heads changed, not what they compute. Showing that these heads perform the comparison would require ablating a head and re-scoring, which I did not run."*

---

## PART F — Cross-Phase Questions

**1. How did tokenizer design affect Phase 2?**
Directly and decisively. Model L's BPE at 10,000 produces 11.44 bytes per token against Model H's 8.88. That single property is large enough to reverse the ranking between perplexity and bits-per-byte. Any cross-language perplexity comparison in this project is a statement about tokenizers at least as much as about models.

**2. How did Phase 1 data quality affect Phase 2?**
Model L's corpus is nominally larger (520M vs 485M train tokens) but 91.6% of its manual lane is a single news source, while Model H draws on five distinct manual sources and five downloaded partitions. Model L saw *more* tokens of *narrower* text. That is a plausible contributor to its higher perplexity — though I cannot cleanly separate it from tokenizer effects.

**3. Which Phase 1 decision most affected later phases?**
The missing byte fallback on the Nepali tokenizer. In Phase 1 it appeared only as a column of zeros I misread as a good score. In Phase 2 it surfaced as a 0.17–0.34% unknown-token rate. In Phase 3 it became outright corruption — no Devanagari digit in the vocabulary, so 4.8% of the generated reasoning corpus became `<unk>`. One configuration flag, invisible for two phases, then responsible for a data defect.

**4. Why might a model with better language modelling still fail reasoning?**
Because next-token prediction and relational inference are different capabilities. Both my models are at chance on the comparison itself despite being usable language models. What better pretraining bought was groundedness — Model H answers from the given candidates 93–100% of the time against Model L's 71–81% — not reasoning.

**5. How did Phase 2 attention analysis help Phase 3?**
It gave me a baseline. Phase 2 located previous-token heads at layer 3; Phase 3 found the largest finetuning-induced shifts in those same middle layers. Without the Phase 2 measurement, the Phase 3 deltas would have had nothing to be relative to.

**6. What evidence supports your main conclusion?**
That most of the H/L perplexity gap is tokenizer granularity: the bytes-per-token measurement (8.88 vs 11.44) is large enough to flip the ordering, and the flip is exactly what bits-per-byte shows.

**7. What evidence does NOT support it?**
Bits-per-byte is tokenizer-neutral but **not corpus-neutral**. The two models are scored on different text of different intrinsic difficulty, so I cannot conclude Model L is the better language model — only that the perplexity gap overstates the difference. Separating those would need a parallel evaluation set, which I do not have.

**8. Which Phase 1 decision would you change?**
Hindi deduplication. Running MinHash near-duplicate removal and train/test decontamination on Hindi, as I did for Nepali.

**9. What is your strongest result?**
The perplexity/bits-per-byte reversal — it is clean, mechanistically explained, and I can derive it live.

**10. What is your weakest result?**
The reasoning results. Both models are at chance, and the training distribution was too narrow to give finetuning a fair chance.

**11. What surprised you?**
That both models independently grew a near-deterministic previous-token head at layer 3, with different languages, tokenizers and seeds. I did not expect that structural convergence at 23M parameters.

**12. What experiment would you run next?**
Head ablation — disable the heads that moved most during finetuning and re-score the reasoning test set. That converts the attention analysis from correlational to causal, which is the main thing it currently lacks.

---

## PART G — Code-Level Questions

Know where things live. If an examiner opens the repo, you should be able to navigate it.

| Area | File | Key names |
|---|---|---|
| Cleaning | `scripts/clean_lib.py` | `normalize_text()`, `devanagari_ratio()`, `malformed_ratio()`, `is_too_short()`, `langdetect_verdict()`, `dedup_key()`, `strip_latin_tokens()` |
| Hindi pipeline | `hindi/data/pipeline/` | `download_hindi.py`, `ingest_manual.py`, `ingest_wikipedia_dump.py`, `clean_hindi.py`, `split_hindi.py` |
| Nepali pipeline | `nepali/scripts/` | `s00_doctor.py` … `s14_corpus_stats.py`, `run_pipeline.py` |
| Nepali dedup | `nepali/scripts/corpuslib/dedup.py` | `MinHasher`, `shingles()`, `build_lsh_buckets()`, `remove_near_duplicates()`, `find_contaminated()` |
| Nepali language ID | `nepali/scripts/corpuslib/langid.py` | `LanguageIdentifier`, `self_test()`, `classify()` |
| Tokenizer | `hindi/tokenizer/train_tokenizer.py` | `analyze_tokenizer.py` |
| Packing | `hindi/data/pack_corpus.py` | `CorpusPacker.pack_split()` |
| Model | `hindi/model/transformer.py` | `GPT`, `TransformerBlock`, `MultiHeadCausalSelfAttention`, `FeedForward`, `LayerNorm`, `sinusoidal_positional_encoding()`, `GPT.generate()`, `GPT.parameter_count()`, `GPT.configure_optimizers()` |
| Training | `hindi/train/train.py` | `Trainer`, `CosineWarmupSchedule`, `_checkpoint_payload()`, `_maybe_resume()`, `evaluate()` |
| Evaluation | `hindi/eval/evaluate.py` | `Evaluator`, `perplexity_and_bpb()`, `generation_suite()`, `attention_report()`, `rouge_l()`, `diversity_metrics()`, `overlap_metrics()` |
| Attention | `hindi/eval/attention.py` | `collect_attention()`, `attention_entropy()`, `mean_attention_distance()`, `summarise_model_attention()`, `uniform_baseline_entropy()`, `head_classification()` |
| Reasoning data | `hindi/reasoning/generate.py` | `ReasoningGenerator`, `.compare2()`, `.transitive3()`, `.numeric2()`, `.numeric3()`, `inflect()`, `to_devanagari()` |
| Reasoning dataset | `hindi/reasoning/dataset.py` | `ReasoningDataset`, `.batches()`, `.prompt_batches()`, `.sequence_stats()` |
| Finetuning | `hindi/reasoning/finetune.py` | `FineTuner`, `FinetuneConfig` |
| Reasoning eval | `hindi/reasoning/evaluate_reasoning.py` | `ReasoningEvaluator`, `.greedy_answers()`, `.forced_choice()`, `.answer_loss()`, `position_analysis()`, `breakdown()` |
| Attention compare | `hindi/reasoning/attention_compare.py` | `compare()`, `per_head_table()`, `reasoning_prompt_batch()` |
| Tests | `hindi/tests/test_model.py` | `test_causal_mask_blocks_the_future()`, `test_parameter_count_matches_analytic_estimate()`, `test_weight_tying_shares_storage()`, `test_no_prebuilt_attention_modules_used()` |

### Likely code questions

**Q: What does `dedup_key()` do and what would break without it?**
It produces a SHA-1 hash of the text after collapsing whitespace, stripping punctuation and case-folding, so formatting variants of the same document collide. Without it, exact duplicates survive into training, inflating apparent corpus size and risking train/test overlap. Its own docstring is explicit that it does **not** catch fuzzy or paraphrase similarity.

**Q: What does `test_no_prebuilt_attention_modules_used()` check?**
It walks the module tree and fails if `nn.Transformer*`, `nn.MultiheadAttention` or any pre-built attention block appears. It exists so the hard constraint cannot be broken by a later edit — I cannot accidentally violate the rule.

**Q: In `MultiHeadCausalSelfAttention.forward()`, what are the tensor shapes?**
Input (B, T, 512). Each of the three projections gives (B, T, 512) → `view` (B, T, 8, 64) → `transpose` (B, 8, T, 64). Scores QKᵀ are (B, 8, T, T). After masking, softmax, dropout and × V we get (B, 8, T, 64) → transpose and reshape back to (B, T, 512) → W_O.

**Q: Where is dropout applied?**
Four places: attention probabilities, attention output, feed-forward output, and the embedding sum. Dropout on the attention probabilities is the one people forget — it stops the model relying on one specific key.

**Q: What is in `_checkpoint_payload()`?**
Model weights, optimizer state, scheduler state, training step, best validation loss, tokens seen, both RNG states, and the full config.

**Q: What does `forced_choice()` compute?**
For each test example it scores every candidate entity name from the question by its length-normalised mean log-probability under the model, and picks the highest. Length normalisation matters because otherwise longer names are systematically penalised.

**Q: Why does `ReasoningDataset.batches()` group by length?**
Because padding would shift position indices under learned absolute positional embeddings. See E6.

---

## PART H — Results and Metrics Questions

### H1. Perplexity vs bits-per-byte (⚠ the single most important section)

**The formulas:**

```
Cross-entropy (nats/token):   L = −(1/N) Σ log p(x_t | x_<t)

Perplexity:                   PPL = exp(L)

Bits per token:               bits = L / ln(2)

Bits per byte:                BPB  = bits per token / bytes per token
```

**Derive it live with my numbers:**

```
Model H:  L = 3.3109 nats
          PPL = exp(3.3109) = 27.41
          bits/token = 3.3109 / 0.6931 = 4.7766
          bytes/token = 17,749,798 / 1,999,872 = 8.875
          BPB = 4.7766 / 8.875 = 0.5382

Model L:  L = 3.8433 nats
          PPL = exp(3.8433) = 46.68
          bits/token = 3.8433 / 0.6931 = 5.5447
          bytes/token = 22,880,653 / 1,999,872 = 11.441
          BPB = 5.5447 / 11.441 = 0.4846
```

**The story:**

> Perplexity says Model H is 41% better. Bits-per-byte says Model L is slightly better. Both are correct, because they measure per different units. Perplexity is per *token*, and a token is not the same amount of text in the two models — Model L's BPE holds 11.44 bytes where Model H's Unigram holds 8.88. Model L's tokens are individually harder to predict, but each one carries about 29% more text. Dividing by bytes takes the tokenizer out of the comparison, and when you do, the ordering reverses.

**Examiner: "Why is the model with worse perplexity not necessarily worse?"**
Because perplexity is per token and the token is defined by the tokenizer, not by the language. A coarser tokenizer makes each prediction harder — more options, more content per step — so it inflates perplexity even if the model is representing the text just as well. You are comparing two different measuring sticks.

**Examiner: "Why did the ranking change?"**
Because bytes-per-token differs by 29% between the two models, which is enough to more than offset a 16% difference in bits per token.

**Examiner: "Does BPB completely solve the comparison problem?"**
**No** — and this is the answer that earns credit. BPB is tokenizer-neutral but **not corpus-neutral**. The two models are evaluated on different text of different intrinsic difficulty: Model L's corpus is nominally larger but 91.6% of its manual lane is one news source, so it may simply be more predictable text. Separating "better model" from "easier corpus" would need a parallel or translated evaluation set, which I do not have.

**Examiner: "So which model is better?"**
*"On their own test sets, the honest answer is that I can't rank them cleanly. I can say the perplexity gap overstates the difference, and I can say why."*

### H2. Why is BLEU so low if the text is reasonable?

BLEU counts exact n-gram overlap against a reference. Open-ended continuation has many valid outputs, and the one reference I have is only one of them. A fluent, plausible continuation that shares no 4-grams with that specific reference scores near zero. BLEU is designed for tasks with constrained outputs, like translation. For free-form generation in a morphologically rich language it is close to uninformative.

### H3. Attention entropy and distance — see D16.

### H4. Numbers that prove things

- Causal mask leakage: **exactly 0.000e+00**
- Non-embedding parameters: **18,915,328 for both models** (capacity controlled)
- Cross-corpus document overlap: **0**
- Nepali split document-id overlap: **0**
- Generated reasoning data unknown-token rate after the digit fix: **exactly 0**

---

## PART I — "Why Did You Make This Choice?"

| Question | BEST SHORT ANSWER | DEEPER IF ASKED | POSSIBLE FOLLOW-UP |
|---|---|---|---|
| Why this tokenizer per language? | I ran a sweep and measured fertility and unknown rate on held-out text; Unigram won for Hindi, BPE for Nepali. | Full sweep in `measured_tokenizer_metrics.json`; I did not pick by preference. | "Would BPE have worked for Hindi?" — Probably; I chose on measurement, not principle. |
| Why vocab 8,000 for Hindi? | Fertility keeps improving but 8k→10k buys less than 5k→8k and costs ~1M parameters out of 25M. | Diminishing returns against a hard parameter budget. | "So bigger is better if you had room?" — For fertility yes; it is a budget trade. |
| Why 6 layers? | Each block is 3.15M parameters, so 6 plus embeddings is 23.27M; 7 would overshoot to 26.4M. | I preferred depth over width because attention needs several rounds of composition to build multi-hop dependencies. | "Why not 4 wider layers?" — Same budget, less composition depth. |
| Why context 512? | It keeps the T×T attention matrix at 262k entries per head, which is what makes a 500M-token budget affordable in a Colab/Kaggle session. | Attention is O(T²); doubling context quadruples that cost. | "What limits you to 512 at inference?" — Learned absolute positions: no embedding row for position 512. |
| Why causal attention? | The objective is next-token prediction; without the mask the model could see the token it is predicting. | See D5 — and I verified it empirically, not just structurally. | "Could you train bidirectionally?" — Different objective, different model. |
| Why AdamW? | It decouples weight decay from the adaptive update, so regularisation is applied consistently. | In plain Adam, L2 gets scaled by the per-parameter learning rate. | "Why decay only matrices?" — Decaying LayerNorm gains and biases just shrinks them for no benefit. |
| Why LR 6e-4? | Standard for this model scale; paired with 500 warmup steps and cosine decay to 6e-5. | Warmup because Adam's moment estimates are unreliable at step 0. | "Did you tune it?" — No, I spent the compute budget on the two full runs. Honest answer. |
| Why gradient accumulation? | The 32,768-token effective batch does not fit in T4 memory as one micro-batch. | 32 × 2 × 512; lower batch_size and raise accum to run on a smaller GPU. | "Is it equivalent to a real big batch?" — For the gradient yes; for BatchNorm it would not be, but there is none here. |
| Why weight tying? | Saves 4,096,000 parameters (~15%) and the two matrices describe the same vocabulary in the same space. | Also acts as a regulariser. | "Downside?" — Constrains input and output to share a space. |
| Why synthetic reasoning data? | So I know the ground truth by construction and can guarantee disjoint names and templates across splits. | The brief also forbids a downloaded benchmark as the primary set. | "Isn't synthetic data too easy?" — Evidently not; both models are at chance on two-hop. |
| Why no pretrained model? | The brief forbids it, and the point is to understand every tensor operation. | I also added a test that fails if a pre-built attention block appears. | — |
| Why forced choice? | Because exact match is pinned at zero before finetuning and cannot separate "learned nothing" from "can't phrase it". | It is also the less flattering number, which is why it is the informative one. | "Why not just exact match?" — It would have let me claim reasoning I cannot support. |
| Why is attention analysis not enough? | It is correlational. It shows which heads are sharp or changed, not what they contribute. | A causal claim needs head ablation and re-scoring. | "So why do it?" — It is still evidence, and the Phase 2/Phase 3 comparison is meaningful. |
| Why document-level split? | So two halves of the same article cannot land on opposite sides and turn memorisation into test accuracy. | Nepali also holds out before quality filtering so filters cannot curate the test set. | "Does that waste data?" — Marginally; correctness is worth more. |
| Why remove duplicates? | Duplicates inflate apparent corpus size, overweight repeated text in training, and can leak into evaluation. | Hindi only does exact; Nepali does near-duplicate and decontamination. | ⚠ Be honest about the asymmetry. |
| Why evaluate every 20 steps in finetuning? | Run A's 100-step grid selected its best checkpoint at step 100, exactly where warmup ends — too coarse to find the optimum. | Changing only that raised L from 14.90% to 29.45%. | "Why not every step?" — Evaluation cost; 20 samples the window properly. |

---

## PART J — What Went Wrong / Limitations

Each of these is **Problem → Detection → Fix → Result/Lesson**.

**1. Nepali language ID silently deleted the corpus.**
*Problem:* the `fasttext` wheel calls `np.array(probs, copy=False)`, which raises on NumPy ≥2; every `predict()` threw and the caller swallowed it as "not our language". Millions of documents were deleted while the stage logged SUCCESS.
*Detection:* the surviving fraction was 5.85%, and it was exactly the one source configured to skip language ID.
*Fix:* call the raw binding directly, bypassing the broken wrapper; add a startup self-test on known Hindi/Nepali/English strings; count errors separately from rejections and raise past a threshold.
*Lesson:* a bare `except Exception` around a classifier turns a library bug into silent data loss.

**2. Checkpoint-selection grid too coarse.**
*Problem:* 100-step evaluation grid selected step 100, exactly where warmup ends.
*Detection:* validation loss rising monotonically from the first eval point — a suspicious pattern.
*Fix:* 20-step grid, nothing else changed.
*Result:* Model L 14.90% → 29.45%, Model H 32.70% → 35.15%. Old run kept in `report/phase3/run_a_40names/`.

**3. Gender-agreement leak in the Hindi reasoning data.**
*Problem:* adjective inflection made 181/400 examples solvable without comparing.
*Detection:* manual inspection of generated examples for shortcuts.
*Fix:* draw all entities in an example from one gender.
*Lesson:* in an inflected language, grammar itself can be a leak channel.

**4. Nepali tokenizer has no byte fallback.**
*Problem:* no Devanagari digit in the vocabulary; 4.8% of generated reasoning data became `<unk>`.
*Detection:* tokenized the generated set and checked its unknown rate **before** training.
*Fix:* switched Nepali numerals to Arabic digits.
*Lesson:* a column of zeros is not automatically a good score — check whether it means "none" or "absent".

**5. Neither pretraining run converged.**
Both hit the 11-hour guard at ~97% of the planned token budget with validation loss still falling. Perplexities are upper bounds. Mitigated by both runs stopping under the same rule with near-identical budgets, so the *comparison* is unaffected.

**6. Hindi deduplication is weaker than Nepali's.**
No near-duplicate removal, no decontamination. An unquantified confound in the direction that flatters Model H.

**7. The reasoning training distribution is too narrow.**
40 entity names and 14 templates, with every example of a template identical apart from slot fillers. Both models overfit by step 60. Fixes: more entities (200+), and per-template linguistic variation — synonym substitution, clause reordering, optional discourse particles. The generator has no variation layer today.

**8. Attention analysis is correlational.**
No head-ablation probe was run.

**9. Manual-share methodology differs between the languages.**
Hindi's share is character-weighted over the cleaned corpus; Nepali's is token-weighted. Not strictly like-for-like.

**10. Evaluation subsampled.**
Intrinsic metrics are computed over 2M-token subsets, not the full 10.1M/57.7M test sets — but with an identical budget for both models.

---

## PART K — Trick / Follow-Up Questions

**Q: "Your perplexity is lower, so Model H is definitely the better model, right?"**
No. Perplexity is per token, and the two tokenizers produce different-sized tokens — 8.88 against 11.44 bytes. On bits-per-byte the ordering actually reverses, 0.5382 against 0.4846. The honest statement is that the perplexity gap overstates the difference, and that I cannot cleanly rank the two because they are scored on different corpora.

**Q: "Your attention analysis shows the model is reasoning, correct?"**
No. It shows which heads are sharp, how far they look, and which ones changed during finetuning. It does not show what any head contributes to the prediction. Establishing that would require ablating a head and re-scoring. And separately, my reasoning metrics say the models are at chance, so there is no reasoning for attention to be evidence of.

**Q: "Exact match went from 0% to 35%, so finetuning taught reasoning?"**
It taught the answer *format*. Under forced choice — restricting the model to the names already in the question — both models sit at or below the 41.67% chance rate. Per task they land on their individual chance lines. The one family that leaves chance leaves it downward.

**Q: "Why not just use BLEU for generation?"**
Because BLEU counts exact 4-gram overlap with one reference, and open-ended continuation has many valid outputs. My BLEU is near the floor at every setting while chrF++ varies meaningfully and peaks at T=1.0. In morphologically rich Devanagari, one changed suffix destroys a whole-word match but leaves most character n-grams intact, so chrF++ is the better instrument.

**Q: "Why not train longer?"**
I would have liked to — both validation curves were still falling. The constraint was the Kaggle GPU quota; an 11-hour wall-clock guard in the trainer stopped each run so a dying session could never leave an unusable state. I report the runs as-run rather than implying convergence.

**Q: "Why not use a bigger model?"**
The brief specifies approximately 25M parameters. Within that, I chose 6 layers over a wider 4-layer model, because a seventh layer would have overshot to 26.4M.

**Q: "Why not one tokenizer for both languages?"**
The brief forbids sharing vocabulary, and it would have made this a multilingual model rather than two independent monolingual ones. It would also have destroyed the comparison — I could not attribute any difference to the tokenizer if both used the same one. It would have been technically easy since both use Devanagari, which is presumably why the rule exists.

**Q: "Why not split randomly at the chunk level? It's simpler."**
Because two halves of the same article could land on opposite sides, and the test score would partly measure memorisation rather than generalisation.

**Q: "Why do you need a validation set at all?"**
To select the checkpoint without touching test. In Phase 3 this was decisive: validation loss bottoms at step 60 and rises afterwards, so the final checkpoint is badly overfit. Without validation-based selection my reported numbers would have come from a memorised model.

**Q: "Why not use the final checkpoint?"**
See above — at step 1,250 training loss is 0.4441 while validation loss has risen to 3.5188. The final checkpoint is the worst one to report.

**Q: "Why does a lower loss not always mean better reasoning?"**
Because loss here is answer-token cross-entropy, and it falls if the model simply learns that the answer is always one of the two or three names in a fixed position. My answer-token loss fell from 5.2439 to 0.9408 while forced-choice accuracy went *down* slightly, from 39.80% to 36.35%. That is the cleanest demonstration in the project that loss and capability can move in opposite directions.

**Q: "Isn't your Bhojpuri work wasted?"**
It is a documented negative result. I built the full pipeline, it passed validation, and public text ran out at 84,217,032 tokens — 16.8% of target. The lesson I took is to measure how much text exists for a language before building a pipeline around it.

**Q: "Your two models have different parameter counts — isn't that unfair?"**
The difference is entirely the embedding table, because the vocabularies differ (8,000 vs 10,000). The **non-embedding** count is identical at 18,915,328. The compute spent on actual transformation is the same.

**Q: "Could your Hindi results be inflated by train/test leakage?"**
Possibly, and I would rather say so. Hindi deduplicates only on exact SHA-1 and has no decontamination stage, while Nepali has both. The direction of that bias flatters Hindi. I did not quantify it; fixing it would mean running the Nepali MinHash decontamination over the Hindi corpus.

---

## PART L — Rapid-Fire (1–2 sentence answers)

1. **Model H language?** Hindi, higher-resource.
2. **Model L language?** Nepali, from the allowed list.
3. **Why the same script?** To hold orthography constant so differences come from data and tokenizer.
4. **Hindi corpus size?** 505,852,028 tokens, 704,101 documents.
5. **Nepali corpus size?** 635,204,333 tokens, 1,918,265 documents.
6. **Manual share?** 30.66% Hindi, 26.41% Nepali; floor was 20%.
7. **Hindi tokenizer?** SentencePiece Unigram, vocab 8,000.
8. **Nepali tokenizer?** SentencePiece BPE, vocab 10,000.
9. **Hindi fertility?** 1.3956 tokens per word.
10. **Nepali unknown rate?** 0.003387 — non-zero because there is no byte fallback.
11. **Hindi unknown rate?** Exactly 0.000000, because byte fallback is on.
12. **Split ratios?** Hindi 96/2/2, Nepali 85/7.5/7.5, both by document.
13. **Hindi dedup?** SHA-1 exact only — 2,983 removed.
14. **Nepali dedup?** MinHash + LSH — 2,350 exact, 49,734 near.
15. **MinHash config?** 100 permutations, 20 bands × 5 rows, 5-char shingles, Jaccard 0.55.
16. **Architecture?** 6 layers, 8 heads, d=512, d_ff=2048, context 512, pre-norm, GELU.
17. **Parameters?** 23,273,472 (H) and 24,297,472 (L).
18. **Non-embedding parameters?** 18,915,328 — identical for both.
19. **Positional encoding?** Learned absolute; sinusoidal and none also implemented.
20. **Weight tying?** Yes — saves 4,096,000 parameters for Model H.
21. **Why √d_k?** d_k = 64, so dot products have std ≈ 8; without scaling softmax saturates and gradients vanish.
22. **Causal mask leakage?** Exactly 0.000e+00, verified by a test that also checks the converse.
23. **Optimiser?** AdamW, β = (0.9, 0.95), weight decay 0.1 on matrices only.
24. **Learning rate?** 6e-4 peak, 500 warmup, cosine to 6e-5.
25. **Effective batch?** 32 × 2 × 512 = 32,768 tokens per step.
26. **Training tokens?** 475,791,360 (H) and 473,169,920 (L).
27. **Hardware and time?** Kaggle Tesla T4, 10.99 hours each.
28. **Test perplexity?** 27.41 (H), 46.68 (L).
29. **Test bits-per-byte?** 0.5382 (H), 0.4846 (L) — the ranking reverses.
30. **Why does it reverse?** Bytes per token: 8.88 vs 11.44.
31. **Best decoding setting?** T = 1.0 — best chrF++ (16.29 / 18.05) and near-zero repetition.
32. **Greedy repetition rate?** 0.598 (H), 0.577 (L).
33. **Most trustworthy generation metric?** chrF++, because character n-grams survive morphology.
34. **Mean attention entropy?** 3.2047 (H), 3.0898 (L) against a uniform baseline of 5.2565.
35. **Most interesting head?** Layer 3 previous-token head in **both** models — entropy 0.06 @ distance 1.3 and 0.14 @ 1.1.
36. **Reasoning dataset size?** 20,000 / 2,000 / 2,000 per language, four task families.
37. **Leakage control?** 40/10/10 names and 14/4/4 templates, both disjoint, asserted in the generator.
38. **Finetuning LR?** 6e-5, one tenth of pretraining.
39. **Loss masking?** Answer tokens only, `ignore_index=-1` over the prompt.
40. **Why length-grouped batches?** Padding would shift position indices under learned absolute positions.
41. **Exact match after finetuning?** 35.15% (H), 29.45% (L), from 0.00%.
42. **Forced choice?** 36.35% (H), 39.30% (L) — chance is 41.67%.
43. **What does that mean?** Finetuning taught the answer format, not the comparison.
44. **Worst task?** `transitive3` — 12.40% and 13.40% exact match, below chance.
45. **Why below chance?** The model picks the twice-mentioned middle entity 72.6% of the time, and the middle entity is never correct.
46. **Off-prompt answers?** 0.0–7.2% for H, 19.2–28.8% for L.
47. **What does that show?** Groundedness is the real resource-tier difference, not reasoning.
48. **Best finetuning checkpoint?** Step 60, by validation loss; validation rises after that.
49. **Attention after finetuning?** Slightly sharper and slightly longer-range; biggest shifts in middle layers.
50. **Can you claim causality there?** No — that needs head ablation, which I did not run.

---

## PART M — Final 1-Minute Conclusion

> If I had to summarise the project in one minute: I built two independent 23-million-parameter Transformers from scratch, one Hindi and one Nepali, with separate corpora, separate tokenizers and separate weights, and I held the script and the architecture constant so the comparison between them would actually mean something.
>
> The two results I would stand behind are both cases where a headline number was misleading. Hindi's 41% perplexity advantage mostly disappears under bits-per-byte, because the gap is largely tokenizer granularity rather than model quality. And finetuning raised exact match from 0% to 35%, but forced-choice accuracy stayed at chance, so the models learned the answer format rather than the comparison — the two-hop task actually falls below chance because the model picks the name that gets mentioned twice.
>
> What the resource tier really bought was not reasoning but groundedness: Model H answers from the candidates actually in its prompt 93–100% of the time, Model L only 71–81%.
>
> And the thing I would take into the next project is that two Phase 1 decisions — a missing byte fallback and an exact-match-only deduplicator — both stayed invisible for two phases and then caused concrete problems. Corpus and tokenizer construction are experimental choices, not setup.

---

## FINAL CHEAT SHEET

### Project in one sentence
Two independent ~23M-parameter decoder-only Transformers built from scratch for Hindi and Nepali, with script and architecture held constant so that the resource-tier comparison isolates data and tokenizer effects.

### Phase 1 — key points
1. Hindi 505,852,028 tokens / 704,101 docs; Nepali 635,204,333 / 1,918,265.
2. Manual share 30.66% and 26.41% against a 20% floor.
3. Normalisation = NFC + strip zero-width/control + collapse whitespace + collapse blank lines; ZWNJ/ZWJ **kept**.
4. Language ID is three-valued, so "unsure" is not "wrong".
5. Hindi dedup = SHA-1 exact only (2,983). Nepali = MinHash+LSH (2,350 + 49,734) + decontamination (17,111).
6. MinHash: 100 perms, 20 bands × 5 rows, 5-char shingles, Jaccard 0.55.
7. Splits by **document**: H 96/2/2 seed 42; L 85/7.5/7.5 seed 20260817, held out before filtering.
8. Hindi Unigram 8,000: fertility 1.3956, unk 0.000000, byte fallback 0.019613.
9. Nepali BPE 10,000: fertility 1.4825, unk 0.003387, byte fallback **0.000000 = absent**.
10. 0 documents shared between the two corpora.

### Phase 2 — key points
1. 6 layers, 8 heads, d=512, d_ff=2048, context 512, dropout 0.1, pre-norm, GELU, tied embeddings, learned absolute positions.
2. Params 23,273,472 (H) / 24,297,472 (L); **non-embedding 18,915,328 for both**.
3. Block = 3,152,384 parameters; 6 blocks + final LayerNorm = 18,915,328.
4. Weight tying saves 4,096,000 (~15%).
5. Attention: QKᵀ/√64, mask with −∞ before softmax, 8 heads of 64.
6. Causal mask leakage exactly 0.000e+00; the test checks the converse too.
7. AdamW β=(0.9,0.95), wd 0.1 on matrices, 6e-4 peak, 500 warmup, cosine to 6e-5, clip 1.0 after unscale.
8. 32,768 tokens/step; ~475M tokens; 10.99 h on a T4; stopped at 14,520 / 14,440 of 15,000.
9. Checkpoints carry weights + optimizer + scheduler + step + best val + tokens + RNG + config.
10. Test PPL 27.41 / 46.68; BPB 0.5382 / 0.4846 — **ranking reverses**.
11. Bytes per token 8.875 / 11.441 — the mechanism.
12. Generation best at T=1.0: chrF++ 16.29 / 18.05, repetition ~0.
13. Greedy repetition 0.598 / 0.577 — a property of the decoding rule, not a bug.
14. Attention entropy 3.2047 / 3.0898 vs uniform baseline 5.2565; distance 62.36 / 54.51.
15. Layer-3 previous-token head in both models, independently.

### Phase 3 — key points
1. Four families: `compare2`, `numeric2`, `numeric3`, `transitive3`, 5,000 train each.
2. 20,000 / 2,000 / 2,000 per language.
3. Names 40/10/10, templates 14/4/4, both disjoint and **asserted**.
4. Finetune from each model's own checkpoint, tokenizer fixed, LR 6e-5, 2 epochs, batch 32, eval every 20 steps, seed 20260914.
5. Answer-token-only loss; length-grouped batches (positions would shift under padding).
6. Exact match 0.00% → 35.15% (H) and 29.45% (L).
7. Forced choice 39.80 → 36.35% (H), 38.80 → 39.30% (L); **chance 41.67%**.
8. Answer-token loss 5.2439 → 0.9408 and 5.5367 → 1.4351.
9. `transitive3` below chance: 12.40% / 13.40%.
10. Middle-entity selection 72.6% (H) where the middle entity is never correct.
11. Recency bias on one-hop: 62.0% / 65.0% to the second-named entity.
12. Off-prompt 0.0–7.2% (H) vs 19.2–28.8% (L) — the groundedness gap.
13. Best val at step 60; final train 0.4441 / 0.6048, final val 3.5188 / 3.5911.
14. Run A → Run B: eval grid 100 → 20 steps raised L from 14.90% to 29.45%.
15. Attention after finetuning: entropy 1.9102→1.8691 (H), 1.8327→1.7833 (L); distance 10.303→10.360 and 9.415→10.157.

### Must-memorise numbers

| Quantity | Model H | Model L |
|---|---:|---:|
| Corpus tokens | 505,852,028 | 635,204,333 |
| Documents | 704,101 | 1,918,265 |
| Manual share | 30.66% | 26.41% |
| Tokenizer / vocab | Unigram 8,000 | BPE 10,000 |
| Fertility | 1.3956 | 1.4825 |
| Unknown rate | 0.000000 | 0.003387 |
| Train tokens | 485,280,741 | 519,906,359 |
| Total parameters | 23,273,472 | 24,297,472 |
| Non-embedding | 18,915,328 | 18,915,328 |
| Tokens trained | 475,791,360 | 473,169,920 |
| Evaluated checkpoint | step 14,500 | step 14,000 |
| Test perplexity | 27.41 | 46.68 |
| Test BPB | 0.5382 | 0.4846 |
| Bytes per token | 8.875 | 11.441 |
| Attention entropy | 3.2047 | 3.0898 |
| Attention distance | 62.36 | 54.51 |
| Exact match after FT | 35.15% | 29.45% |
| Forced choice after FT | 36.35% | 39.30% |

Constants: uniform entropy baseline **5.2565**; chance rate **41.67%**; context **512**; effective batch **32,768 tokens**.

### Formulas

| Formula | Symbols | Why | Tiny example |
|---|---|---|---|
| `Attention = softmax(QKᵀ/√d_k + M)V` | Q,K,V projections; d_k=64; M additive mask | Weighted blend of values by query–key relevance | scores [2,1,0] → [0.67,0.24,0.09] |
| `H = −Σ p log p` | p = attention weights | Entropy: how diffuse a head is | uniform over 2 → 0.69 nats |
| `L = −(1/N) Σ log p(x_t | x_<t)` | N tokens | Cross-entropy objective | — |
| `PPL = exp(L)` | L in nats | Effective branching factor | exp(3.3109)=27.41 |
| `bits = L / ln 2` | ln 2 = 0.6931 | Convert nats to bits | 3.3109/0.6931=4.7766 |
| `BPB = bits per token / bytes per token` | — | Tokenizer-neutral comparison | 4.7766/8.875=0.5382 |
| `J(A,B) = |A∩B| / |A∪B|` | shingle sets | Near-duplicate similarity | 3/7 = 0.43 |
| `params/block = 4(d²+d) + 2d·d_ff + d_ff + d + 4d` | d=512, d_ff=2048 | Parameter budgeting | = 3,152,384 |

### Top 10 things I must NOT claim
1. That Hindi used MinHash. It did not — SHA-1 exact only.
2. That the models perform reasoning. Forced choice is at or below chance.
3. That Model L is a better language model because its BPB is lower. BPB is not corpus-neutral.
4. That attention analysis proves any head does the comparison. It is correlational.
5. That the ablation proves causal masking leaks position. Consistent with, not proven.
6. That either pretraining run converged. Both stopped early with loss still falling.
7. That exact match improving means capability improved. Loss fell while forced choice fell too.
8. Any wall-clock comparison between the baseline and the ablation — different GPUs.
9. That the corpora are free of train/test leakage. Hindi's is unverified.
10. That the manual-share percentages are like-for-like. Different weighting per language.

### Final conclusion
Hold the script and the architecture constant, and what is left is data and tokenizer — and that is where the differences turned out to live. The perplexity gap is mostly tokenizer granularity. The reasoning gain is mostly answer formatting. The resource tier shows up as groundedness, not reasoning. And the two Phase 1 decisions that caused real problems two phases later are the argument for treating corpus and tokenizer construction as experiments rather than setup.

# Phase 1 — Dataset Statistics

Hindi frozen measurement time: **2026-08-18, 11:24 IST**, read directly
from pipeline provenance files (`data/raw/*/metadata.jsonl`,
`data/metadata/clean_stats.json`, `data/splits/*_manifest.json`). Nepali
measurement time: **2026-08-20, 22:34 IST** (its pipeline's own
`corpus_stats.json` timestamp), read from
`nepali/data/final/corpus_stats.json` and `split_stats.json`. None are
hand-entered or estimated, except where explicitly marked
`NOT YET MEASURED`.

Hindi's manual vs. downloaded split is the `source_type` field carried on
every document from ingestion through to the cleaned corpus, computed by
`scripts/validate_phase1.py`, **char-weighted** over the cleaned corpus.
Nepali's pipeline never added per-lane character tracking; its manual
share below is **token-weighted** instead — a real methodology difference
between the two languages' pipelines, not a like-for-like comparison.

---

## Hindi (Model H) — measured

### Raw corpus

| | Docs | Characters |
|---|---:|---:|
| Manual | 229,011 | 540,367,035 |
| Downloaded | 544,595 | 1,299,990,112 |
| **Total raw** | **773,606** | **1,840,357,147** |

Manual share of raw characters: **29.36%**.

Manual sources: Gadya Kosh (prose wiki, exhausted), Kavita Kosh (poetry
wiki), Patrika (patrika.com newspaper archive), a self-extracted Hindi
Wikipedia XML dump (`hi_wikipedia_dump`, 160,591 docs / 234.1M chars — the
single largest manual contributor), and scattered archive.org items
(Rajya Sabha/Lok Sabha debate transcripts, religious texts, misc books).
The HF-mirrored Hindi Wikipedia parquet (`wikimedia/wikipedia:20231101.hi`,
163,091 docs / 253.3M chars) has been moved to
`data/raw/_quarantined_hf_hiwiki_mirror/` since it is now redundant with the
self-extracted dump and does not count as manual collection.

Downloaded sources: `HuggingFaceFW/fineweb-2:hin_Deva`,
`ai4bharat/sangraha:unverified/hin`,
`ai4bharat/sangraha:verified/hin:{web,pdf,speech}`.

### Cleaned corpus

From `hindi/data/metadata/clean_stats.json` (full non-resume run of
`clean_hindi.py`, launched after fixing a process-locking bug — see commit
history — that had corrupted an earlier, now-discarded partial run):

| Metric | Value |
|---|---:|
| Docs in | 773,606 |
| Docs cleaned (kept) | 704,103 |
| Dropped: malformed Unicode | 17 |
| Dropped: too short (<200 chars) | 63,132 |
| Dropped: low Devanagari ratio | 228 |
| Dropped: likely Bhojpuri (active reject filter) | 1,365 |
| Dropped: langdetect mismatch | 1,778 |
| Dropped: exact/near-duplicate | 2,983 |
| Characters in | 1,840,356,619 |
| **Characters cleaned** | **1,764,598,107** |
| Latin-script tokens stripped | 8,016,102 |
| Cleaning retention | 95.9% of raw chars |

**Language-ID caveat** (quoted verbatim from `clean_stats.json`'s
`bhojpuri_reject_caveat` field, since it governs the `likely_bhojpuri` drop
count above — a filter that predates and is independent of Bhojpuri's
current status as a superseded, no-longer-Model-L language; it still runs
as part of Hindi's own cleaning regardless): *"langdetect has no 'bho'
label, so it cannot itself catch Bhojpuri text misclassified as Hindi (they
share vocabulary and script). dropped_language.likely_bhojpuri is an active
reject filter (see clean_lib.marker_counts / language_ok below), not just a
count: any document where Bhojpuri lexical markers
(बा/बाटे/बाड़ें/एगो/गइल/भइल/कइल/etc., >=2 hits) outnumber Hindi markers
(है/हैं/एक/गया/हुआ/किया/etc.) is dropped before reaching data/cleaned/. This
is a coarse lexical heuristic, not a language model, so it will not catch
every case - documented as a known limitation, not a solved problem."*

That limitation is not hypothetical: see "Cross-language contamination"
below for two concrete cases it missed.

### Manual share (the hard 20% floor)

**28.65% manual** — from `validate_final.log`: 505,509,566 manual
characters / 1,764,596,782 total cleaned characters (the post-cross-
language-fix denominator; 1,764,598,107 was the pre-fix figure, see
"Cross-language contamination" below). Independently verified two ways:
`scripts/validate_phase1.py`'s own char-weighted computation from the
`source_type` provenance field, and a standalone recomputation directly
from `hindi/data/cleaned/metadata.jsonl` — both give 28.65% exactly.
Clears the 20% requirement with a comfortable margin — note this is the
*cleaned*-corpus, *char*-weighted figure; the *token*-weighted figure
from `scripts/count_tokens.py` (Step 3, chosen tokenizer vocab_size=8000)
is a separate, also-real measurement: 30.66% (155,104,651 manual tokens /
505,852,028 total tokens). Neither is a substitute for the other — they
measure different units — and both clear the floor independently. Also
not a substitute: the raw-pool 29.36% quoted above, measured on a
different (larger, uncleaned) denominator, which is not itself evidence
of passing the floor.

### Cross-language contamination (found, fixed — historical, from when Bhojpuri was Model L)

`scripts/validate_phase1.py`'s cross-language check, run while Bhojpuri
was this project's Model L, flagged 2 documents whose content-hash
(`dedup_key`) was shared between the Hindi and Bhojpuri cleaned corpora.
Inspection (`scripts/remove_cross_language_duplicates.py`) confirmed both
are genuinely Bhojpuri news text — containing Bhojpuri verb forms
("कहलन", "जइहें") not in the fixed marker list `clean_hindi.py`'s
contamination filter checks against, so they passed the langdetect+marker
filter into the Hindi corpus. Both were sourced as `hin` by
`ai4bharat/sangraha` but correctly labeled `bho_Deva` by
`HuggingFaceFW/fineweb-2` for the same underlying text — a labeling error
in the upstream public corpus, not introduced by this pipeline. Removed
from the Hindi cleaned corpus (kept in Bhojpuri, where they belong
linguistically): `hi_downloaded_sangraha_e6bb7e97f6` and
`hi_downloaded_sangraha_unverified_7ca84f1c71`. Post-removal: **704,101**
cleaned docs, 1,764,596,782 chars (704,103 / 1,764,598,107 before). This
correction is permanent and stays part of Hindi's cleaned corpus
regardless of which language is currently Model L. The current, active
cross-language safeguard is against **Nepali**, not Bhojpuri — see
"Cross-language check" under Nepali, below.

### Train/validation/test split

`split_hindi.py` (seed=42, document-level, ~96/2/2), regenerated after the
cross-language-duplicate removal above. Verified no-overlap by
`scripts/validate_phase1.py`.

### Token count (500M target)

**Measured — chosen tokenizer (vocab_size=8,000; see
`tokenizer_analysis.md` for the full sweep, including why 10,000 was
initially chosen and then revised once 8,000 was measured):**

| Split | Docs | Characters | Tokens |
|---|---:|---:|---:|
| train | 675,937 | 1,692,881,268 | 485,280,741 |
| validation | 14,082 | 36,497,051 | 10,484,640 |
| test | 14,082 | 35,218,463 | 10,086,647 |
| **Total** | **704,101** | **1,764,596,782** | **505,852,028** |

**505,852,028 tokens — 101.2% of the ~500M target**, exceeding it.
30.66% manual by token count (155,104,651 manual / 505,852,028 total),
clearing the 20% floor at the token level as well as the char level
(28.65%, measured earlier).

---

## Nepali (Model L) — measured

Nepali's pipeline (`nepali/scripts/`, stage-based S00–S14, its own
`corpuslib`) is structurally different from Hindi's single-pass cleaner —
see `phase1_report.md` §3 for the full stage-by-stage breakdown. It gates
on two **absolute** token targets rather than a manual-share ratio: manual
≥ 100,000,000 tokens, downloaded ≥ 400,000,000 tokens.

### Raw → final corpus

| | Docs | Words | Tokens |
|---|---:|---:|---:|
| Manual | 337,783 | 116,207,907 | 167,787,251 |
| Downloaded | 1,580,482 | 319,077,639 | 467,417,082 |
| **Total (final)** | **1,918,265** | **435,285,546** | **635,204,333** |

**Manual sources** (final, post-cleaning):

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

`news_onlinekhabar` alone supplies 91.6% of manual tokens — the manual
source with the largest scrapable archive; the pipeline's growth loop
(`auto_grow_and_tokenize.py`) raised its collection target specifically
because the other manual sources are genuinely smaller pools for Nepali
than their Hindi equivalents.

**Downloaded sources** (final, post-cleaning):

| Source | Docs | Words | Tokens |
|---|---:|---:|---:|
| ai4bharat/sangraha:verified | 579,743 | 173,902,717 | 258,462,424 |
| ai4bharat/sangraha:unverified | 341,901 | 99,767,699 | 141,886,279 |
| ai4bharat/IndicCorpV2 | 658,838 | 45,407,223 | 67,068,379 |

### Cleaning / retention funnel

From `nepali/data/final/corpus_stats.json`'s `retention_funnel`:

| Stage | Docs in | Docs out | Retention |
|---|---:|---:|---:|
| S03 normalise | 2,655,504 | 2,655,504 | 100.00% |
| S04 language ID | 2,655,504 | 2,649,643 | 99.78% |
| S07 train-pool split-off | 2,649,643 | 2,251,378 | 84.97% |
| S05 heuristic filters | 2,251,378 | 2,245,354 | 99.73% |
| S06 perplexity filter | 2,245,354 | 2,178,418 | 97.02% |
| S08 dedup (MinHash/LSH) | 2,178,418 | 2,126,334 | 97.61% |
| S09 decontaminate | 2,126,334 | 2,109,223 | 99.20% |
| S10 balance lanes (train only) | 2,109,223 | 1,520,000 | 72.06% |

Deduplication (S08) is MinHash/LSH (`num_perm=100, bands=20, rows=5,
shingle_size=5, threshold=0.55`), not Hindi's exact-hash matching —
removed 52,084 docs (2,350 exact, 49,734 near); 21,273 cross-lane
duplicate pairs resolved to the manual copy. Decontamination (S09, no
Hindi equivalent) removed 17,111 train documents matching held-out
content; a rescan found 0 remaining matches. Lane balancing (S10) trimmed
589,223 downloaded documents to bring train's manual share from 23.27% to
27.42%, without touching manual or loosening any filter.

### Manual share (the hard 20% floor)

**26.41% manual, token-weighted** — 167,787,251 manual tokens /
635,204,333 total tokens. No char-weighted figure exists for Nepali (its
pipeline never tracked per-lane characters — see the note at the top of
this file). Both of the pipeline's own absolute lane targets are also
independently cleared: manual 167,787,251 ≥ 100,000,000; downloaded
467,417,082 ≥ 400,000,000.

### Cross-language check (Nepali vs. Hindi)

Run via `scripts/check_nepali_hindi_overlap.py`: every one of Nepali's
1,918,265 final documents had its `dedup_key` (identical algorithm to
`clean_lib.dedup_key`, computed fresh) checked against the full set of
704,101 dedup_keys already computed for Hindi's cleaned corpus.
**Result: 0 shared documents.** Full run took 467.7 seconds. Nepali was
not checked against Bhojpuri, since Bhojpuri is not part of the final
two-language submission.

### Train/validation/test split

Nepali's split happens at S07 — val/test are held out from the full
document pool *before* S08 (dedup), S09 (decontamination), and S10 (lane
balancing) run on what remains, so those later stages never touch
documents that end up in eval. This differs from Hindi's approach
(document-level hash split applied once, after cleaning is complete).

| Split | Docs | Manual share |
|---|---:|---:|
| Train | 1,520,000 | 27.42% |
| Val | 199,428 | 23.71% |
| Test | 198,837 | 23.11% |
| **Total** | **1,918,265** | 26.70% (word) / 26.41% (token) |

Zero document-id overlap, confirmed by `nepali/data/final/split_stats.json`
(`zero_doc_id_overlap: true`, `flagged_splits: []`) and independently
re-checked by `scripts/validate_phase1_nepali.py`.

### Token count (500M target)

**Measured — chosen tokenizer (vocab_size=10,000, BPE; see
`tokenizer_analysis.md` for the full sweep):**

| Split | Docs | Tokens |
|---|---:|---:|
| train | 1,520,000 | 519,906,359 |
| val | 199,428 | 57,839,396 |
| test | 198,837 | 57,458,578 |
| **Total** | **1,918,265** | **635,204,333** |

**635,204,333 tokens — 127.0% of the ~500M target**, exceeding it by a
wider margin than Hindi's 101.2%. Unlike Bhojpuri (this project's original
Model L, which fell honestly short at 16.8% of target — see
`phase1_report.md` §6 for that history), Nepali's larger public-data
availability let its pipeline's growth loop (`auto_grow_and_tokenize.py`)
close the gap to both absolute lane targets without needing to document a
shortfall.

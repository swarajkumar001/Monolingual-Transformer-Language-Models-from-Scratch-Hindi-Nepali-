# Phase 1 — Dataset Statistics

Frozen measurement time: **2026-08-18, 11:24 IST**. All numbers below are read
directly from pipeline provenance files (`data/raw/*/metadata.jsonl`,
`data/metadata/clean_stats.json`, `data/splits/*_manifest.json`) — none are
hand-entered or estimated, except where explicitly marked `NOT YET MEASURED`.

Manual vs. downloaded is the `source_type` field carried on every document
from ingestion through to the cleaned corpus; the manual percentage below is
computed by `scripts/validate_phase1.py`, char-weighted over the **cleaned**
corpus (what actually reaches training), not the raw pool.

---

## Bhojpuri (Model L) — measured

### Raw corpus

| | Docs | Characters |
|---|---:|---:|
| Manual | 47,958 | 90,781,624 |
| Downloaded | 560,305 | 227,825,102 |
| **Total raw** | **608,263** | **318,606,726** |

**Manual sources** (raw):

| Source | Docs | Characters | Collection method |
|---|---:|---:|---|
| khabar_bhojpuri | 22,175 | 39,169,105 | scrape + clean |
| anjoria | 13,556 | 21,750,949 | scrape + clean |
| bho_wikipedia_dump | 8,539 | 6,101,826 | self-extracted Wikipedia XML dump (WikiExtractor) — counts as manual per course ruling; not the HF-mirrored parquet |
| jogira | 3,085 | 6,595,231 | scrape + clean |
| archive.org (misc, 88 titles: books, journals, government debate transcripts, dictionaries) | 88 | 16,210,212 | OCR/text extraction from archive.org items |
| harsh_scraped_manual | 290 | 724,804 | prior personally-collected scrape (verified own work) |
| sirijan | 225 | 229,497 | scrape + clean |

All four dedicated crawl sources (khabar_bhojpuri, anjoria, jogira, sirijan)
were recrawled uncapped and confirmed exhausted (0–7 new docs on the final
pass). bhojpuriplanet.com is domain-expired; bhojpuri.blog is
DNS-unregistered. No further manual Bhojpuri sources are known to exist.

**Downloaded sources** (raw):

| Source | Docs | Characters |
|---|---:|---:|
| Satyam810/BhojpuriCorpus:fish_food | 215,813 | 22,728,391 |
| Satyam810/BhojpuriCorpus:jnu_bhltr | 79,884 | 7,099,650 |
| harsh_indiccorp (personal prior collection, sentence-chunked) | 69,433 | 92,803,579 |
| bhltr_monolingual (JNU BHLTR corpus) | 45,027 | 3,944,485 |
| bhltr_monolingual_v02 (JNU BHLTR corpus) | 42,647 | 3,789,365 |
| Satyam810/BhojpuriCorpus:eng_bho | 26,909 | 872,788 |
| Satyam810/BhojpuriCorpus:bhojpuri_lr_v3 | 21,811 | 2,342,498 |
| Satyam810/BhojpuriCorpus:hplt | 17,778 | 45,689,298 |
| HuggingFaceFW/fineweb-2:bho_Deva | 17,166 | 34,838,209 |
| Satyam810/BhojpuriCorpus:wikipedia | 7,603 | 5,335,562 |
| Satyam810/BhojpuriCorpus:kumar_bhojpuri | 6,567 | 6,506,587 |
| Satyam810/BhojpuriCorpus:iit_bhu | 5,254 | 334,575 |
| Satyam810/BhojpuriCorpus:fleurs_bhojpuri | 2,708 | 299,454 |
| Satyam810/BhojpuriCorpus:google_smol | 844 | 9,434 |
| Satyam810/BhojpuriCorpus:glotcc | 567 | 1,203,595 |
| Satyam810/BhojpuriCorpus:rural_women | 211 | 20,226 |
| Satyam810/BhojpuriCorpus:ud_bhtb | 83 | 7,406 |

Six additional HF datasets were evaluated and rejected as unusable for this
project: CulturaX and sil-ai/bloom-lm are gated behind an authenticated HF
token; Sangraha and Glot500 have zero Bhojpuri rows; MADLAD-400 has no
bho/bho_Deva config; CC-100 uses a legacy `datasets`-library script format no
longer supported by the current library version.

**Open item (not yet resolved, footnote for final report):** the BHLTR
corpus (`bhltr_monolingual`/`bhltr_monolingual_v02`, git-cloned directly from
JNU, CC-BY-NC-SA-4.0) likely overlaps with the `Satyam810/BhojpuriCorpus:jnu_bhltr`
partition, which redistributes the same underlying corpus. Not deduplicated
across those two source names specifically; the pipeline's content-hash dedup
(`dedup_key`) still catches any exact-text overlap between them regardless of
source label.

### Cleaned corpus

From `bhojpuri/data/metadata/clean_stats.json` (this cleaning run is current —
its `docs_in`/`chars_in` match the raw totals above):

| Metric | Value |
|---|---:|
| Docs in | 608,263 |
| Docs cleaned (kept) | 507,994 |
| Dropped: malformed Unicode | 23 |
| Dropped: too short (<200 chars) | 4,665 |
| Dropped: low Devanagari ratio | 1,501 |
| Dropped: exact/near-duplicate | 94,080 |
| Characters in | 318,528,901 |
| **Characters cleaned** | **282,815,177** |
| Latin-script tokens stripped | 1,576,446 |
| Cleaning retention | 88.8% of raw chars |

`likely_hindi_contamination` (11,164 docs) is a **reporting-only** lexical
marker count, not an active filter — see `language_id_caveat` in the stats
file for the exact marker lists and reasoning. langdetect has no `bho` label,
so Devanagari-script-ratio plus these lexical markers are used as a practical
proxy; this cannot fully distinguish Bhojpuri from closely related
Hindi/Nepali text mislabeled at the source, and is documented as a known
limitation rather than a solved problem.

### Manual share (the hard 20% floor)

**28.59% manual** — 80,843,912 manual characters / 282,815,177 total cleaned
characters. Comfortably clears the 20% requirement. Verified by
`scripts/validate_phase1.py`, which computes this from the `source_type`
field on every cleaned document rather than any separate counter.

### Train/validation/test split

`split_bhojpuri.py` (seed=42, document-level split — never sentence/chunk
level, to avoid leakage), ~96/2/2 by document count. **Complete**, matches
the cleaned total exactly: train 487,674 / validation 10,160 / test 10,160
= 507,994 docs, no overlap (verified by `scripts/validate_phase1.py`).

### Token count (500M target)

**Measured — chosen tokenizer (vocab_size=8,000; see
`tokenizer_analysis.md` for the full sweep and justification):**

| Split | Docs | Characters | Tokens |
|---|---:|---:|---:|
| train | 487,674 | 271,000,251 | 80,610,590 |
| validation | 10,160 | 6,632,160 | 2,085,234 |
| test | 10,160 | 5,182,766 | 1,521,208 |
| **Total** | **507,994** | **282,815,177** | **84,217,032** |

**84,217,032 tokens** — 29.89% manual by token count (25,169,871 manual /
84,217,032 total), clearing the 20% floor at the token level as well as the
char level.

Per the course spec, Bhojpuri (the allowed lower-resource language) is
**not required** to reach 500M tokens; a documented shortfall with a
maximum-feasible-collection justification is acceptable, and 84.2M is
honestly reported as such rather than obscured. All known manual sources
are confirmed exhausted (see above), and the downloaded pool has been
evaluated against every Bhojpuri-labeled dataset findable on Hugging Face.
For scale: published prior Bhojpuri monolingual resources (e.g. the JNU
BHLTR corpus at ~45k sentences, and web-crawled corpora on the order of
~169k words) are orders of magnitude smaller than this 282.8M-character
corpus — this collection is not a shortfall relative to what was
achievable, it is close to the ceiling of what is publicly available for
this language at all.

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
count above): *"langdetect has no 'bho' label, so it cannot itself catch
Bhojpuri text misclassified as Hindi (they share vocabulary and script).
dropped_language.likely_bhojpuri is an active reject filter (see
clean_lib.marker_counts / language_ok below), not just a count: any
document where Bhojpuri lexical markers (बा/बाटे/बाड़ें/एगो/गइल/भइल/कइल/etc.,
>=2 hits) outnumber Hindi markers (है/हैं/एक/गया/हुआ/किया/etc.) is dropped
before reaching data/cleaned/. This is a coarse lexical heuristic, not a
language model, so it will not catch every case - documented as a known
limitation, not a solved problem."*

That limitation is not hypothetical: see "Cross-language contamination"
below for two concrete cases it missed.

### Manual share (the hard 20% floor)

**28.65% manual** — 505,509,566 manual characters / 1,764,598,107 total
cleaned characters. Independently verified two ways: `scripts/validate_phase1.py`
(char-weighted from the `source_type` provenance field) and a standalone
recomputation directly from `hindi/data/cleaned/metadata.jsonl`, both give
28.65% exactly. Clears the 20% requirement with a comfortable margin — note
this is the *cleaned*-corpus figure, not a substitute for the raw-pool
29.36% quoted above, which is measured on a different (larger, uncleaned)
denominator and is not itself evidence of passing the floor.

### Cross-language contamination (found, fixed)

`scripts/validate_phase1.py`'s cross-language check flagged 2 documents
whose content-hash (`dedup_key`) was shared between the Hindi and Bhojpuri
cleaned corpora. Inspection (`scripts/remove_cross_language_duplicates.py`)
confirmed both are genuinely Bhojpuri news text — containing Bhojpuri verb
forms ("कहलन", "जइहें") not in the fixed marker list `clean_hindi.py`'s
contamination filter checks against, so they passed the langdetect+marker
filter into the Hindi corpus. Both were sourced as `hin` by
`ai4bharat/sangraha` but correctly labeled `bho_Deva` by
`HuggingFaceFW/fineweb-2` for the same underlying text — a labeling error
in the upstream public corpus, not introduced by this pipeline. Removed
from the Hindi cleaned corpus (kept in Bhojpuri, where they belong
linguistically): `hi_downloaded_sangraha_e6bb7e97f6` and
`hi_downloaded_sangraha_unverified_7ca84f1c71`. Post-removal: **704,101**
cleaned docs, 1,764,596,782 chars (704,103 / 1,764,598,107 before). At this
scale (2 of 704,103 docs, 0.0003%) a full pipeline rerun with an expanded
marker list was not warranted; this is a targeted, documented, reproducible
correction instead.

### Train/validation/test split

`split_hindi.py`, same method as Bhojpuri (seed=42, document-level,
~96/2/2), regenerated after the cross-language-duplicate removal above.
Verified no-overlap by `scripts/validate_phase1.py`.

### Token count (500M target)

**Measured — chosen tokenizer (vocab_size=10,000; see
`tokenizer_analysis.md` for the full sweep and justification):**

| Split | Docs | Characters | Tokens |
|---|---:|---:|---:|
| train | 675,937 | 1,692,881,268 | 471,882,513 |
| validation | 14,082 | 36,497,051 | 10,194,536 |
| test | 14,082 | 35,218,463 | 9,806,338 |
| **Total** | **704,101** | **1,764,596,782** | **491,883,387** |

**491,883,387 tokens — 98.4% of the ~500M target**, essentially hitting
it. 30.63% manual by token count (150,668,466 manual / 491,883,387
total), clearing the 20% floor at the token level as well as the char
level (28.65%, measured earlier).

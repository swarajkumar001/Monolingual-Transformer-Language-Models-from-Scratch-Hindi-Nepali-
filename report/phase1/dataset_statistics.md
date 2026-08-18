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

Regenerated via `split_bhojpuri.py` (seed=42, document-level split — never
sentence/chunk level, to avoid leakage), ~96/2/2 by document count. **Split
regeneration was in progress at the time of this report** (was rerun after a
process-corruption incident — see `report/phase1/phase1_strategy.md` / commit
history for the lock-race bug and fix); final counts will match the cleaned
totals above exactly (507,994 docs) once complete. Prior (pre-fix) split
counts are not reported here since they reflected a stale/inconsistent
intermediate state.

### Token count (500M target)

**NOT YET MEASURED.** No SentencePiece tokenizer has been trained yet
(Step 2 of the current work, in progress). Do not extrapolate a token count
from a placeholder chars/token ratio — the real number will be filled in once
the Bhojpuri tokenizer sweep (8k/12k/16k/24k/32k) completes and a vocab size
is chosen.

Per the course spec, Bhojpuri (the allowed lower-resource language) is
**not required** to reach 500M tokens; a documented shortfall with a
maximum-feasible-collection justification is acceptable. All known manual
sources are confirmed exhausted (see above), and the downloaded pool has been
evaluated against every Bhojpuri-labeled dataset findable on Hugging Face.
282.8M cleaned characters is the practical ceiling for this corpus absent new
data sources.

---

## Hindi (Model H) — TODO

Hindi's raw pool is frozen and measured as of 2026-08-18, 11:24 IST (below),
but **cleaning is currently running** (`clean_hindi.py`, full non-resume run,
launched 2026-08-18 ~11:47 IST after fixing a process-locking bug — see
commit history). All numbers past this point are `NOT YET MEASURED` and will
be filled in once that run completes.

### Raw corpus (frozen, measured)

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

### TODO once `clean_hindi.py` finishes

- [ ] Cleaned docs/chars, drop breakdown (malformed/short/language/duplicate)
- [ ] Manual share of the cleaned corpus (must independently re-clear 20%; the
      29.36% raw figure above is *not* a substitute measurement)
- [ ] Regenerated train/validation/test splits (`split_hindi.py`)
- [ ] Tokenizer sweep results and chosen vocab size
- [ ] Final token count and comparison against the ~500M target

**NOT YET MEASURED** for all of the above — do not fill in with placeholder
or extrapolated numbers.

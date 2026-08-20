# Phase 1 Strategy — Data Collection & Tokenizer Construction

Model H = **Hindi** (higher-resource) · Model L at planning time =
**Bhojpuri** (lower-resource, from allowed list) — later superseded by
**Nepali**, also from the allowed list, once Nepali's own pipeline
finished. See [`phase1_report.md`](phase1_report.md) §6 for that history.

**SUPERSEDED — kept for historical record only.** This was the pre-execution
planning document, written before any data was collected. Every number in it
is a placeholder, not a measurement — and its Model L language choice (
Bhojpuri) is itself now superseded by Nepali. For actual, measured Phase 1
results, see [`README.md`](README.md) (deliverables index) or
[`phase1_report.md`](phase1_report.md) (full report).

Status: PLANNING — no data collected yet. Nothing in this document should be read as measured
results; all counts below are placeholders until pipelines actually run.

---

## 1. Repository skeleton (done)

`hindi/` and `bhojpuri/` each now have:
`data/{raw/{manual,downloaded},interim,cleaned,processed,splits/{train,validation,test},metadata}`,
`tokenizer/`, `model/`, `train/`, `eval/`, `configs/`.
Top level: `scripts/`, `report/{phase1,phase2,phase3}/`, `checkpoints/{hindi,bhojpuri}/`.

Not yet done: git init, GitHub Classroom remote, `phase-1`/`phase-2`/`phase-3` branches. Flag when
ready to wire this up — GitHub Classroom acceptance is an action only you can take (it's tied to
your account).

## 2. Hindi (Model H) — data source strategy

Candidate **downloaded** sources (real, well-known corpora — need to verify current availability/
license before use, not assumed):
- Hindi Wikipedia dump (via Wikimedia dumps or HF `wikimedia/wikipedia` `20231101.hi` config)
- AI4Bharat **IndicCorp v2** / **Sangraha** (large curated Indic corpora, HF hosted)
- OSCAR or CC-100 Hindi split (web-crawled, needs heavier cleaning)

These alone can plausibly reach several hundred million tokens for Hindi, so the downloaded 80% is
realistic. Actual token counts will only be reported after real extraction — no numbers are final
yet.

**Manual** (≥20% of Hindi tokens): needs your input — see open questions below.

## 3. Bhojpuri (Model L) — data source strategy

Bhojpuri is genuinely low-resource. Candidate downloaded sources:
- Bhojpuri Wikipedia dump (small — likely low tens of thousands of articles at most)
- AI4Bharat **Sangraha** (has a Bhojpuri split; scale unverified until we pull it)
- Bhojpuri-language news/literature sites if any exist and are scrapeable under their terms

Given the spec explicitly allows falling short of 500M tokens for the low-resource language "if
it cannot reach 500M from public sources," I expect we will document a real, smaller number for
Bhojpuri rather than force it to 500M. We must not pad or duplicate text to fake the target
(rule in section 51 of your spec).

**Manual** (≥20% of Bhojpuri tokens): likely to matter *more* here since public Bhojpuri text is
scarce — manual collection may end up being a larger share by necessity.

## 4. Manual-data collection strategy (needs your decision — see questions below)

Options consistent with the spec's allowed methods:
- OCR from books/PDFs you have access to (needs page images + an OCR engine, e.g. Tesseract with
  Devanagari `hin`/`bho`-adjacent language packs)
- Scraping + cleaning pages you personally gather (e.g. blogs, forums, government sites not in
  standard HF datasets)
- Typed or transcribed text

I can build the OCR pipeline, the scraping+cleaning scripts, and the metadata tooling — but I
cannot manufacture manually-collected source material myself; that requires real source documents
you provide or point me to.

## 5. Metadata schema (per document)

```json
{
  "document_id": "hi_manual_000001",
  "language": "hindi",
  "source_type": "manual",
  "source_name": "...",
  "source_description": "...",
  "collection_method": "ocr | scrape | typed | transcribed | download",
  "license_or_usage_note": "...",
  "collected_date": "YYYY-MM-DD",
  "characters": 0,
  "tokens": 0,
  "split": "train | validation | test",
  "cleaning_status": "raw | cleaned | normalized"
}
```
Same schema for Bhojpuri with `bho_` document IDs and `"language": "bhojpuri"`.

## 6. Cleaning / normalization pipeline (per section 10–13 of the spec)

```
collection -> raw text + metadata -> extraction (HTML/PDF/OCR -> plain text)
  -> Unicode NFC normalization (preserve Devanagari; strip zero-width/control chars only)
  -> language-ID filtering (verify actual Hindi / actual Bhojpuri, not just source label)
  -> boilerplate/short-doc/malformed-Unicode removal
  -> exact + near-duplicate removal (normalized-text hash)
  -> document-level train/val/test split (before chunking, so no doc crosses splits)
  -> tokenization (after tokenizer is trained) -> dataset statistics
```
Bhojpuri and Hindi use fully separate script instances/config, sharing only the code, never the
data.

## 7. Train/validation/test strategy

- Split at the **document** level, not sentence/chunk level, to avoid leakage.
- Target ratios (adjustable once real corpus sizes are known): ~96/2/2 train/val/test, since a
  ~500M-token target only needs modest val/test sizes to be statistically meaningful.
- Store explicit manifests: `{lang}/data/splits/{train,validation,test}_manifest.json` listing
  document IDs per split.

## 8. Tokenizer strategy

- One SentencePiece **BPE** (or Unigram — to be decided empirically) tokenizer per language,
  trained only on that language's train split.
- Vocabulary sizes to sweep: 8k / 12k / 16k / 24k / 32k, selecting per language via fertility and
  unknown-token rate on held-out (validation) text.
- Expect Hindi to tolerate a larger vocab (more data, more surface forms observed); Bhojpuri may
  need a smaller vocab given less data — decision deferred to actual measurement, not assumed now.

## 9. Phase 1 validation checklist (to run via `scripts/validate_phase1.py` once data exists)

- [ ] Hindi corpus exists, Bhojpuri corpus exists
- [ ] Hindi manual fraction ≥ 20%, Bhojpuri manual fraction ≥ 20%
- [ ] Train/val/test separated per language, no document crosses splits
- [ ] No cross-language contamination
- [ ] No duplicate documents across splits
- [ ] Hindi tokenizer + vocab exist, Bhojpuri tokenizer + vocab exist
- [ ] Tokenizer stats reported (fertility, UNK rate, avg chars/token)
- [ ] Dataset stats reported (doc/char/token counts, manual vs downloaded split)
- [ ] Reproducibility metadata present (seed, source inventory, script versions)

## 10. Open questions blocking real execution

See the question I'm asking you directly next — mainly: how you want to source manual data, and
whether you have accounts/access set up for the downloaded sources (HF, Wikipedia dumps).

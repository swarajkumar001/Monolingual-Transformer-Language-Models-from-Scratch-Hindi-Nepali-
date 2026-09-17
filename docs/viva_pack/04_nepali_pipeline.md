# Phase 1 - the Nepali stage pipeline (differs from Hindi)

Part of the LMA individual project viva pack. Line numbers below match the files in the repository exactly, so any line cited here can be opened directly in the IDE.

Files in this pack:

- `nepali/configs/language.yaml`
- `nepali/scripts/s03_normalize.py`
- `nepali/scripts/s04_langid.py`
- `nepali/scripts/s07_holdout.py`
- `nepali/scripts/s08_dedup.py`
- `nepali/scripts/s09_decontaminate.py`
- `nepali/scripts/s11_finalize_splits.py`
- `nepali/scripts/s12_train_tokenizer.py`


## nepali/configs/language.yaml

```yaml
   1  # =============================================================================
   2  # Nepali (Model L) -- language configuration
   3  # =============================================================================
   4  # Nepali is on the assignment's allowed lower-resource list. Every threshold
   5  # below is looser than the Hindi equivalent for one reason: Hindi has surplus
   6  # data, so a discarded document costs nothing, whereas here every discarded
   7  # document is unrecoverable.
   8  
   9  lang: nepali
  10  lang_code: ne                 # ISO 639-1, and the fastText lid.176 label
  11  model_role: L                 # lower-resource model in the H/L contrast
  12  display_name: Nepali
  13  
  14  # -----------------------------------------------------------------------------
  15  # Justification for the report
  16  # -----------------------------------------------------------------------------
  17  justification: >
  18    Nepali appears on the assignment's permitted lower-resource list. It is the
  19    official language of its own country, so it has a live news ecosystem
  20    (Kantipur, Setopati, Onlinekhabar, Nagarik) that keeps producing native text,
  21    a 30,041-article Wikipedia, and a dedicated `nep` subset in Sangraha and
  22    IndicCorp v2. That is enough to build a real corpus while remaining
  23    genuinely lower-resource: Nepali Wikipedia is 5.7x smaller than Hindi's
  24    (30,041 vs 170,826 articles) and its Sangraha verified split has 25 shards
  25    against Hindi's 100. The resulting data-scale gap is the contrast Phase 3
  26    asks us to explain.
  27  
  28    The principal risk is language identification. Nepali shares Devanagari with
  29    Hindi, Maithili, Bhojpuri, Doteli and Sanskrit, and fastText confuses them:
  30    a probe sentence about Kathmandu municipal government scored ne=0.891 with
  31    Doteli at 0.064. The pipeline therefore scores the margin between Nepali and
  32    the best competing Devanagari label rather than raw confidence alone.
  33  
  34  # -----------------------------------------------------------------------------
  35  # Targets
  36  # -----------------------------------------------------------------------------
  37  targets:
  38    # Overridden 2026-08-20: two independent ABSOLUTE token targets, not a
  39    # ratio+cap pair derived from an overall total -- see hindi/configs/language.yaml
  40    # for the full rationale. The brief allows Model L to fall short provided the
  41    # exact count is reported and any shortfall justified; that applies per-lane
  42    # now, not to an overall-total aspiration.
  43    #
  44    # manual_tokens_target: minimum final manual-lane tokens. Collection is grown
  45    # (more/deeper sources in sources.yaml) until met; manual is never trimmed.
  46    #
  47    # downloaded_tokens_target: s10_balance_lanes trims the DOWNLOADED lane to
  48    # approximately this many tokens, full stop. Measured 2026-08-17: the
  49    # previous ratio-forced version cut downloaded from 256M raw words to 41.1M
  50    # just to match a manual lane collected at only 5.8% of the corpus (15.8M of
  51    # 271.8M raw words). Composition (this target) and manual volume (grown at
  52    # collection time) are handled separately. corpuslib/rawtrim.py keeps raw
  53    # disk usage in line with this same target so collected-but-unusable excess
  54    # doesn't sit on disk.
  55    manual_tokens_target: 100_000_000
  56    downloaded_tokens_target: 400_000_000
  57  
  58    assumed_tokens_per_word: 1.90   # Nepali morphology fragments slightly more than Hindi
  59  
  60    # Larger than Hindi's: the conservative filters keep more, but the smaller
  61    # source pool means collection must reach deeper to find the same volume.
  62    collection_headroom: 1.6
  63  
  64  # -----------------------------------------------------------------------------
  65  # Normalisation
  66  # -----------------------------------------------------------------------------
  67  normalization:
  68    form: NFC
  69    # CRITICAL DIFFERENCE FROM HINDI: joiners are NOT stripped. ZWJ is
  70    # linguistically load-bearing in Nepali conjuncts -- पुर्‍याउने contains a
  71    # legitimate ZWJ, and removing it corrupts the word. The Hindi config sets
  72    # this true because Hindi web text uses joiners spuriously; Nepali does not.
  73    strip_joiners: false
  74    normalize_digits: true
  75    collapse_whitespace: true
  76    redact_pii: true
  77  
  78  # -----------------------------------------------------------------------------
  79  # Language identification
  80  # -----------------------------------------------------------------------------
  81  # Looser confidence than Hindi, because fastText was trained on far less Nepali
  82  # and systematically under-confident on it -- but a WIDER margin requirement,
  83  # because the failure mode here is absorbing Hindi text into the Nepali corpus,
  84  # which would destroy the independence the assignment requires.
  85  langid:
  86    target_label: ne
  87    min_confidence: 0.55
  88    min_margin: 0.22
  89    min_chars: 150              # slightly stricter: short Nepali is easily confused
  90    min_devanagari_ratio: 0.60
  91    max_error_rate: 0.01
  92  
  93    expected_retention_min: 0.35   # lower band than Hindi; LID is harder here
  94    # 0.999, not 0.99: Hindi hit a false BLOCKED at 99.87% retention because its
  95    # downloaded lane (Sangraha/IndicCorp) is already language-filtered before
  96    # collection, so near-total retention there is correct, not a sign LID
  97    # stopped discriminating (see hindi/configs/language.yaml). Nepali's
  98    # downloaded lane is built the same way, so the same ceiling applies here
  99    # pre-emptively rather than waiting to hit it on a real run.
 100    expected_retention_max: 0.999
 101  
 102  # -----------------------------------------------------------------------------
 103  # Splits
 104  # -----------------------------------------------------------------------------
 105  # Wider val/test than Hindi's 96/2/2. With a smaller corpus a 2% test split can
 106  # be too small for the held-out perplexity and BPB numbers in Phase 2 to be
 107  # stable, and noisy eval on Model L would undermine the whole H/L comparison.
 108  # The wider split costs Model L nothing: neither model completes a full epoch,
 109  # so the train pool it never reaches is not an advantage forgone, and both
 110  # models are evaluated on an identical 1,999,872-token budget per split.
 111  splits:
 112    train: 0.85
 113    val: 0.075
 114    test: 0.075
 115    by: document
 116    seed: 20260817
```


## nepali/scripts/s03_normalize.py

```python
   1  #!/usr/bin/env python3
   2  """
   3  Step 03 -- normalise every collected document.
   4  
   5      python hindi/scripts/s03_normalize.py
   6      python hindi/scripts/s03_normalize.py --limit 5000     # smoke test
   7  
   8  NFC normalisation, invisible-character stripping, optional ZWJ/ZWNJ and digit
   9  normalisation, whitespace hygiene and PII redaction -- all driven by
  10  ``configs/language.yaml: normalization`` so the two languages can differ
  11  (Hindi strips joiners, Nepali does not -- ZWJ is load-bearing inside Nepali
  12  conjuncts such as ``पुर्‍याउने``).
  13  
  14  This step never drops a document. Its only job is to put every document into
  15  canonical form; deciding whether a document is *good* is S05's job, on text
  16  this step has already cleaned up. Dropping here would make retention numbers
  17  from both stages impossible to attribute to one cause.
  18  
  19  ``doc_id`` is recomputed from the normalised text (schema.py's own contract:
  20  "doc_id is a content hash of the *normalized* text" -- true from this step
  21  onward, not from collection, since no real normalisation had run yet at
  22  collection time). The original collection-time id is kept at
  23  ``meta.raw_doc_id`` for traceability.
  24  """
  25  
  26  from __future__ import annotations
  27  
  28  import argparse
  29  import sys
  30  import time
  31  from pathlib import Path
  32  
  33  sys.path.insert(0, str(Path(__file__).resolve().parent))
  34  
  35  from corpuslib import runlog, schema, stageio  # noqa: E402
  36  from corpuslib.config import load_language_config  # noqa: E402
  37  from corpuslib.textnorm import normalize, redact_pii  # noqa: E402
  38  
  39  STAGE_NAME = "s03_normalized"
  40  
  41  
  42  def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  43      parser = argparse.ArgumentParser(description=__doc__)
  44      parser.add_argument("--limit", type=int, default=0, help="process at most N documents")
  45      return parser.parse_args(argv)
  46  
  47  
  48  def main(argv: list[str] | None = None) -> int:
  49      args = parse_args(argv)
  50      config = load_language_config()
  51      config.ensure_dirs()
  52      log = runlog.StepLogger(config.lang, config.events_path)
  53      started = time.time()
  54  
  55      norm_cfg = config.normalization
  56      form = norm_cfg.get("form", "NFC")
  57      strip_joiners = bool(norm_cfg.get("strip_joiners", False))
  58      normalize_digits = bool(norm_cfg.get("normalize_digits", False))
  59      collapse_whitespace = bool(norm_cfg.get("collapse_whitespace", True))
  60      do_redact_pii = bool(norm_cfg.get("redact_pii", False))
  61  
  62      print(
  63          f"S03 normalise ({config.display_name}) -- form={form} "
  64          f"strip_joiners={strip_joiners} normalize_digits={normalize_digits} "
  65          f"redact_pii={do_redact_pii}",
  66          flush=True,
  67      )
  68  
  69      docs_in = 0
  70      docs_out = 0
  71      errors = 0
  72      pii_counts = {"email": 0, "phone": 0}
  73      by_lane_in: dict[str, int] = {}
  74      by_lane_out: dict[str, int] = {}
  75  
  76      def records():
  77          nonlocal docs_in, docs_out, errors
  78          for record in stageio.iter_raw_documents(config.raw_dir):
  79              if args.limit and docs_in >= args.limit:
  80                  break
  81              docs_in += 1
  82              by_lane_in[record.get("lane", "?")] = by_lane_in.get(record.get("lane", "?"), 0) + 1
  83              try:
  84                  text = normalize(
  85                      record.get("text", ""),
  86                      form=form,
  87                      strip_joiners=strip_joiners,
  88                      normalize_digits=normalize_digits,
  89                      collapse_whitespace=collapse_whitespace,
  90                  )
  91                  if do_redact_pii:
  92                      text, counts = redact_pii(text)
  93                      pii_counts["email"] += counts["email"]
  94                      pii_counts["phone"] += counts["phone"]
  95  
  96                  record["meta"] = dict(record.get("meta") or {})
  97                  record["meta"]["raw_doc_id"] = record["doc_id"]
  98                  record["text"] = text
  99                  record["doc_id"] = schema.content_hash(text)
 100                  record["n_chars"] = len(text)
 101                  record["n_words"] = schema.count_words(text)
 102              except Exception as exc:  # noqa: BLE001 - a normalisation bug must not vanish
 103                  errors += 1
 104                  print(
 105                      f"  ERROR normalising doc {record.get('doc_id', '?')}: "
 106                      f"{type(exc).__name__}: {exc}",
 107                      file=sys.stderr,
 108                  )
 109                  continue
 110  
 111              docs_out += 1
 112              by_lane_out[record.get("lane", "?")] = by_lane_out.get(record.get("lane", "?"), 0) + 1
 113              if docs_out % 200_000 == 0:
 114                  print(f"  ...{docs_out:,} documents normalised", flush=True)
 115              yield record
 116  
 117      written = stageio.write_stage(config.interim_dir / STAGE_NAME, records())
 118      elapsed = time.time() - started
 119  
 120      payload = {
 121          "docs_in": docs_in,
 122          "docs_out": docs_out,
 123          "docs_written": written,
 124          "errors": errors,
 125          "by_lane_in": by_lane_in,
 126          "by_lane_out": by_lane_out,
 127          "pii_redacted": pii_counts,
 128          "seconds": round(elapsed, 1),
 129      }
 130      stageio.write_stage_manifest(config.interim_dir / STAGE_NAME, payload)
 131  
 132      ok = docs_in == docs_out == written and errors == 0
 133      print(f"  docs_in={docs_in:,}  docs_out={docs_out:,}  written={written:,}  errors={errors}")
 134      print(f"  PII redacted: {pii_counts['email']:,} emails, {pii_counts['phone']:,} phone numbers")
 135  
 136      log(
 137          "S03",
 138          f"Normalise ({config.display_name})",
 139          runlog.SUCCESS if ok else runlog.BLOCKED,
 140          (
 141              f"Normalised {docs_out:,} of {docs_in:,} documents; counts match, "
 142              f"zero documents dropped."
 143              if ok
 144              else (
 145                  f"docs_in ({docs_in:,}) != docs_out ({docs_out:,}) or {errors} errors -- "
 146                  f"normalisation must never drop documents; this is a bug, not a filter."
 147              )
 148          ),
 149          metrics=payload,
 150      )
 151      if not ok:
 152          print("BLOCKED: docs_in != docs_out (or errors occurred). This step must not drop documents.")
 153          return 1
 154      return 0
 155  
 156  
 157  if __name__ == "__main__":
 158      raise SystemExit(main())
```


## nepali/scripts/s04_langid.py

```python
   1  #!/usr/bin/env python3
   2  """
   3  Step 04 -- language identification over every document, including manual sources.
   4  
   5      python hindi/scripts/s04_langid.py
   6      python hindi/scripts/s04_langid.py --smoke     # 5,000 docs, inspect samples first
   7  
   8  fastText ``lid.176`` scores every document -- manual sources are not exempt.
   9  Trusting a source by its origin (the previous run's fallback when LID broke)
  10  is what let a NumPy incompatibility delete 94% of the corpus in silence for
  11  four hours: everything was "accepted" by construction because the classifier
  12  never ran.
  13  
  14  The decision uses the *margin* between the target label and the best
  15  competing Devanagari label (hi/ne/mai/bho/mr/sa/dty/...), not raw confidence
  16  -- see ``corpuslib.langid`` for why raw confidence alone does not separate
  17  Hindi/Nepali/Maithili reliably. Documents shorter than ``min_chars`` are
  18  never scored.
  19  
  20  This step **aborts** (nonzero exit, BLOCKED in the log) if retention falls
  21  outside ``langid.expected_retention_{min,max}`` from ``configs/language.yaml``,
  22  rather than logging SUCCESS on a run that quietly deleted most of the corpus.
  23  ``--smoke`` skips the abort (a 5,000-document sample is not a reliable
  24  retention estimate) so samples can be inspected before committing to a full
  25  run.
  26  """
  27  
  28  from __future__ import annotations
  29  
  30  import argparse
  31  import sys
  32  import time
  33  from collections import Counter
  34  from pathlib import Path
  35  
  36  sys.path.insert(0, str(Path(__file__).resolve().parent))
  37  
  38  from corpuslib import runlog, stageio  # noqa: E402
  39  from corpuslib.config import load_language_config  # noqa: E402
  40  from corpuslib.langid import LanguageIdentifier, LidErrorRate, LidUnavailable  # noqa: E402
  41  
  42  STAGE_IN = "s03_normalized"
  43  STAGE_OUT = "s04_langid"
  44  SAMPLES_DIR_NAME = "langid_samples"
  45  SAMPLE_CAP_PER_REASON = 20
  46  
  47  
  48  def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  49      parser = argparse.ArgumentParser(description=__doc__)
  50      parser.add_argument(
  51          "--smoke",
  52          action="store_true",
  53          help="process 5,000 documents and skip the retention-band abort",
  54      )
  55      parser.add_argument("--limit", type=int, default=0, help="process at most N documents")
  56      return parser.parse_args(argv)
  57  
  58  
  59  def main(argv: list[str] | None = None) -> int:
  60      args = parse_args(argv)
  61      config = load_language_config()
  62      config.ensure_dirs()
  63      log = runlog.StepLogger(config.lang, config.events_path)
  64      started = time.time()
  65  
  66      limit = args.limit or (5000 if args.smoke else 0)
  67      lid_cfg = config.langid
  68  
  69      stage_in_dir = config.interim_dir / STAGE_IN
  70      if not stageio.stage_manifest(stage_in_dir):
  71          print(f"BLOCKED: {stage_in_dir} has no stage manifest. Run s03_normalize.py first.")
  72          log("S04", f"Language ID ({config.display_name})", runlog.BLOCKED,
  73              "s03_normalized has not been run yet.")
  74          return 1
  75  
  76      print(f"S04 langid ({config.display_name}) -- loading fastText lid.176...", flush=True)
  77      try:
  78          identifier = LanguageIdentifier(
  79              target_label=lid_cfg.get("target_label", config.lang_code),
  80              min_confidence=float(lid_cfg.get("min_confidence", 0.65)),
  81              min_margin=float(lid_cfg.get("min_margin", 0.15)),
  82              min_chars=int(lid_cfg.get("min_chars", 120)),
  83              min_devanagari_ratio=float(lid_cfg.get("min_devanagari_ratio", 0.55)),
  84              max_error_rate=float(lid_cfg.get("max_error_rate", 0.01)),
  85          )
  86      except LidUnavailable as exc:
  87          print(f"BLOCKED: LID model unavailable: {exc}")
  88          log("S04", f"Language ID ({config.display_name})", runlog.BLOCKED, str(exc))
  89          return 1
  90      print("  self-test passed.", flush=True)
  91  
  92      docs_in = 0
  93      kept_count = 0
  94      reject_reasons: Counter[str] = Counter()
  95      # A handful of kept/dropped examples per reason, for the reviewable samples.
  96      samples_kept: dict[str, list[dict]] = {}
  97      samples_dropped: dict[str, list[dict]] = {}
  98  
  99      def bucket_reason(reason: str) -> str:
 100          # Collapse parametrised reasons ("narrow_margin_vs_hi:0.12") to their
 101          # family so sample files don't fragment into hundreds of near-empty ones.
 102          return reason.split(":", 1)[0]
 103  
 104      def records():
 105          nonlocal docs_in, kept_count
 106          try:
 107              for record in stageio.read_stage(stage_in_dir):
 108                  if limit and docs_in >= limit:
 109                      break
 110                  docs_in += 1
 111                  verdict = identifier.classify(record.get("text", ""))
 112                  reason_key = bucket_reason(verdict.reason)
 113  
 114                  if verdict.accepted:
 115                      kept_count += 1
 116                      record["meta"] = dict(record.get("meta") or {})
 117                      record["meta"].update(verdict.as_meta())
 118                      bucket = samples_kept.setdefault(reason_key, [])
 119                      if len(bucket) < SAMPLE_CAP_PER_REASON:
 120                          bucket.append(record)
 121                      if docs_in % 200_000 == 0:
 122                          print(f"  ...{docs_in:,} scored, {kept_count:,} kept", flush=True)
 123                      yield record
 124                  else:
 125                      reject_reasons[reason_key] += 1
 126                      bucket = samples_dropped.setdefault(reason_key, [])
 127                      if len(bucket) < SAMPLE_CAP_PER_REASON:
 128                          bucket.append(record)
 129                      if docs_in % 200_000 == 0:
 130                          print(f"  ...{docs_in:,} scored, {kept_count:,} kept", flush=True)
 131          except LidErrorRate as exc:
 132              print(f"BLOCKED mid-run: {exc}", file=sys.stderr)
 133              raise
 134  
 135      try:
 136          written = stageio.write_stage(config.interim_dir / STAGE_OUT, records())
 137      except LidErrorRate as exc:
 138          log("S04", f"Language ID ({config.display_name})", runlog.BLOCKED, str(exc),
 139              metrics={"docs_scored": docs_in, "docs_kept": kept_count})
 140          return 1
 141  
 142      samples_root = config.interim_dir / SAMPLES_DIR_NAME
 143      for reason_key in set(samples_kept) | set(samples_dropped):
 144          stageio.write_samples(
 145              samples_root, reason_key,
 146              samples_kept.get(reason_key, []), samples_dropped.get(reason_key, []),
 147              n=SAMPLE_CAP_PER_REASON,
 148          )
 149  
 150      elapsed = time.time() - started
 151      retention = written / docs_in if docs_in else 0.0
 152      ret_min = float(lid_cfg.get("expected_retention_min", 0.0))
 153      ret_max = float(lid_cfg.get("expected_retention_max", 1.0))
 154      in_band = ret_min <= retention <= ret_max
 155  
 156      payload = {
 157          "docs_in": docs_in,
 158          "docs_kept": written,
 159          "retention": round(retention, 4),
 160          "expected_retention_min": ret_min,
 161          "expected_retention_max": ret_max,
 162          "reject_reasons": dict(reject_reasons),
 163          "lid_error_rate": round(identifier.error_rate, 5),
 164          "lid_docs_scored": identifier.n_scored,
 165          "lid_docs_errored": identifier.n_errors,
 166          "seconds": round(elapsed, 1),
 167          "smoke": bool(args.smoke),
 168      }
 169      stageio.write_stage_manifest(config.interim_dir / STAGE_OUT, payload)
 170  
 171      print(f"  docs_in={docs_in:,}  kept={written:,}  retention={retention:.2%}")
 172      print(f"  reject reasons: {dict(reject_reasons)}")
 173      print(f"  LID error rate: {identifier.error_rate:.3%} (halt threshold {identifier.max_error_rate:.1%})")
 174      print(f"  samples written to {samples_root}")
 175  
 176      should_abort = (not in_band) and not args.smoke
 177      status = runlog.SUCCESS if in_band else (runlog.WARNING if args.smoke else runlog.BLOCKED)
 178      log(
 179          "S04",
 180          f"Language ID ({config.display_name})",
 181          status,
 182          (
 183              f"Retention {retention:.2%} of {docs_in:,} documents, inside the "
 184              f"expected [{ret_min:.0%}, {ret_max:.0%}] band."
 185              if in_band
 186              else (
 187                  f"Retention {retention:.2%} is outside the expected "
 188                  f"[{ret_min:.0%}, {ret_max:.0%}] band -- inspect "
 189                  f"{samples_root} before trusting this run."
 190              )
 191          ),
 192          metrics=payload,
 193      )
 194      if should_abort:
 195          print(
 196              f"BLOCKED: retention {retention:.2%} outside expected "
 197              f"[{ret_min:.0%}, {ret_max:.0%}]. Read {samples_root} before proceeding."
 198          )
 199          return 1
 200      return 0
 201  
 202  
 203  if __name__ == "__main__":
 204      raise SystemExit(main())
```


## nepali/scripts/s07_holdout.py

```python
   1  #!/usr/bin/env python3
   2  """
   3  Step 07 -- carve out val/test before any quality filtering touches them.
   4  
   5      python hindi/scripts/s07_holdout.py
   6  
   7  Despite the file number, this reads S04's output (normalised + language-ID'd,
   8  nothing more) and runs **before** S05/S06 in the actual data flow, not after.
   9  The reason is in the plan: held-out documents must pass through normalisation
  10  and LID only, because if the same heuristic/perplexity filters that select
  11  training data also curate the evaluation set, evaluation partly measures the
  12  filters' own preferences rather than genuine model quality. ``s05_heuristics``
  13  and ``s06_perplexity`` read this step's ``train_pool`` output, never S04's
  14  directly -- that is what keeps val/test unfiltered.
  15  
  16  Splitting is by **document**, never by line (a line split leaks context
  17  across train/eval), and is a deterministic hash of ``doc_id`` against the
  18  configured seed -- not a shuffle held in memory -- so the assignment is
  19  reproducible from the seed alone and the whole pool never needs to fit in
  20  RAM at once.
  21  """
  22  
  23  from __future__ import annotations
  24  
  25  import argparse
  26  import hashlib
  27  import sys
  28  import time
  29  from pathlib import Path
  30  
  31  sys.path.insert(0, str(Path(__file__).resolve().parent))
  32  
  33  from corpuslib import runlog, stageio  # noqa: E402
  34  from corpuslib.config import load_language_config  # noqa: E402
  35  
  36  STAGE_IN = "s04_langid"
  37  STAGE_TRAIN_POOL = "s07_train_pool"
  38  STAGE_VAL = "s07_val"
  39  STAGE_TEST = "s07_test"
  40  
  41  
  42  def split_bucket(doc_id: str, seed: int) -> float:
  43      """Deterministic pseudo-uniform value in [0, 1) for one document."""
  44      digest = hashlib.blake2b(f"{seed}:{doc_id}".encode("utf-8"), digest_size=8).digest()
  45      return int.from_bytes(digest, "big") / 2**64
  46  
  47  
  48  def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  49      parser = argparse.ArgumentParser(description=__doc__)
  50      parser.add_argument("--limit", type=int, default=0, help="process at most N documents")
  51      return parser.parse_args(argv)
  52  
  53  
  54  def main(argv: list[str] | None = None) -> int:
  55      args = parse_args(argv)
  56      config = load_language_config()
  57      config.ensure_dirs()
  58      log = runlog.StepLogger(config.lang, config.events_path)
  59      started = time.time()
  60  
  61      stage_in_dir = config.interim_dir / STAGE_IN
  62      if not stageio.stage_manifest(stage_in_dir):
  63          print(f"BLOCKED: {stage_in_dir} has no stage manifest. Run s04_langid.py first.")
  64          log("S07", f"Hold out val/test ({config.display_name})", runlog.BLOCKED,
  65              "s04_langid has not been run yet.")
  66          return 1
  67  
  68      splits_cfg = config.section("splits")
  69      seed = int(splits_cfg.get("seed", 20260817))
  70      train_ratio = float(splits_cfg.get("train", 0.90))
  71      val_ratio = float(splits_cfg.get("val", 0.05))
  72      test_ratio = float(splits_cfg.get("test", 0.05))
  73      total_ratio = train_ratio + val_ratio + test_ratio
  74      if abs(total_ratio - 1.0) > 1e-6:
  75          print(f"BLOCKED: split ratios sum to {total_ratio}, not 1.0 -- fix configs/language.yaml")
  76          log("S07", f"Hold out val/test ({config.display_name})", runlog.BLOCKED,
  77              f"train+val+test = {total_ratio}, expected 1.0")
  78          return 1
  79  
  80      print(
  81          f"S07 holdout ({config.display_name}) -- seed={seed} "
  82          f"train={train_ratio:.0%} val={val_ratio:.0%} test={test_ratio:.0%}",
  83          flush=True,
  84      )
  85  
  86      counts = {"train": 0, "val": 0, "test": 0}
  87      seen_doc_ids: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
  88      docs_in = 0
  89  
  90      def bucketed(target: str):
  91          nonlocal docs_in
  92          for record in stageio.read_stage(stage_in_dir):
  93              if args.limit and docs_in >= args.limit:
  94                  break
  95              docs_in_local = docs_in
  96              docs_in += 1
  97              bucket = split_bucket(record["doc_id"], seed)
  98              if bucket < train_ratio:
  99                  name = "train"
 100              elif bucket < train_ratio + val_ratio:
 101                  name = "val"
 102              else:
 103                  name = "test"
 104              counts[name] += 1
 105              seen_doc_ids[name].add(record["doc_id"])
 106              if name == target:
 107                  yield record
 108  
 109      # Three independent passes over S04's output rather than three parallel
 110      # generators sharing one iterator -- simpler, and S04's output is already
 111      # on disk so re-reading it costs I/O, not re-computation.
 112      n_train = stageio.write_stage(config.interim_dir / STAGE_TRAIN_POOL, bucketed("train"))
 113      docs_in = 0
 114      n_val = stageio.write_stage(config.interim_dir / STAGE_VAL, bucketed("val"))
 115      docs_in = 0
 116      n_test = stageio.write_stage(config.interim_dir / STAGE_TEST, bucketed("test"))
 117  
 118      overlap_train_val = seen_doc_ids["train"] & seen_doc_ids["val"]
 119      overlap_train_test = seen_doc_ids["train"] & seen_doc_ids["test"]
 120      overlap_val_test = seen_doc_ids["val"] & seen_doc_ids["test"]
 121      zero_overlap = not (overlap_train_val or overlap_train_test or overlap_val_test)
 122  
 123      total = n_train + n_val + n_test
 124      elapsed = time.time() - started
 125      payload = {
 126          "docs_in": total,
 127          "train_docs": n_train,
 128          "val_docs": n_val,
 129          "test_docs": n_test,
 130          "train_fraction": round(n_train / total, 4) if total else 0.0,
 131          "val_fraction": round(n_val / total, 4) if total else 0.0,
 132          "test_fraction": round(n_test / total, 4) if total else 0.0,
 133          "target_train": train_ratio,
 134          "target_val": val_ratio,
 135          "target_test": test_ratio,
 136          "zero_doc_id_overlap": zero_overlap,
 137          "seed": seed,
 138          "seconds": round(elapsed, 1),
 139      }
 140      stageio.write_stage_manifest(config.interim_dir / STAGE_TRAIN_POOL, payload)
 141  
 142      print(f"  train={n_train:,} ({payload['train_fraction']:.2%})  "
 143            f"val={n_val:,} ({payload['val_fraction']:.2%})  "
 144            f"test={n_test:,} ({payload['test_fraction']:.2%})")
 145      print(f"  zero doc_id overlap across splits: {zero_overlap}")
 146  
 147      ok = zero_overlap and total > 0
 148      log(
 149          "S07",
 150          f"Hold out val/test ({config.display_name})",
 151          runlog.SUCCESS if ok else runlog.BLOCKED,
 152          (
 153              f"Split {total:,} documents: train {n_train:,} "
 154              f"({payload['train_fraction']:.1%}), val {n_val:,} "
 155              f"({payload['val_fraction']:.1%}), test {n_test:,} "
 156              f"({payload['test_fraction']:.1%}). Zero doc_id overlap."
 157              if ok
 158              else "Split produced overlapping doc_ids or zero documents -- a bug, not a filter."
 159          ),
 160          metrics=payload,
 161      )
 162      if not ok:
 163          print("BLOCKED: doc_id overlap detected between splits, or no documents produced.")
 164          return 1
 165      return 0
 166  
 167  
 168  if __name__ == "__main__":
 169      raise SystemExit(main())
```


## nepali/scripts/s08_dedup.py

```python
   1  #!/usr/bin/env python3
   2  """
   3  Step 08 -- deduplicate the train pool, manual-priority on every collision.
   4  
   5      python hindi/scripts/s08_dedup.py
   6      python hindi/scripts/s08_dedup.py --limit 50000     # smoke test
   7  
   8  Exact ``doc_id`` matches first, then MinHash + LSH (100 permutations, b=20
   9  bands of r=5 rows, threshold ~0.55 -- see ``corpuslib.dedup``). Runs on S06's
  10  output: the train pool only, after quality filtering. val/test (S07) are
  11  never touched here -- decontamination (S09) is the mechanism that keeps train
  12  from leaking into them, not this step.
  13  
  14  **The rule that matters:** cross-lane collisions resolve in favour of the
  15  manual copy. Sangraha and IndicCorp are web-crawl derived and do contain
  16  Wikipedia and news text our own manual sources also collect -- that overlap
  17  is expected and was measured with MinHash before this pipeline existed. A
  18  lane-blind dedup pass would delete manual documents at random and erode the
  19  20% requirement, which is exactly how the previous run's manual share
  20  evaporated. ``cross_lane_duplicate_pairs`` and
  21  ``manual_docs_preserved_by_priority`` in the report below are the direct,
  22  quantified answer to that.
  23  
  24  Both dedup passes need the full candidate set in memory at once -- the LSH
  25  bucket index has to see every document before it can tell which are
  26  duplicates of which, so there is no streaming form of this step. That is a
  27  property of ``corpuslib.dedup`` (already covered by test_corpuslib.py), not
  28  new here.
  29  
  30  MinHash signature computation (the expensive part -- 100 permutations per
  31  document) runs across CPU cores via ``--workers`` (default: cpu_count-1); the
  32  LSH bucket insertion that depends on manual-first ordering stays
  33  single-threaded, since that is what makes the priority rule deterministic.
  34  """
  35  
  36  from __future__ import annotations
  37  
  38  import argparse
  39  import sys
  40  import time
  41  from pathlib import Path
  42  
  43  sys.path.insert(0, str(Path(__file__).resolve().parent))
  44  
  45  from corpuslib import dedup, runlog, stageio  # noqa: E402
  46  from corpuslib.config import load_language_config, load_yaml  # noqa: E402
  47  
  48  STAGE_IN = "s06_perplexity"
  49  STAGE_OUT = "s08_dedup"
  50  
  51  
  52  def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  53      parser = argparse.ArgumentParser(description=__doc__)
  54      parser.add_argument("--limit", type=int, default=0, help="process at most N documents")
  55      parser.add_argument(
  56          "--workers", type=int, default=None,
  57          help="CPU workers for MinHash signature computation (default: cpu_count-1)",
  58      )
  59      return parser.parse_args(argv)
  60  
  61  
  62  def main(argv: list[str] | None = None) -> int:
  63      args = parse_args(argv)
  64      config = load_language_config()
  65      config.ensure_dirs()
  66      log = runlog.StepLogger(config.lang, config.events_path)
  67      started = time.time()
  68  
  69      stage_in_dir = config.interim_dir / STAGE_IN
  70      if not stageio.stage_manifest(stage_in_dir):
  71          print(f"BLOCKED: {stage_in_dir} has no stage manifest. Run s06_perplexity.py first.")
  72          log("S08", f"Deduplicate ({config.display_name})", runlog.BLOCKED,
  73              "s06_perplexity has not been run yet.")
  74          return 1
  75  
  76      cleaning_cfg = load_yaml(config.configs_dir / "cleaning.yaml")
  77      dcfg = cleaning_cfg.get("dedup", {}) or {}
  78      num_perm = int(dcfg.get("num_perm", dedup.DEFAULT_NUM_PERM))
  79      bands = int(dcfg.get("bands", dedup.DEFAULT_BANDS))
  80      rows = int(dcfg.get("rows", dedup.DEFAULT_ROWS))
  81      shingle_size = int(dcfg.get("shingle_size", dedup.DEFAULT_SHINGLE))
  82      threshold = float(dcfg.get("threshold", 0.55))
  83  
  84      print(
  85          f"S08 dedup ({config.display_name}) -- num_perm={num_perm} "
  86          f"bands={bands} rows={rows} threshold={threshold}",
  87          flush=True,
  88      )
  89  
  90      records = list(stageio.read_stage(stage_in_dir))
  91      if args.limit:
  92          records = records[: args.limit]
  93      print(f"  loaded {len(records):,} documents", flush=True)
  94  
  95      stats = dedup.DedupStats()
  96      after_exact = dedup.remove_exact(records, stats)
  97      print(f"  exact dedup: {stats.exact_removed:,} removed, {len(after_exact):,} remain", flush=True)
  98  
  99      def progress(position: int, kept: int) -> None:
 100          print(f"  ...LSH pass at {position:,} documents, {kept:,} kept so far", flush=True)
 101  
 102      after_near = dedup.remove_near_duplicates(
 103          after_exact,
 104          num_perm=num_perm,
 105          bands=bands,
 106          rows=rows,
 107          shingle_size=shingle_size,
 108          threshold=threshold,
 109          stats=stats,
 110          on_progress=progress,
 111          workers=args.workers,
 112      )
 113      print(f"  near dedup: {stats.near_removed:,} removed, {len(after_near):,} remain", flush=True)
 114  
 115      written = stageio.write_stage(config.interim_dir / STAGE_OUT, after_near)
 116      elapsed = time.time() - started
 117  
 118      report = stats.to_dict()
 119      report["seconds"] = round(elapsed, 1)
 120      report["config"] = {
 121          "num_perm": num_perm, "bands": bands, "rows": rows,
 122          "shingle_size": shingle_size, "threshold": threshold,
 123      }
 124      stageio.write_stage_manifest(config.interim_dir / STAGE_OUT, report)
 125  
 126      print(
 127          f"  input={stats.input_docs:,}  kept={written:,}  retention={report['retention']:.2%}\n"
 128          f"  cross_lane_duplicate_pairs={stats.cross_lane_pairs:,}  "
 129          f"manual_docs_preserved_by_priority={stats.manual_saved_by_priority:,}\n"
 130          f"  removed_by_lane={dict(stats.removed_by_lane)}"
 131      )
 132  
 133      log(
 134          "S08",
 135          f"Deduplicate ({config.display_name})",
 136          runlog.SUCCESS,
 137          (
 138              f"Removed {report['total_removed']:,} of {stats.input_docs:,} documents "
 139              f"({stats.exact_removed:,} exact, {stats.near_removed:,} near). "
 140              f"{stats.cross_lane_pairs:,} cross-lane duplicate pairs found; "
 141              f"{stats.manual_saved_by_priority:,} resolved to the manual copy."
 142          ),
 143          metrics=report,
 144      )
 145      return 0
 146  
 147  
 148  if __name__ == "__main__":
 149      raise SystemExit(main())
```


## nepali/scripts/s09_decontaminate.py

```python
   1  #!/usr/bin/env python3
   2  """
   3  Step 09 -- remove train documents that leak into val or test.
   4  
   5      python hindi/scripts/s09_decontaminate.py
   6      python hindi/scripts/s09_decontaminate.py --limit 50000
   7  
   8  Same MinHash machinery as S08, run between the deduplicated train pool and
   9  the union of val+test (both from S07, untouched since). A paraphrased test
  10  document leaks nearly as much signal as an exact copy, so exact ``doc_id``
  11  comparison alone is not enough -- this is the near-duplicate check, applied
  12  in the other direction from S08: here it only ever removes from **train**,
  13  never from val/test, because the held-out sets are the fixed reference an
  14  exact-copy training document would be leaking from.
  15  
  16  **Verify:** a final rescan reports zero above-threshold train/holdout
  17  matches. If it does not, that is this step failing to do its job, not a
  18  retention number to shrug at.
  19  
  20  Holdout signatures are computed once and its LSH index built once, then
  21  queried twice (removal pass, then rescan) -- the original version rebuilt
  22  both the holdout index and every train signature from scratch for the
  23  rescan, which was pure repeated work: the rescan only needs to know whether
  24  the *already-computed* kept-train signatures still collide with the *same*
  25  holdout index. Train-signature hashing itself is parallelised across CPU cores via
  26  ``--workers`` (default: cpu_count-1). Together, eliminating the duplicate
  27  hashing pass and parallelising what remains should cut this step well below
  28  the ~2h it measured at on the 1.56M-document Nepali run -- not re-benchmarked
  29  end to end, since that run itself takes hours.
  30  """
  31  
  32  from __future__ import annotations
  33  
  34  import argparse
  35  import sys
  36  import time
  37  from pathlib import Path
  38  
  39  sys.path.insert(0, str(Path(__file__).resolve().parent))
  40  
  41  from corpuslib import dedup, runlog, stageio  # noqa: E402
  42  from corpuslib.config import load_language_config, load_yaml  # noqa: E402
  43  
  44  STAGE_TRAIN_IN = "s08_dedup"
  45  STAGE_VAL = "s07_val"
  46  STAGE_TEST = "s07_test"
  47  STAGE_OUT = "s09_decontaminated"
  48  
  49  
  50  def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  51      parser = argparse.ArgumentParser(description=__doc__)
  52      parser.add_argument("--limit", type=int, default=0, help="process at most N train documents")
  53      parser.add_argument(
  54          "--workers", type=int, default=None,
  55          help="CPU workers for MinHash signature computation (default: cpu_count-1)",
  56      )
  57      return parser.parse_args(argv)
  58  
  59  
  60  def main(argv: list[str] | None = None) -> int:
  61      args = parse_args(argv)
  62      config = load_language_config()
  63      config.ensure_dirs()
  64      log = runlog.StepLogger(config.lang, config.events_path)
  65      started = time.time()
  66  
  67      train_dir = config.interim_dir / STAGE_TRAIN_IN
  68      val_dir = config.interim_dir / STAGE_VAL
  69      test_dir = config.interim_dir / STAGE_TEST
  70      if not stageio.stage_manifest(train_dir):
  71          print(f"BLOCKED: {train_dir} has no stage manifest. Run s08_dedup.py first.")
  72          log("S09", f"Decontaminate ({config.display_name})", runlog.BLOCKED,
  73              "s08_dedup has not been run yet.")
  74          return 1
  75  
  76      cleaning_cfg = load_yaml(config.configs_dir / "cleaning.yaml")
  77      dccfg = cleaning_cfg.get("decontamination", {}) or {}
  78      if not dccfg.get("enabled", True):
  79          written = stageio.write_stage(
  80              config.interim_dir / STAGE_OUT, stageio.read_stage(train_dir)
  81          )
  82          stageio.write_stage_manifest(
  83              config.interim_dir / STAGE_OUT, {"applied": False, "docs_out": written}
  84          )
  85          log("S09", f"Decontaminate ({config.display_name})", runlog.FALLBACK,
  86              "Disabled in config; train pool passed through unchanged.",
  87              metrics={"docs_out": written}, fallback="disabled_in_config")
  88          return 0
  89  
  90      threshold = float(dccfg.get("threshold", 0.55))
  91      dedup_cfg = cleaning_cfg.get("dedup", {}) or {}
  92      num_perm = int(dedup_cfg.get("num_perm", dedup.DEFAULT_NUM_PERM))
  93      bands = int(dedup_cfg.get("bands", dedup.DEFAULT_BANDS))
  94      rows = int(dedup_cfg.get("rows", dedup.DEFAULT_ROWS))
  95      shingle_size = int(dedup_cfg.get("shingle_size", dedup.DEFAULT_SHINGLE))
  96  
  97      print(f"S09 decontaminate ({config.display_name}) -- threshold={threshold}", flush=True)
  98  
  99      train = list(stageio.read_stage(train_dir))
 100      if args.limit:
 101          train = train[: args.limit]
 102      holdout = list(stageio.read_stage(val_dir)) + list(stageio.read_stage(test_dir))
 103      print(f"  train={len(train):,}  holdout(val+test)={len(holdout):,}", flush=True)
 104  
 105      # Holdout side: hash once, index once -- reused for both the removal pass
 106      # and the rescan below.
 107      holdout_signatures = dedup.compute_signatures(
 108          [r["text"] for r in holdout],
 109          num_perm=num_perm, shingle_size=shingle_size, workers=args.workers,
 110      )
 111      holdout_buckets = dedup.build_lsh_buckets(holdout_signatures, bands=bands, rows=rows)
 112  
 113      train_signatures = dedup.compute_signatures(
 114          [r["text"] for r in train],
 115          num_perm=num_perm, shingle_size=shingle_size, workers=args.workers,
 116      )
 117      hits = dedup.query_matches(
 118          train_signatures, holdout_signatures, holdout_buckets,
 119          bands=bands, rows=rows, threshold=threshold,
 120      )
 121      contaminated_ids = {train[i]["doc_id"] for i, hit in enumerate(hits) if hit is not None}
 122      kept = [r for r, hit in zip(train, hits) if hit is None]
 123      kept_signatures = [s for s, hit in zip(train_signatures, hits) if hit is None]
 124      written = stageio.write_stage(config.interim_dir / STAGE_OUT, kept)
 125  
 126      # Final rescan: zero above-threshold matches should remain between what
 127      # was kept and the holdout sets. Reuses kept_signatures/holdout_buckets --
 128      # nothing here needs re-hashing, since neither side's text changed.
 129      rescan_hits = dedup.query_matches(
 130          kept_signatures, holdout_signatures, holdout_buckets,
 131          bands=bands, rows=rows, threshold=threshold,
 132      )
 133      rescan_ids = {kept[i]["doc_id"] for i, hit in enumerate(rescan_hits) if hit is not None}
 134  
 135      elapsed = time.time() - started
 136      payload = {
 137          "applied": True,
 138          "train_docs_in": len(train),
 139          "holdout_docs": len(holdout),
 140          "contaminated_removed": len(contaminated_ids),
 141          "train_docs_out": written,
 142          "rescan_remaining_matches": len(rescan_ids),
 143          "threshold": threshold,
 144          "seconds": round(elapsed, 1),
 145      }
 146      stageio.write_stage_manifest(config.interim_dir / STAGE_OUT, payload)
 147  
 148      print(f"  removed {len(contaminated_ids):,} contaminated train documents")
 149      print(f"  final rescan above-threshold matches: {len(rescan_ids)}")
 150  
 151      ok = len(rescan_ids) == 0
 152      log(
 153          "S09",
 154          f"Decontaminate ({config.display_name})",
 155          runlog.SUCCESS if ok else runlog.BLOCKED,
 156          (
 157              f"Removed {len(contaminated_ids):,} train documents matching val/test "
 158              f"above {threshold}. Final rescan: 0 matches remain."
 159              if ok
 160              else (
 161                  f"Final rescan still finds {len(rescan_ids)} above-threshold matches "
 162                  f"after removal -- decontamination did not fully succeed."
 163              )
 164          ),
 165          metrics=payload,
 166      )
 167      if not ok:
 168          print(f"BLOCKED: {len(rescan_ids)} contaminating matches remain after removal.")
 169          return 1
 170      return 0
 171  
 172  
 173  if __name__ == "__main__":
 174      raise SystemExit(main())
```


## nepali/scripts/s11_finalize_splits.py

```python
   1  #!/usr/bin/env python3
   2  """
   3  Step 11 -- write the final train/val/test files.
   4  
   5      python hindi/scripts/s11_finalize_splits.py
   6  
   7  Train comes from S10 (balanced, deduplicated, decontaminated). Val and test
   8  come straight from S07 -- normalised and language-ID'd only, never touched by
   9  quality filtering, dedup, or lane balancing, so they stay a fair, unbiased
  10  sample of what was collected.
  11  
  12  **Verify:** manual ratio is reported *per split*, not only overall. A split
  13  whose manual share deviates by more than 10 points from the overall figure is
  14  flagged -- lane balancing only targets the train pool, so val/test could in
  15  principle drift if collection itself were lane-skewed in a way that happened
  16  to concentrate in one split, and that would silently bias evaluation.
  17  """
  18  
  19  from __future__ import annotations
  20  
  21  import argparse
  22  import sys
  23  import time
  24  from pathlib import Path
  25  
  26  sys.path.insert(0, str(Path(__file__).resolve().parent))
  27  
  28  from corpuslib import runlog, schema, stageio  # noqa: E402
  29  from corpuslib.config import load_language_config  # noqa: E402
  30  from corpuslib.jsonlio import atomic_write_json, write_jsonl  # noqa: E402
  31  
  32  STAGE_TRAIN_IN = "s10_balanced"
  33  STAGE_VAL_IN = "s07_val"
  34  STAGE_TEST_IN = "s07_test"
  35  DEVIATION_FLAG_POINTS = 0.10
  36  
  37  
  38  def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  39      return argparse.ArgumentParser(description=__doc__).parse_args(argv)
  40  
  41  
  42  def _split_stats(records: list[dict]) -> dict:
  43      manual_words = sum(r["n_words"] for r in records if r["lane"] == schema.LANE_MANUAL)
  44      downloaded_words = sum(r["n_words"] for r in records if r["lane"] == schema.LANE_DOWNLOADED)
  45      total_words = manual_words + downloaded_words
  46      return {
  47          "docs": len(records),
  48          "manual_words": manual_words,
  49          "downloaded_words": downloaded_words,
  50          "total_words": total_words,
  51          "manual_ratio": round(manual_words / total_words, 4) if total_words else 0.0,
  52      }
  53  
  54  
  55  def main(argv: list[str] | None = None) -> int:
  56      parse_args(argv)
  57      config = load_language_config()
  58      config.ensure_dirs()
  59      log = runlog.StepLogger(config.lang, config.events_path)
  60      started = time.time()
  61  
  62      sources = {"train": config.interim_dir / STAGE_TRAIN_IN,
  63                 "val": config.interim_dir / STAGE_VAL_IN,
  64                 "test": config.interim_dir / STAGE_TEST_IN}
  65      missing = [name for name, d in sources.items()
  66                 if not stageio.stage_manifest(d) and not list(d.glob("part-*.jsonl"))]
  67      if missing:
  68          print(f"BLOCKED: missing stage output for {missing}. Run the preceding steps first.")
  69          log("S11", f"Finalise splits ({config.display_name})", runlog.BLOCKED,
  70              f"Missing stage output: {missing}")
  71          return 1
  72  
  73      print(f"S11 finalise splits ({config.display_name})", flush=True)
  74  
  75      all_doc_ids: dict[str, set[str]] = {}
  76      stats: dict[str, dict] = {}
  77      for name, stage_dir in sources.items():
  78          records = list(stageio.read_stage(stage_dir))
  79          all_doc_ids[name] = {r["doc_id"] for r in records}
  80          stats[name] = _split_stats(records)
  81          out_path = config.final_dir / f"{name}.jsonl"
  82          write_jsonl(out_path, records)
  83          print(f"  {name}: {stats[name]['docs']:,} docs, "
  84                f"manual ratio {stats[name]['manual_ratio']:.2%} -> {out_path}", flush=True)
  85  
  86      overlap = {
  87          "train_val": sorted(all_doc_ids["train"] & all_doc_ids["val"])[:5],
  88          "train_test": sorted(all_doc_ids["train"] & all_doc_ids["test"])[:5],
  89          "val_test": sorted(all_doc_ids["val"] & all_doc_ids["test"])[:5],
  90      }
  91      zero_overlap = not any(overlap.values())
  92  
  93      overall_words = sum(s["total_words"] for s in stats.values())
  94      overall_manual = sum(s["manual_words"] for s in stats.values())
  95      overall_ratio = overall_manual / overall_words if overall_words else 0.0
  96  
  97      flagged = [
  98          name for name, s in stats.items()
  99          if s["total_words"] and abs(s["manual_ratio"] - overall_ratio) > DEVIATION_FLAG_POINTS
 100      ]
 101  
 102      elapsed = time.time() - started
 103      payload = {
 104          "splits": stats,
 105          "overall_manual_ratio": round(overall_ratio, 4),
 106          "zero_doc_id_overlap": zero_overlap,
 107          "overlap_samples": overlap,
 108          "flagged_splits": flagged,
 109          "deviation_flag_points": DEVIATION_FLAG_POINTS,
 110          "seconds": round(elapsed, 1),
 111      }
 112      atomic_write_json(config.final_dir / "split_stats.json", payload)
 113  
 114      print(f"  overall manual ratio: {overall_ratio:.2%}")
 115      print(f"  zero doc_id overlap across splits: {zero_overlap}")
 116      if flagged:
 117          print(f"  FLAGGED (>{DEVIATION_FLAG_POINTS:.0%} deviation from overall): {flagged}")
 118  
 119      ok = zero_overlap and all(s["docs"] > 0 for s in stats.values())
 120      log(
 121          "S11",
 122          f"Finalise splits ({config.display_name})",
 123          runlog.SUCCESS if ok else runlog.BLOCKED,
 124          (
 125              f"Wrote train/val/test ({stats['train']['docs']:,}/{stats['val']['docs']:,}/"
 126              f"{stats['test']['docs']:,} docs). Overall manual ratio {overall_ratio:.2%}."
 127              + (f" Flagged for deviation: {flagged}." if flagged else "")
 128              if ok
 129              else "Non-empty split or doc_id overlap violated -- a bug, not a filter."
 130          ),
 131          metrics=payload,
 132      )
 133      if not ok:
 134          print("BLOCKED: empty split, or doc_id overlap detected between splits.")
 135          return 1
 136      return 0
 137  
 138  
 139  if __name__ == "__main__":
 140      raise SystemExit(main())
```


## nepali/scripts/s12_train_tokenizer.py

```python
   1  #!/usr/bin/env python3
   2  """
   3  Step 12 -- sweep vocabulary sizes and select by the fertility elbow.
   4  
   5      python hindi/scripts/s12_train_tokenizer.py
   6  
   7  Trains one SentencePiece BPE model per size in ``configs/tokenizer.yaml:
   8  vocab_sizes`` on the **train** split (never val/test -- selection must not
   9  see the data it will later be evaluated against). ``character_coverage``
  10  is set explicitly (0.9995) because SentencePiece's default is tuned for
  11  Latin scripts and under-covers Devanagari's combining-mark range otherwise.
  12  
  13  ``byte_fallback`` is **disabled** on purpose: with it on, every byte sequence
  14  is representable, UNK is identically zero at every vocabulary size, and the
  15  brief's "unknown-token statistics" become meaningless, as does any selection
  16  rule keyed on UNK.
  17  
  18  Selection is the **fertility elbow**: walk the swept sizes in increasing
  19  order and stop at the first one where growing further buys less than
  20  ``min_relative_gain`` relative fertility improvement, subject to
  21  ``max_unk_rate``. The brief asks for "tens of thousands", so an elbow below
  22  ``preferred_min_vocab`` is overridden up to the smallest candidate that meets
  23  it -- logged explicitly, since the plan requires that override to be
  24  justified from the measured curve, not silently applied.
  25  
  26  For Nepali, if the selected BPE model's fertility exceeds
  27  ``compare_unigram.trigger_fertility_above``, a Unigram model is also trained
  28  at the same vocabulary size and whichever scores lower fertility (subject to
  29  its own UNK ceiling) is kept -- agglutinative morphology is exactly where
  30  Unigram's probabilistic segmentation tends to beat greedy BPE merges, and the
  31  brief wants this as a genuine H/L finding, not an assumption.
  32  
  33  **Verify:** two ``.model`` files exist (Hindi, Nepali) with different content
  34  hashes -- the brief forbids sharing vocabularies between the two models. This
  35  step only ever writes its own language's file, so that is guaranteed by
  36  construction; the QA gate (S16) still checks it because a construction
  37  guarantee is not a substitute for verifying the artifact.
  38  """
  39  
  40  from __future__ import annotations
  41  
  42  import argparse
  43  import hashlib
  44  import os
  45  import sys
  46  import time
  47  from pathlib import Path
  48  
  49  sys.path.insert(0, str(Path(__file__).resolve().parent))
  50  
  51  from corpuslib import runlog, tokeval  # noqa: E402
  52  from corpuslib.config import load_language_config  # noqa: E402
  53  from corpuslib.jsonlio import atomic_write_json, read_jsonl  # noqa: E402
  54  
  55  
  56  def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  57      parser = argparse.ArgumentParser(description=__doc__)
  58      parser.add_argument("--val-sample", type=int, default=20_000,
  59                           help="val documents used to measure fertility/UNK per candidate")
  60      return parser.parse_args(argv)
  61  
  62  
  63  def _file_sha256(path: Path) -> str:
  64      digest = hashlib.sha256()
  65      digest.update(Path(path).read_bytes())
  66      return digest.hexdigest()[:16]
  67  
  68  
  69  def _train_one(*, model_type, vocab_size, train_path, model_path_prefix,
  70                  character_coverage, byte_fallback, training_cfg, special_tokens):
  71      import sentencepiece as spm
  72  
  73      def sentences():
  74          for record in read_jsonl(train_path):
  75              for line in record["text"].split("\n"):
  76                  line = line.strip()
  77                  if line:
  78                      yield line
  79  
  80      with open(os.devnull, "w", encoding="utf-8") as devnull:
  81          spm.SentencePieceTrainer.train(
  82              sentence_iterator=sentences(),
  83              model_prefix=str(model_path_prefix),
  84              vocab_size=vocab_size,
  85              model_type=model_type,
  86              character_coverage=character_coverage,
  87              byte_fallback=byte_fallback,
  88              input_sentence_size=int(training_cfg.get("input_sentence_size", 3_000_000)),
  89              shuffle_input_sentence=bool(training_cfg.get("shuffle_input_sentence", True)),
  90              num_threads=int(training_cfg.get("num_threads", 8)),
  91              max_sentence_length=int(training_cfg.get("max_sentence_length", 8192)),
  92              pad_piece=special_tokens.get("pad", "<pad>"),
  93              unk_piece=special_tokens.get("unk", "<unk>"),
  94              bos_piece=special_tokens.get("bos", "<s>"),
  95              eos_piece=special_tokens.get("eos", "</s>"),
  96              pad_id=0, unk_id=1, bos_id=2, eos_id=3,
  97              logstream=devnull,
  98          )
  99      return Path(str(model_path_prefix) + ".model")
 100  
 101  
 102  def main(argv: list[str] | None = None) -> int:
 103      args = parse_args(argv)
 104      config = load_language_config()
 105      config.ensure_dirs()
 106      log = runlog.StepLogger(config.lang, config.events_path)
 107      started = time.time()
 108  
 109      train_path = config.final_dir / "train.jsonl"
 110      val_path = config.final_dir / "val.jsonl"
 111      if not train_path.exists() or not val_path.exists():
 112          print(f"BLOCKED: {train_path} or {val_path} missing. Run s11_finalize_splits.py first.")
 113          log("S12", f"Train tokenizer ({config.display_name})", runlog.BLOCKED,
 114              "final train/val splits do not exist yet.")
 115          return 1
 116  
 117      import yaml
 118      tok_cfg = yaml.safe_load((config.configs_dir / "tokenizer.yaml").read_text("utf-8"))
 119  
 120      model_type = tok_cfg.get("model_type", "bpe")
 121      model_prefix_name = tok_cfg.get("model_prefix", f"{config.lang}_bpe")
 122      character_coverage = float(tok_cfg.get("character_coverage", 0.9995))
 123      byte_fallback = bool(tok_cfg.get("byte_fallback", False))
 124      vocab_sizes = sorted(int(v) for v in tok_cfg.get("vocab_sizes", [16000, 24000, 32000, 48000]))
 125      preferred_min_vocab = int(tok_cfg.get("preferred_min_vocab", 24000))
 126      selection_cfg = tok_cfg.get("selection", {}) or {}
 127      max_unk_rate = float(selection_cfg.get("max_unk_rate", 0.001))
 128      min_relative_gain = float(selection_cfg.get("min_relative_gain", 0.03))
 129      training_cfg = tok_cfg.get("training", {}) or {}
 130      special_tokens = tok_cfg.get("special_tokens", {}) or {}
 131      compare_unigram_cfg = tok_cfg.get("compare_unigram", {}) or {}
 132  
 133      print(
 134          f"S12 train tokenizer ({config.display_name}) -- model_type={model_type} "
 135          f"vocab_sizes={vocab_sizes} coverage={character_coverage}",
 136          flush=True,
 137      )
 138  
 139      # Val sample for measurement, loaded once and reused across candidates.
 140      val_texts: list[str] = []
 141      for record in read_jsonl(val_path):
 142          val_texts.append(record["text"])
 143          if len(val_texts) >= args.val_sample:
 144              break
 145      print(f"  measuring candidates against {len(val_texts):,} val documents", flush=True)
 146  
 147      sweep_dir = config.tokenizer_dir / "sweep"
 148      sweep_dir.mkdir(parents=True, exist_ok=True)
 149      candidates: list[dict] = []
 150  
 151      for vocab_size in vocab_sizes:
 152          t0 = time.time()
 153          model_prefix_path = sweep_dir / f"{model_prefix_name}_{model_type}_{vocab_size}"
 154          print(f"  training {model_type} @ {vocab_size:,}...", flush=True)
 155          model_path = _train_one(
 156              model_type=model_type, vocab_size=vocab_size, train_path=train_path,
 157              model_path_prefix=model_prefix_path, character_coverage=character_coverage,
 158              byte_fallback=byte_fallback, training_cfg=training_cfg, special_tokens=special_tokens,
 159          )
 160          import sentencepiece as spm
 161          processor = spm.SentencePieceProcessor(model_file=str(model_path))
 162          m = tokeval.measure(processor, val_texts)
 163          candidates.append({
 164              "model_type": model_type,
 165              "vocab_size": vocab_size,
 166              "fertility": m["fertility"],
 167              "unk_rate": m["unk_rate"],
 168              "chars_per_token": m["chars_per_token"],
 169              "model_path": str(model_path),
 170              "seconds": round(time.time() - t0, 1),
 171          })
 172          print(
 173              f"    fertility={m['fertility']:.4f}  unk_rate={m['unk_rate']:.4%}  "
 174              f"({time.time()-t0:.0f}s)", flush=True,
 175          )
 176  
 177      # -- fertility-elbow selection ------------------------------------------
 178      eligible = [c for c in candidates if c["unk_rate"] <= max_unk_rate]
 179      selection_notes = []
 180      if not eligible:
 181          eligible = sorted(candidates, key=lambda c: c["unk_rate"])[:1]
 182          selection_notes.append(
 183              f"no candidate met max_unk_rate={max_unk_rate:.3%}; falling back to the "
 184              f"lowest-UNK candidate ({eligible[0]['vocab_size']:,})."
 185          )
 186  
 187      chosen = eligible[-1]  # default: largest eligible, if no elbow found
 188      for i in range(len(eligible) - 1):
 189          cur, nxt = eligible[i], eligible[i + 1]
 190          if cur["fertility"] <= 0:
 191              continue
 192          relative_gain = (cur["fertility"] - nxt["fertility"]) / cur["fertility"]
 193          if relative_gain < min_relative_gain:
 194              chosen = cur
 195              selection_notes.append(
 196                  f"elbow at {cur['vocab_size']:,}: growing to {nxt['vocab_size']:,} would "
 197                  f"buy only {relative_gain:.1%} relative fertility gain "
 198                  f"(< {min_relative_gain:.0%} threshold)."
 199              )
 200              break
 201      else:
 202          selection_notes.append(
 203              f"no elbow found within the swept range; fertility was still improving by "
 204              f"more than {min_relative_gain:.0%} at every step, so the largest candidate "
 205              f"({chosen['vocab_size']:,}) was selected."
 206          )
 207  
 208      if chosen["vocab_size"] < preferred_min_vocab:
 209          promoted = [c for c in eligible if c["vocab_size"] >= preferred_min_vocab]
 210          if promoted:
 211              selection_notes.append(
 212                  f"elbow selection ({chosen['vocab_size']:,}) is below preferred_min_vocab "
 213                  f"({preferred_min_vocab:,}); overridden to {promoted[0]['vocab_size']:,} per "
 214                  f"the brief's 'tens of thousands' guidance."
 215              )
 216              chosen = promoted[0]
 217          else:
 218              selection_notes.append(
 219                  f"elbow selection ({chosen['vocab_size']:,}) is below preferred_min_vocab "
 220                  f"({preferred_min_vocab:,}), and no swept candidate reaches it -- kept the "
 221                  f"elbow choice; widen vocab_sizes in tokenizer.yaml if this needs revisiting."
 222              )
 223  
 224      # -- Nepali-only: compare against Unigram at the chosen size ------------
 225      if compare_unigram_cfg.get("enabled") and chosen["fertility"] > float(
 226          compare_unigram_cfg.get("trigger_fertility_above", 999)
 227      ):
 228          print(
 229              f"  BPE fertility {chosen['fertility']:.4f} exceeds "
 230              f"{compare_unigram_cfg['trigger_fertility_above']} -- training Unigram at "
 231              f"{chosen['vocab_size']:,} for comparison...",
 232              flush=True,
 233          )
 234          uni_prefix = sweep_dir / f"{model_prefix_name}_unigram_{chosen['vocab_size']}"
 235          uni_model_path = _train_one(
 236              model_type="unigram", vocab_size=chosen["vocab_size"], train_path=train_path,
 237              model_path_prefix=uni_prefix, character_coverage=character_coverage,
 238              byte_fallback=byte_fallback, training_cfg=training_cfg, special_tokens=special_tokens,
 239          )
 240          import sentencepiece as spm
 241          uni_processor = spm.SentencePieceProcessor(model_file=str(uni_model_path))
 242          uni_m = tokeval.measure(uni_processor, val_texts)
 243          selection_notes.append(
 244              f"Unigram @ {chosen['vocab_size']:,}: fertility={uni_m['fertility']:.4f} "
 245              f"unk_rate={uni_m['unk_rate']:.4%} vs BPE fertility={chosen['fertility']:.4f} "
 246              f"unk_rate={chosen['unk_rate']:.4%}."
 247          )
 248          if uni_m["fertility"] < chosen["fertility"] and uni_m["unk_rate"] <= max_unk_rate:
 249              selection_notes.append("Unigram beat BPE on fertility within the UNK ceiling -- selected.")
 250              chosen = {
 251                  "model_type": "unigram", "vocab_size": chosen["vocab_size"],
 252                  "fertility": uni_m["fertility"], "unk_rate": uni_m["unk_rate"],
 253                  "chars_per_token": uni_m["chars_per_token"], "model_path": str(uni_model_path),
 254              }
 255          else:
 256              selection_notes.append("BPE retained -- Unigram did not improve fertility within the UNK ceiling.")
 257  
 258      # -- install the chosen model as the language's tokenizer ---------------
 259      final_prefix = config.tokenizer_dir / model_prefix_name
 260      final_model = Path(str(final_prefix) + ".model")
 261      final_vocab = Path(str(final_prefix) + ".vocab")
 262      chosen_model_path = Path(chosen["model_path"])
 263      final_model.write_bytes(chosen_model_path.read_bytes())
 264      final_vocab.write_bytes(chosen_model_path.with_suffix(".vocab").read_bytes())
 265  
 266      elapsed = time.time() - started
 267      payload = {
 268          "chosen": {k: v for k, v in chosen.items() if k != "model_path"},
 269          "chosen_model_path": str(final_model),
 270          "candidates": candidates,
 271          "selection_notes": selection_notes,
 272          "max_unk_rate": max_unk_rate,
 273          "min_relative_gain": min_relative_gain,
 274          "preferred_min_vocab": preferred_min_vocab,
 275          "model_sha256_16": _file_sha256(final_model),
 276          "seconds": round(elapsed, 1),
 277      }
 278      atomic_write_json(config.tokenizer_dir / "selection_report.json", payload)
 279  
 280      print(f"\n  SELECTED: {chosen['model_type']} @ {chosen['vocab_size']:,}  "
 281            f"fertility={chosen['fertility']:.4f}  unk_rate={chosen['unk_rate']:.4%}")
 282      print(f"  installed at {final_model}")
 283      for note in selection_notes:
 284          print(f"  - {note}")
 285  
 286      log(
 287          "S12",
 288          f"Train tokenizer ({config.display_name})",
 289          runlog.SUCCESS,
 290          f"Swept {len(candidates)} candidates; selected {chosen['model_type']} @ "
 291          f"{chosen['vocab_size']:,} (fertility {chosen['fertility']:.4f}, "
 292          f"UNK {chosen['unk_rate']:.4%}).",
 293          metrics=payload,
 294          decision="; ".join(selection_notes),
 295      )
 296      return 0
 297  
 298  
 299  if __name__ == "__main__":
 300      raise SystemExit(main())
```

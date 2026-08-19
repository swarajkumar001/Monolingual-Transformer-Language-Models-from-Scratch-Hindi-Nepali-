#!/usr/bin/env python3
"""
Step 03 -- normalise every collected document.

    python hindi/scripts/s03_normalize.py
    python hindi/scripts/s03_normalize.py --limit 5000     # smoke test

NFC normalisation, invisible-character stripping, optional ZWJ/ZWNJ and digit
normalisation, whitespace hygiene and PII redaction -- all driven by
``configs/language.yaml: normalization`` so the two languages can differ
(Hindi strips joiners, Nepali does not -- ZWJ is load-bearing inside Nepali
conjuncts such as ``पुर्‍याउने``).

This step never drops a document. Its only job is to put every document into
canonical form; deciding whether a document is *good* is S05's job, on text
this step has already cleaned up. Dropping here would make retention numbers
from both stages impossible to attribute to one cause.

``doc_id`` is recomputed from the normalised text (schema.py's own contract:
"doc_id is a content hash of the *normalized* text" -- true from this step
onward, not from collection, since no real normalisation had run yet at
collection time). The original collection-time id is kept at
``meta.raw_doc_id`` for traceability.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpuslib import runlog, schema, stageio  # noqa: E402
from corpuslib.config import load_language_config  # noqa: E402
from corpuslib.textnorm import normalize, redact_pii  # noqa: E402

STAGE_NAME = "s03_normalized"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="process at most N documents")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_language_config()
    config.ensure_dirs()
    log = runlog.StepLogger(config.lang, config.events_path)
    started = time.time()

    norm_cfg = config.normalization
    form = norm_cfg.get("form", "NFC")
    strip_joiners = bool(norm_cfg.get("strip_joiners", False))
    normalize_digits = bool(norm_cfg.get("normalize_digits", False))
    collapse_whitespace = bool(norm_cfg.get("collapse_whitespace", True))
    do_redact_pii = bool(norm_cfg.get("redact_pii", False))

    print(
        f"S03 normalise ({config.display_name}) -- form={form} "
        f"strip_joiners={strip_joiners} normalize_digits={normalize_digits} "
        f"redact_pii={do_redact_pii}",
        flush=True,
    )

    docs_in = 0
    docs_out = 0
    errors = 0
    pii_counts = {"email": 0, "phone": 0}
    by_lane_in: dict[str, int] = {}
    by_lane_out: dict[str, int] = {}

    def records():
        nonlocal docs_in, docs_out, errors
        for record in stageio.iter_raw_documents(config.raw_dir):
            if args.limit and docs_in >= args.limit:
                break
            docs_in += 1
            by_lane_in[record.get("lane", "?")] = by_lane_in.get(record.get("lane", "?"), 0) + 1
            try:
                text = normalize(
                    record.get("text", ""),
                    form=form,
                    strip_joiners=strip_joiners,
                    normalize_digits=normalize_digits,
                    collapse_whitespace=collapse_whitespace,
                )
                if do_redact_pii:
                    text, counts = redact_pii(text)
                    pii_counts["email"] += counts["email"]
                    pii_counts["phone"] += counts["phone"]

                record["meta"] = dict(record.get("meta") or {})
                record["meta"]["raw_doc_id"] = record["doc_id"]
                record["text"] = text
                record["doc_id"] = schema.content_hash(text)
                record["n_chars"] = len(text)
                record["n_words"] = schema.count_words(text)
            except Exception as exc:  # noqa: BLE001 - a normalisation bug must not vanish
                errors += 1
                print(
                    f"  ERROR normalising doc {record.get('doc_id', '?')}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                continue

            docs_out += 1
            by_lane_out[record.get("lane", "?")] = by_lane_out.get(record.get("lane", "?"), 0) + 1
            if docs_out % 200_000 == 0:
                print(f"  ...{docs_out:,} documents normalised", flush=True)
            yield record

    written = stageio.write_stage(config.interim_dir / STAGE_NAME, records())
    elapsed = time.time() - started

    payload = {
        "docs_in": docs_in,
        "docs_out": docs_out,
        "docs_written": written,
        "errors": errors,
        "by_lane_in": by_lane_in,
        "by_lane_out": by_lane_out,
        "pii_redacted": pii_counts,
        "seconds": round(elapsed, 1),
    }
    stageio.write_stage_manifest(config.interim_dir / STAGE_NAME, payload)

    ok = docs_in == docs_out == written and errors == 0
    print(f"  docs_in={docs_in:,}  docs_out={docs_out:,}  written={written:,}  errors={errors}")
    print(f"  PII redacted: {pii_counts['email']:,} emails, {pii_counts['phone']:,} phone numbers")

    log(
        "S03",
        f"Normalise ({config.display_name})",
        runlog.SUCCESS if ok else runlog.BLOCKED,
        (
            f"Normalised {docs_out:,} of {docs_in:,} documents; counts match, "
            f"zero documents dropped."
            if ok
            else (
                f"docs_in ({docs_in:,}) != docs_out ({docs_out:,}) or {errors} errors -- "
                f"normalisation must never drop documents; this is a bug, not a filter."
            )
        ),
        metrics=payload,
    )
    if not ok:
        print("BLOCKED: docs_in != docs_out (or errors occurred). This step must not drop documents.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

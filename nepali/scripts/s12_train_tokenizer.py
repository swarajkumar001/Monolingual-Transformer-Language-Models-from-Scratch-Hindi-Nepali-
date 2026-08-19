#!/usr/bin/env python3
"""
Step 12 -- sweep vocabulary sizes and select by the fertility elbow.

    python hindi/scripts/s12_train_tokenizer.py

Trains one SentencePiece BPE model per size in ``configs/tokenizer.yaml:
vocab_sizes`` on the **train** split (never val/test -- selection must not
see the data it will later be evaluated against). ``character_coverage``
is set explicitly (0.9995) because SentencePiece's default is tuned for
Latin scripts and under-covers Devanagari's combining-mark range otherwise.

``byte_fallback`` is **disabled** on purpose: with it on, every byte sequence
is representable, UNK is identically zero at every vocabulary size, and the
brief's "unknown-token statistics" become meaningless, as does any selection
rule keyed on UNK.

Selection is the **fertility elbow**: walk the swept sizes in increasing
order and stop at the first one where growing further buys less than
``min_relative_gain`` relative fertility improvement, subject to
``max_unk_rate``. The brief asks for "tens of thousands", so an elbow below
``preferred_min_vocab`` is overridden up to the smallest candidate that meets
it -- logged explicitly, since the plan requires that override to be
justified from the measured curve, not silently applied.

For Nepali, if the selected BPE model's fertility exceeds
``compare_unigram.trigger_fertility_above``, a Unigram model is also trained
at the same vocabulary size and whichever scores lower fertility (subject to
its own UNK ceiling) is kept -- agglutinative morphology is exactly where
Unigram's probabilistic segmentation tends to beat greedy BPE merges, and the
brief wants this as a genuine H/L finding, not an assumption.

**Verify:** two ``.model`` files exist (Hindi, Nepali) with different content
hashes -- the brief forbids sharing vocabularies between the two models. This
step only ever writes its own language's file, so that is guaranteed by
construction; the QA gate (S16) still checks it because a construction
guarantee is not a substitute for verifying the artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpuslib import runlog, tokeval  # noqa: E402
from corpuslib.config import load_language_config  # noqa: E402
from corpuslib.jsonlio import atomic_write_json, read_jsonl  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-sample", type=int, default=20_000,
                         help="val documents used to measure fertility/UNK per candidate")
    return parser.parse_args(argv)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()[:16]


def _train_one(*, model_type, vocab_size, train_path, model_path_prefix,
                character_coverage, byte_fallback, training_cfg, special_tokens):
    import sentencepiece as spm

    def sentences():
        for record in read_jsonl(train_path):
            for line in record["text"].split("\n"):
                line = line.strip()
                if line:
                    yield line

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        spm.SentencePieceTrainer.train(
            sentence_iterator=sentences(),
            model_prefix=str(model_path_prefix),
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            byte_fallback=byte_fallback,
            input_sentence_size=int(training_cfg.get("input_sentence_size", 3_000_000)),
            shuffle_input_sentence=bool(training_cfg.get("shuffle_input_sentence", True)),
            num_threads=int(training_cfg.get("num_threads", 8)),
            max_sentence_length=int(training_cfg.get("max_sentence_length", 8192)),
            pad_piece=special_tokens.get("pad", "<pad>"),
            unk_piece=special_tokens.get("unk", "<unk>"),
            bos_piece=special_tokens.get("bos", "<s>"),
            eos_piece=special_tokens.get("eos", "</s>"),
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            logstream=devnull,
        )
    return Path(str(model_path_prefix) + ".model")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_language_config()
    config.ensure_dirs()
    log = runlog.StepLogger(config.lang, config.events_path)
    started = time.time()

    train_path = config.final_dir / "train.jsonl"
    val_path = config.final_dir / "val.jsonl"
    if not train_path.exists() or not val_path.exists():
        print(f"BLOCKED: {train_path} or {val_path} missing. Run s11_finalize_splits.py first.")
        log("S12", f"Train tokenizer ({config.display_name})", runlog.BLOCKED,
            "final train/val splits do not exist yet.")
        return 1

    import yaml
    tok_cfg = yaml.safe_load((config.configs_dir / "tokenizer.yaml").read_text("utf-8"))

    model_type = tok_cfg.get("model_type", "bpe")
    model_prefix_name = tok_cfg.get("model_prefix", f"{config.lang}_bpe")
    character_coverage = float(tok_cfg.get("character_coverage", 0.9995))
    byte_fallback = bool(tok_cfg.get("byte_fallback", False))
    vocab_sizes = sorted(int(v) for v in tok_cfg.get("vocab_sizes", [16000, 24000, 32000, 48000]))
    preferred_min_vocab = int(tok_cfg.get("preferred_min_vocab", 24000))
    selection_cfg = tok_cfg.get("selection", {}) or {}
    max_unk_rate = float(selection_cfg.get("max_unk_rate", 0.001))
    min_relative_gain = float(selection_cfg.get("min_relative_gain", 0.03))
    training_cfg = tok_cfg.get("training", {}) or {}
    special_tokens = tok_cfg.get("special_tokens", {}) or {}
    compare_unigram_cfg = tok_cfg.get("compare_unigram", {}) or {}

    print(
        f"S12 train tokenizer ({config.display_name}) -- model_type={model_type} "
        f"vocab_sizes={vocab_sizes} coverage={character_coverage}",
        flush=True,
    )

    # Val sample for measurement, loaded once and reused across candidates.
    val_texts: list[str] = []
    for record in read_jsonl(val_path):
        val_texts.append(record["text"])
        if len(val_texts) >= args.val_sample:
            break
    print(f"  measuring candidates against {len(val_texts):,} val documents", flush=True)

    sweep_dir = config.tokenizer_dir / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict] = []

    for vocab_size in vocab_sizes:
        t0 = time.time()
        model_prefix_path = sweep_dir / f"{model_prefix_name}_{model_type}_{vocab_size}"
        print(f"  training {model_type} @ {vocab_size:,}...", flush=True)
        model_path = _train_one(
            model_type=model_type, vocab_size=vocab_size, train_path=train_path,
            model_path_prefix=model_prefix_path, character_coverage=character_coverage,
            byte_fallback=byte_fallback, training_cfg=training_cfg, special_tokens=special_tokens,
        )
        import sentencepiece as spm
        processor = spm.SentencePieceProcessor(model_file=str(model_path))
        m = tokeval.measure(processor, val_texts)
        candidates.append({
            "model_type": model_type,
            "vocab_size": vocab_size,
            "fertility": m["fertility"],
            "unk_rate": m["unk_rate"],
            "chars_per_token": m["chars_per_token"],
            "model_path": str(model_path),
            "seconds": round(time.time() - t0, 1),
        })
        print(
            f"    fertility={m['fertility']:.4f}  unk_rate={m['unk_rate']:.4%}  "
            f"({time.time()-t0:.0f}s)", flush=True,
        )

    # -- fertility-elbow selection ------------------------------------------
    eligible = [c for c in candidates if c["unk_rate"] <= max_unk_rate]
    selection_notes = []
    if not eligible:
        eligible = sorted(candidates, key=lambda c: c["unk_rate"])[:1]
        selection_notes.append(
            f"no candidate met max_unk_rate={max_unk_rate:.3%}; falling back to the "
            f"lowest-UNK candidate ({eligible[0]['vocab_size']:,})."
        )

    chosen = eligible[-1]  # default: largest eligible, if no elbow found
    for i in range(len(eligible) - 1):
        cur, nxt = eligible[i], eligible[i + 1]
        if cur["fertility"] <= 0:
            continue
        relative_gain = (cur["fertility"] - nxt["fertility"]) / cur["fertility"]
        if relative_gain < min_relative_gain:
            chosen = cur
            selection_notes.append(
                f"elbow at {cur['vocab_size']:,}: growing to {nxt['vocab_size']:,} would "
                f"buy only {relative_gain:.1%} relative fertility gain "
                f"(< {min_relative_gain:.0%} threshold)."
            )
            break
    else:
        selection_notes.append(
            f"no elbow found within the swept range; fertility was still improving by "
            f"more than {min_relative_gain:.0%} at every step, so the largest candidate "
            f"({chosen['vocab_size']:,}) was selected."
        )

    if chosen["vocab_size"] < preferred_min_vocab:
        promoted = [c for c in eligible if c["vocab_size"] >= preferred_min_vocab]
        if promoted:
            selection_notes.append(
                f"elbow selection ({chosen['vocab_size']:,}) is below preferred_min_vocab "
                f"({preferred_min_vocab:,}); overridden to {promoted[0]['vocab_size']:,} per "
                f"the brief's 'tens of thousands' guidance."
            )
            chosen = promoted[0]
        else:
            selection_notes.append(
                f"elbow selection ({chosen['vocab_size']:,}) is below preferred_min_vocab "
                f"({preferred_min_vocab:,}), and no swept candidate reaches it -- kept the "
                f"elbow choice; widen vocab_sizes in tokenizer.yaml if this needs revisiting."
            )

    # -- Nepali-only: compare against Unigram at the chosen size ------------
    if compare_unigram_cfg.get("enabled") and chosen["fertility"] > float(
        compare_unigram_cfg.get("trigger_fertility_above", 999)
    ):
        print(
            f"  BPE fertility {chosen['fertility']:.4f} exceeds "
            f"{compare_unigram_cfg['trigger_fertility_above']} -- training Unigram at "
            f"{chosen['vocab_size']:,} for comparison...",
            flush=True,
        )
        uni_prefix = sweep_dir / f"{model_prefix_name}_unigram_{chosen['vocab_size']}"
        uni_model_path = _train_one(
            model_type="unigram", vocab_size=chosen["vocab_size"], train_path=train_path,
            model_path_prefix=uni_prefix, character_coverage=character_coverage,
            byte_fallback=byte_fallback, training_cfg=training_cfg, special_tokens=special_tokens,
        )
        import sentencepiece as spm
        uni_processor = spm.SentencePieceProcessor(model_file=str(uni_model_path))
        uni_m = tokeval.measure(uni_processor, val_texts)
        selection_notes.append(
            f"Unigram @ {chosen['vocab_size']:,}: fertility={uni_m['fertility']:.4f} "
            f"unk_rate={uni_m['unk_rate']:.4%} vs BPE fertility={chosen['fertility']:.4f} "
            f"unk_rate={chosen['unk_rate']:.4%}."
        )
        if uni_m["fertility"] < chosen["fertility"] and uni_m["unk_rate"] <= max_unk_rate:
            selection_notes.append("Unigram beat BPE on fertility within the UNK ceiling -- selected.")
            chosen = {
                "model_type": "unigram", "vocab_size": chosen["vocab_size"],
                "fertility": uni_m["fertility"], "unk_rate": uni_m["unk_rate"],
                "chars_per_token": uni_m["chars_per_token"], "model_path": str(uni_model_path),
            }
        else:
            selection_notes.append("BPE retained -- Unigram did not improve fertility within the UNK ceiling.")

    # -- install the chosen model as the language's tokenizer ---------------
    final_prefix = config.tokenizer_dir / model_prefix_name
    final_model = Path(str(final_prefix) + ".model")
    final_vocab = Path(str(final_prefix) + ".vocab")
    chosen_model_path = Path(chosen["model_path"])
    final_model.write_bytes(chosen_model_path.read_bytes())
    final_vocab.write_bytes(chosen_model_path.with_suffix(".vocab").read_bytes())

    elapsed = time.time() - started
    payload = {
        "chosen": {k: v for k, v in chosen.items() if k != "model_path"},
        "chosen_model_path": str(final_model),
        "candidates": candidates,
        "selection_notes": selection_notes,
        "max_unk_rate": max_unk_rate,
        "min_relative_gain": min_relative_gain,
        "preferred_min_vocab": preferred_min_vocab,
        "model_sha256_16": _file_sha256(final_model),
        "seconds": round(elapsed, 1),
    }
    atomic_write_json(config.tokenizer_dir / "selection_report.json", payload)

    print(f"\n  SELECTED: {chosen['model_type']} @ {chosen['vocab_size']:,}  "
          f"fertility={chosen['fertility']:.4f}  unk_rate={chosen['unk_rate']:.4%}")
    print(f"  installed at {final_model}")
    for note in selection_notes:
        print(f"  - {note}")

    log(
        "S12",
        f"Train tokenizer ({config.display_name})",
        runlog.SUCCESS,
        f"Swept {len(candidates)} candidates; selected {chosen['model_type']} @ "
        f"{chosen['vocab_size']:,} (fertility {chosen['fertility']:.4f}, "
        f"UNK {chosen['unk_rate']:.4%}).",
        metrics=payload,
        decision="; ".join(selection_notes),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

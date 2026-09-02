"""Pack Model H's (hindi) train/val/test splits into uint16 token memmaps.

This file belongs to **Model H** (hindi). Model L (nepali) has its
own independent copy under ``nepali/data/pack_corpus.py``; neither reads the
other's corpus, tokenizer or vocabulary.

Phase 2 trains from contiguous binary token streams rather than from the Phase 1
text corpus directly, for two reasons:

1. **Speed.** Re-reading and re-tokenizing the corpus every epoch would dominate
   training time. Tokenization is a one-time cost paid here instead.
2. **Portability.** Pretraining runs on Kaggle. The text corpus is many
   gigabytes; the packed token stream is ~1GB, because each token is 2 bytes
   rather than several bytes of UTF-8 Devanagari.

Hindi's Phase 1 splits are stored as one ``.txt`` file per document — roughly
676,000 of them for the train split. The worker processes open the files
themselves so that the per-file IO cost is parallelised, which is the dominant
cost for this language.

This language's vocabulary (SentencePiece Unigram, 8,000 tokens) is far below 65,536, so ``uint16`` is
lossless. A sidecar ``<split>.json`` records the exact token count, the
tokenizer that produced it, the document count and the unknown-token rate, so
the report can cite measured numbers rather than re-deriving them.

Usage
-----
    python hindi/data/pack_corpus.py
    python hindi/data/pack_corpus.py --splits train
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import sentencepiece as spm

REPO_ROOT = Path(__file__).resolve().parents[2]
LANGUAGE = "hindi"

# uint16 is only valid while the vocabulary stays below this bound.
DTYPE = np.uint16
MAX_VOCAB_FOR_DTYPE = 2 ** 16

TOKENIZER_MODEL = REPO_ROOT / "hindi/tokenizer/hindi_unigram_8000.model"
OUT_DIR = REPO_ROOT / "hindi/data/packed"
READER = "txt_dir"
SHUFFLE_DOCS = True
TEXT_KEY = "text"

SPLIT_PATHS = {
        "train": REPO_ROOT / "hindi/data/splits/train/text",
        "val": REPO_ROOT / "hindi/data/splits/validation/text",
        "test": REPO_ROOT / "hindi/data/splits/test/text",
}

# Documents are shipped to workers in batches of this size; larger chunks
# amortise the pickling round-trip, smaller ones keep memory flat.
CHUNK_SIZE = 512

# Tokens buffered before flushing to disk (~64MB).
FLUSH_EVERY = 32_000_000


def iter_txt_dir(path: Path, shuffle: bool, seed: int) -> tuple[Iterator[str], int]:
    """Yield file paths for a directory-of-``.txt``-files split.

    Args:
        path: Directory containing one ``.txt`` file per document.
        shuffle: Whether to shuffle the listing with a fixed seed.
        seed: Seed for the shuffle, so packing is reproducible.

    Returns:
        A ``(iterator, count)`` pair; the iterator yields file path strings.
    """
    files = [str(p) for p in path.iterdir() if p.suffix == ".txt"]
    if shuffle:
        random.Random(seed).shuffle(files)
    return iter(files), len(files)


def iter_jsonl(path: Path) -> tuple[Iterator[str], int]:
    """Yield raw JSONL lines for a single-file split.

    Args:
        path: The ``.jsonl`` file.

    Returns:
        A ``(iterator, count)`` pair; ``count`` is ``-1`` because counting lines
        up front would mean reading the whole file twice.
    """

    def _gen() -> Iterator[str]:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield line

    return _gen(), -1


_SP: spm.SentencePieceProcessor | None = None
_EOS: int = -1


def _init_worker(model_path: str) -> None:
    """Load this worker's SentencePiece model once, at process start.

    SentencePiece processors are not picklable, so each worker loads its own
    copy rather than receiving one per task.

    Args:
        model_path: Path to this language's ``.model`` file.
    """
    global _SP, _EOS
    _SP = spm.SentencePieceProcessor(model_file=model_path)
    _EOS = _SP.eos_id()


def _encode(task: str) -> list[int]:
    """Tokenize one document and terminate it with the EOS id.

    The EOS token is what lets the model learn document boundaries in an
    otherwise undelimited concatenated stream.

    Args:
        task: A file path (``txt_dir``) or a raw JSONL line (``jsonl``).

    Returns:
        Token ids followed by EOS; empty if the document had no usable text.
    """
    assert _SP is not None
    if READER == "txt_dir":
        try:
            with open(task, "r", encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            return []
    else:
        try:
            text = json.loads(task).get(TEXT_KEY, "")
        except json.JSONDecodeError:
            return []
    text = text.strip()
    if not text:
        return []
    ids = _SP.encode(text, out_type=int)
    if _EOS >= 0:
        ids.append(_EOS)
    return ids


class CorpusPacker:
    """Tokenizes this language's splits into flat uint16 streams on disk."""

    def __init__(self, workers: int, seed: int = 20260902) -> None:
        """Prepare a packer and validate the tokenizer against the uint16 bound.

        Args:
            workers: Number of tokenizer worker processes.
            seed: Seed for document shuffling.

        Raises:
            FileNotFoundError: If the tokenizer model is missing.
            ValueError: If the vocabulary is too large for uint16.
        """
        self.workers = workers
        self.seed = seed
        if not TOKENIZER_MODEL.exists():
            raise FileNotFoundError(f"tokenizer model not found: {TOKENIZER_MODEL}")
        sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
        self.vocab_size = sp.vocab_size()
        self.eos_id = sp.eos_id()
        self.bos_id = sp.bos_id()
        self.unk_id = sp.unk_id()
        self.pad_id = sp.pad_id()
        if self.vocab_size > MAX_VOCAB_FOR_DTYPE:
            raise ValueError(
                f"vocab_size={self.vocab_size} exceeds uint16 range; widen DTYPE"
            )
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _tasks(self, split: str) -> tuple[Iterator[str], int]:
        """Build the document iterator for one split.

        Args:
            split: ``"train"``, ``"val"`` or ``"test"``.

        Returns:
            A ``(iterator, expected_count)`` pair; count is ``-1`` if unknown.

        Raises:
            FileNotFoundError: If the split is missing.
        """
        path = SPLIT_PATHS[split]
        if not path.exists():
            raise FileNotFoundError(f"{LANGUAGE}/{split} not found at {path}")
        if READER == "txt_dir":
            return iter_txt_dir(path, SHUFFLE_DOCS, self.seed)
        return iter_jsonl(path)

    def pack_split(self, split: str) -> dict:
        """Tokenize one split and write ``<split>.bin`` plus its JSON sidecar.

        Args:
            split: ``"train"``, ``"val"`` or ``"test"``.

        Returns:
            The statistics dict that was written to the sidecar.
        """
        out_bin = OUT_DIR / f"{split}.bin"
        out_json = OUT_DIR / f"{split}.json"
        tasks, expected = self._tasks(split)

        started = time.time()
        total_tokens = total_docs = empty_docs = unk_tokens = 0
        buffer: list[np.ndarray] = []
        buffered = 0

        print(
            f"[{LANGUAGE}/{split}] packing -> {out_bin} "
            f"(vocab={self.vocab_size}, eos={self.eos_id}, workers={self.workers})",
            flush=True,
        )

        with open(out_bin, "wb") as sink:
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                processes=self.workers,
                initializer=_init_worker,
                initargs=(str(TOKENIZER_MODEL),),
            ) as pool:
                for ids in pool.imap(_encode, tasks, chunksize=CHUNK_SIZE):
                    total_docs += 1
                    if not ids:
                        empty_docs += 1
                        continue
                    arr = np.asarray(ids, dtype=DTYPE)
                    if self.unk_id >= 0:
                        unk_tokens += int(np.count_nonzero(arr == self.unk_id))
                    buffer.append(arr)
                    buffered += arr.size
                    total_tokens += arr.size
                    if buffered >= FLUSH_EVERY:
                        np.concatenate(buffer).tofile(sink)
                        buffer, buffered = [], 0
                        elapsed = time.time() - started
                        rate = total_tokens / max(elapsed, 1e-9)
                        print(
                            f"  {total_docs:,} docs / {total_tokens:,} tokens "
                            f"({rate/1e6:.2f}M tok/s, {elapsed/60:.1f} min)",
                            flush=True,
                        )
                if buffer:
                    np.concatenate(buffer).tofile(sink)

        elapsed = time.time() - started
        stats = {
            "language": LANGUAGE,
            "split": split,
            "tokens": total_tokens,
            "documents": total_docs,
            "empty_documents": empty_docs,
            "unk_tokens": unk_tokens,
            "unk_rate": (unk_tokens / total_tokens) if total_tokens else 0.0,
            "expected_documents": expected,
            "dtype": "uint16",
            "bytes_on_disk": out_bin.stat().st_size,
            "tokenizer_model": "hindi/tokenizer/hindi_unigram_8000.model",
            "vocab_size": self.vocab_size,
            "eos_id": self.eos_id,
            "bos_id": self.bos_id,
            "unk_id": self.unk_id,
            "pad_id": self.pad_id,
            "documents_shuffled": SHUFFLE_DOCS,
            "shuffle_seed": self.seed if SHUFFLE_DOCS else None,
            "seconds": round(elapsed, 1),
        }
        out_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(
            f"[{LANGUAGE}/{split}] done: {total_tokens:,} tokens from "
            f"{total_docs:,} docs in {elapsed/60:.1f} min "
            f"({stats['bytes_on_disk']/1e9:.2f} GB)",
            flush=True,
        )
        return stats


def main() -> None:
    """Parse arguments and pack the requested splits for this language."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test", "train"],
        choices=["train", "val", "test"],
        help="Splits to pack. Default packs the small ones first so failures "
        "surface in seconds rather than after the long train pass.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Tokenizer worker processes.",
    )
    args = parser.parse_args()

    packer = CorpusPacker(workers=args.workers)
    summary = {split: packer.pack_split(split) for split in args.splits}
    total = sum(s["tokens"] for s in summary.values())
    print(f"\n[{LANGUAGE}] packed {total:,} tokens across {len(summary)} split(s)")


if __name__ == "__main__":
    main()

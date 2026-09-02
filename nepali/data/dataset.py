"""Batch sampling over the packed uint16 token streams produced by ``pack_corpus``.

Training reads from a flat memory-mapped array of token ids rather than from
text. A batch is formed by drawing ``batch_size`` random offsets into the stream
and slicing ``block_size + 1`` tokens at each: the first ``block_size`` become
the inputs and the same window shifted one position becomes the targets, which
is exactly the causal language-modelling objective.

Because offsets are drawn uniformly at random, the physical order of documents
in the stream does not affect training. Windows may straddle a document
boundary; the EOS token written between documents is the signal the model uses
to learn where one document ends.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class TokenStream:
    """A single packed split: a uint16 memmap plus the sidecar that describes it."""

    def __init__(self, bin_path: str | Path) -> None:
        """Open a packed split for reading.

        Args:
            bin_path: Path to a ``<split>.bin`` file written by
                ``nepali/data/pack_corpus.py``. Its ``<split>.json`` sidecar is read
                alongside, if present.

        Raises:
            FileNotFoundError: If the ``.bin`` file does not exist.
        """
        self.path = Path(bin_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"packed split not found: {self.path}. Run "
                f"`python nepali/data/pack_corpus.py` first."
            )
        self.meta: dict = {}
        sidecar = self.path.with_suffix(".json")
        if sidecar.exists():
            self.meta = json.loads(sidecar.read_text(encoding="utf-8"))
        # mode="r" maps the file read-only; the OS pages it in on demand, so a
        # multi-gigabyte stream never has to fit in RAM.
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        """Return the number of tokens in the stream."""
        return int(self.tokens.shape[0])

    @property
    def n_documents(self) -> int:
        """Return the document count recorded by the packer, or ``-1``."""
        return int(self.meta.get("documents", -1))


class BatchSampler:
    """Draws ``(inputs, targets)`` batches from one language's packed splits.

    Holds only that language's streams — Model H and Model L each construct
    their own sampler over their own corpus, and no sampler ever sees both.
    """

    def __init__(
        self,
        data_dir: str | Path,
        block_size: int,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        """Open every packed split found in ``data_dir``.

        Args:
            data_dir: Directory holding ``train.bin`` / ``val.bin`` / ``test.bin``.
            block_size: Context length; each sampled window is this long.
            device: Device to move batches to.
            seed: Seed for the training-batch RNG.

        Raises:
            FileNotFoundError: If no packed split is present in ``data_dir``.
        """
        self.data_dir = Path(data_dir)
        self.block_size = block_size
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.streams: dict[str, TokenStream] = {}
        for split in ("train", "val", "test"):
            path = self.data_dir / f"{split}.bin"
            if path.exists():
                self.streams[split] = TokenStream(path)
        if not self.streams:
            raise FileNotFoundError(f"no packed splits found under {self.data_dir}")

    def available(self) -> list[str]:
        """Return the names of the splits that were found."""
        return sorted(self.streams)

    def _slice(self, split: str, offsets: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialise a batch from explicit offsets.

        Args:
            split: Split name.
            offsets: Start indices, one per sequence in the batch.

        Returns:
            An ``(x, y)`` pair of ``(B, block_size)`` int64 tensors, where ``y``
            is ``x`` shifted one position later.
        """
        stream = self.streams[split].tokens
        # astype(int64) because nn.Embedding requires a long index tensor;
        # the stream itself stays uint16 on disk.
        x = np.stack(
            [stream[o : o + self.block_size].astype(np.int64) for o in offsets]
        )
        y = np.stack(
            [stream[o + 1 : o + 1 + self.block_size].astype(np.int64) for o in offsets]
        )
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
        if self.device.startswith("cuda"):
            # pin_memory + non_blocking overlaps the host-to-device copy with
            # compute on the previous batch.
            xt = xt.pin_memory().to(self.device, non_blocking=True)
            yt = yt.pin_memory().to(self.device, non_blocking=True)
        else:
            xt, yt = xt.to(self.device), yt.to(self.device)
        return xt, yt

    def _max_offset(self, split: str) -> int:
        """Return the largest valid start index for a window in ``split``.

        Args:
            split: Split name.

        Returns:
            The exclusive upper bound for a start offset, leaving room for
            ``block_size + 1`` tokens.
        """
        return len(self.streams[split]) - self.block_size - 1

    def sample(self, split: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a random training batch.

        Args:
            split: Split name.
            batch_size: Number of sequences.

        Returns:
            An ``(x, y)`` pair of ``(batch_size, block_size)`` tensors.
        """
        offsets = self.rng.integers(0, self._max_offset(split), size=batch_size)
        return self._slice(split, offsets)

    def fixed_batches(
        self, split: str, batch_size: int, n_batches: int, seed: int = 1234
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Build a reproducible list of evaluation batches.

        Validation loss is only comparable across training steps if it is
        measured on the *same* windows every time. Using a separate seeded RNG
        rather than :attr:`rng` also keeps evaluation from perturbing the
        training data order.

        Args:
            split: Split name.
            batch_size: Sequences per batch.
            n_batches: Number of batches.
            seed: Seed fixing which windows are used.

        Returns:
            A list of ``(x, y)`` pairs.
        """
        rng = np.random.default_rng(seed)
        limit = self._max_offset(split)
        return [
            self._slice(split, rng.integers(0, limit, size=batch_size))
            for _ in range(n_batches)
        ]

    def sequential_batches(
        self, split: str, batch_size: int, max_tokens: int | None = None
    ):
        """Iterate the split once in order, in non-overlapping windows.

        Used for held-out perplexity and bits-per-byte, where every token should
        be scored exactly once rather than sampled with replacement.

        Args:
            split: Split name.
            batch_size: Sequences per batch.
            max_tokens: Optional cap on how many tokens to cover, for a faster
                approximate pass.

        Yields:
            ``(x, y)`` pairs of ``(<=batch_size, block_size)`` tensors.
        """
        total = len(self.streams[split]) - 1
        if max_tokens is not None:
            total = min(total, max_tokens)
        stride = self.block_size
        starts = list(range(0, total - stride, stride))
        for i in range(0, len(starts), batch_size):
            chunk = np.asarray(starts[i : i + batch_size])
            if chunk.size:
                yield self._slice(split, chunk)

    def describe(self) -> dict[str, dict]:
        """Summarise every open split, for logging and the report.

        Returns:
            Mapping from split name to its token count, document count and the
            tokenizer that produced it.
        """
        return {
            name: {
                "tokens": len(stream),
                "documents": stream.n_documents,
                "tokenizer_model": stream.meta.get("tokenizer_model"),
                "vocab_size": stream.meta.get("vocab_size"),
            }
            for name, stream in self.streams.items()
        }

"""Turn Model L's Nepali reasoning JSONL into batches the Transformer can train on.

This is supervised finetuning on top of a causal language model, so the format
is deliberately plain: prompt and answer are concatenated into one token
sequence and the model is trained to continue it. Two details matter.

**Loss is computed on the answer only.** The prompt's target positions are set
to ``-1``, which the model's ``cross_entropy(..., ignore_index=-1)`` already
skips. Training on the prompt too would spend most of the gradient teaching the
model to reproduce question templates it will always be given for free at test
time, and the thing being measured -- can it name the right entity -- would get
a small share of the signal.

**Batches are bucketed by prompt length rather than padded to a common one.**
The model uses *learned absolute* positional embeddings, so a token's
representation depends on its index in the sequence. Left-padding would shift
every real token to a position it never occupied during pretraining;
right-padding is fine for training but breaks generation, which appends at the
end. Grouping examples whose prompts tokenize to the same length sidesteps both
problems: every sequence in a batch is genuinely aligned, and no padding is
needed for the prompt at all.

This file belongs to **Model L** (Nepali). Model H has its own independent copy
under ``hindi/reasoning/``; no data, tokenizer or vocabulary is shared.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

LANGUAGE = "nepali"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_TOKENIZER = REPO_ROOT / "nepali/tokenizer/nepali_bpe.model"


def load_tokenizer(model_path: str | Path | None = None):
    """Open this language's SentencePiece model.

    Args:
        model_path: Override for the tokenizer model; defaults to Model L's.

    Returns:
        A loaded ``SentencePieceProcessor``.

    Raises:
        FileNotFoundError: If the model file is missing.
    """
    import sentencepiece as spm

    path = Path(model_path) if model_path else DEFAULT_TOKENIZER
    if not path.is_file():
        raise FileNotFoundError(f"tokenizer model not found: {path}")
    return spm.SentencePieceProcessor(model_file=str(path))


class ReasoningDataset:
    """One split of the reasoning corpus, tokenized and ready to batch."""

    def __init__(
        self,
        split: str,
        tokenizer=None,
        data_dir: str | Path | None = None,
        tokenizer_model: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        """Load and tokenize one split.

        Args:
            split: ``"train"``, ``"val"`` or ``"test"``.
            tokenizer: An already-loaded SentencePiece processor, reused across
                splits so the model file is only read once.
            data_dir: Directory holding the JSONL splits.
            tokenizer_model: Path to the tokenizer, if ``tokenizer`` is None.
            device: Device the produced tensors should live on.

        Raises:
            FileNotFoundError: If the split's JSONL file is missing.
        """
        self.split = split
        self.device = device
        self.sp = tokenizer or load_tokenizer(tokenizer_model)
        self.eos_id = self.sp.eos_id()
        # Nepali's BPE and Hindi's Unigram assign different ids to their special
        # tokens, so nothing here may be hardcoded. Padding ids never contribute
        # to the loss (their targets are -1); when the tokenizer declares no pad
        # token, reusing EOS is harmless for the same reason.
        self.pad_id = self.sp.pad_id() if self.sp.pad_id() >= 0 else self.eos_id

        directory = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        path = directory / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found - run "
                f"`python {LANGUAGE}/reasoning/generate.py` first"
            )
        self.rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        for row in self.rows:
            self._encode(row)

    def _encode(self, row: dict) -> None:
        """Tokenize one example in place.

        The answer is encoded with a leading space so SentencePiece emits it as
        a word-initial piece (``▁बिक्रम``), which is how the name appears in
        running text. Encoding it bare would produce a word-*internal* piece
        that the model has rarely seen in that position.

        Args:
            row: One dataset record; gains ``prompt_ids`` and ``answer_ids``.
        """
        row["prompt_ids"] = self.sp.encode(row["prompt"])
        row["answer_ids"] = self.sp.encode(" " + row["answer"]) + [self.eos_id]

    def __len__(self) -> int:
        """Return the number of examples in this split."""
        return len(self.rows)

    def sequence_stats(self) -> dict:
        """Summarise tokenized lengths, for the report's dataset table.

        Returns:
            Mean and maximum prompt / answer / total token counts.
        """
        prompts = [len(r["prompt_ids"]) for r in self.rows]
        answers = [len(r["answer_ids"]) for r in self.rows]
        totals = [p + a for p, a in zip(prompts, answers)]
        return {
            "examples": len(self.rows),
            "prompt_tokens_mean": round(sum(prompts) / len(prompts), 2),
            "prompt_tokens_max": max(prompts),
            "answer_tokens_mean": round(sum(answers) / len(answers), 2),
            "answer_tokens_max": max(answers),
            "total_tokens_max": max(totals),
            "total_tokens": sum(totals),
        }

    def _tensor_pair(self, rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the ``(inputs, targets)`` pair for a group of examples.

        Every sequence is ``prompt + answer``. Position ``i`` of the input
        predicts token ``i + 1``, so the target at ``i`` is masked whenever
        token ``i + 1`` still belongs to the prompt. The first unmasked target
        is therefore the first answer token, predicted from the final prompt
        token -- exactly the decision the evaluation measures.

        Args:
            rows: Examples to batch. Prompts must already share a length.

        Returns:
            Inputs of shape ``(B, T)`` and targets of shape ``(B, T)``, where
            masked target positions hold ``-1``.
        """
        sequences = [r["prompt_ids"] + r["answer_ids"] for r in rows]
        width = max(len(s) for s in sequences)
        inputs, targets = [], []
        for row, seq in zip(rows, sequences):
            padded = seq + [self.pad_id] * (width - len(seq))
            x = padded[:-1]
            y = padded[1:]
            # -1 everywhere the next token is not part of the answer: the whole
            # prompt, and any padding past the real end of this sequence.
            start = len(row["prompt_ids"]) - 1
            end = len(seq) - 1
            y = [t if start <= i < end else -1 for i, t in enumerate(y)]
            inputs.append(x)
            targets.append(y)
        return (
            torch.tensor(inputs, dtype=torch.long, device=self.device),
            torch.tensor(targets, dtype=torch.long, device=self.device),
        )

    def batches(
        self, batch_size: int, shuffle: bool = True, seed: int = 0
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Build training batches, grouped so each batch shares a prompt length.

        Args:
            batch_size: Examples per batch.
            shuffle: Whether to shuffle within buckets and across batches.
            seed: Seed for that shuffling, so an epoch is reproducible.

        Returns:
            A list of ``(inputs, targets)`` pairs covering the whole split.
        """
        buckets: dict[int, list[dict]] = defaultdict(list)
        for row in self.rows:
            buckets[len(row["prompt_ids"])].append(row)

        rng = random.Random(seed)
        out = []
        for length in sorted(buckets):
            group = list(buckets[length])
            if shuffle:
                rng.shuffle(group)
            for i in range(0, len(group), batch_size):
                out.append(self._tensor_pair(group[i : i + batch_size]))
        if shuffle:
            rng.shuffle(out)
        return out

    def prompt_batches(
        self, batch_size: int
    ) -> list[tuple[torch.Tensor, list[dict]]]:
        """Build prompt-only batches for generation, bucketed by length.

        Generation appends tokens at the end of the sequence, so every prompt in
        a batch must end at the same index. Bucketing by token length gives that
        exactly, with no padding and therefore no positional distortion.

        Args:
            batch_size: Prompts per batch.

        Returns:
            A list of ``(prompt_ids, rows)`` pairs, where ``prompt_ids`` has
            shape ``(B, T)`` and ``rows`` are the matching dataset records.
        """
        buckets: dict[int, list[dict]] = defaultdict(list)
        for row in self.rows:
            buckets[len(row["prompt_ids"])].append(row)

        out = []
        for length in sorted(buckets):
            group = buckets[length]
            for i in range(0, len(group), batch_size):
                chunk = group[i : i + batch_size]
                ids = torch.tensor(
                    [r["prompt_ids"] for r in chunk],
                    dtype=torch.long,
                    device=self.device,
                )
                out.append((ids, chunk))
        return out

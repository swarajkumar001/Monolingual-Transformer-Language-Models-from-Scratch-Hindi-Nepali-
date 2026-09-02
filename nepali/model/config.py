"""Model and training configuration objects, loadable from and dumpable to YAML.

Every run's full configuration is written next to its checkpoints, so a run can
be reproduced (and resumed) from the repository alone, as the spec requires.
The two languages use *separate* config files with different vocabulary sizes;
nothing here is shared at runtime between Model H and Model L.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# Positional-encoding schemes this implementation supports. "none" exists for
# the optional Phase 2 bonus ablation (retrain with position information
# removed) — it is a config flag rather than a separate model class so the
# ablation runs through exactly the same code path as the standard model.
POS_ENCODINGS = ("learned", "sinusoidal", "none")


@dataclass
class GPTConfig:
    """Architecture hyperparameters for one decoder-only Transformer.

    The defaults target the spec's ~25M trainable parameters. With
    ``n_embd=512`` and ``n_layer=6`` the block stack contributes roughly
    ``12 * n_embd^2`` parameters per layer (``4 * n_embd^2`` for the four
    attention projections, ``8 * n_embd^2`` for the two feed-forward layers at
    ``d_ff = 4 * n_embd``), giving ~18.9M, on top of a ``vocab_size * n_embd``
    embedding table that differs between the two languages because their
    vocabularies differ.

    Attributes:
        vocab_size: Size of this language's tokenizer vocabulary. Differs
            between Model H (8,000) and Model L (10,000); the embedding and
            output layers therefore differ in size between the two models.
        block_size: Maximum context length in tokens. With learned absolute
            positional embeddings this is a hard ceiling: the model has no
            embedding row for a position beyond ``block_size``, so it cannot be
            evaluated on longer sequences without interpolating or retraining
            the positional table.
        n_layer: Number of stacked Transformer blocks.
        n_head: Number of attention heads. Must divide ``n_embd``.
        n_embd: Model dimension ``d``.
        d_ff: Inner width of the position-wise feed-forward network. Defaults to
            ``4 * n_embd`` when left as ``None``.
        dropout: Dropout probability used for embeddings, attention
            probabilities and sublayer outputs.
        bias: Whether linear layers and LayerNorms carry bias terms.
        tie_weights: Whether the output projection reuses the input embedding
            matrix. Saves ``vocab_size * n_embd`` parameters (4.1M for Hindi,
            5.1M for Nepali) and ties the two representations of a token.
        pos_encoding: One of :data:`POS_ENCODINGS`.
        init_std: Standard deviation for normal weight initialisation.
    """

    vocab_size: int
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    d_ff: int | None = None
    dropout: float = 0.1
    bias: bool = True
    tie_weights: bool = True
    pos_encoding: str = "learned"
    init_std: float = 0.02

    def __post_init__(self) -> None:
        """Fill in derived defaults and reject invalid combinations early.

        Raises:
            ValueError: If the head count does not divide the model dimension,
                or an unknown positional-encoding scheme is requested.
        """
        if self.d_ff is None:
            self.d_ff = 4 * self.n_embd
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd={self.n_embd} must be divisible by n_head={self.n_head}; "
                "every head needs an equal-width d_k = n_embd / n_head subspace"
            )
        if self.pos_encoding not in POS_ENCODINGS:
            raise ValueError(
                f"pos_encoding={self.pos_encoding!r} not in {POS_ENCODINGS}"
            )

    @property
    def head_dim(self) -> int:
        """Width ``d_k`` of each attention head's subspace."""
        return self.n_embd // self.n_head

    def estimated_parameters(self) -> dict[str, int]:
        """Analytically estimate the parameter count, broken down by component.

        This is a paper estimate used for sizing configs before instantiating a
        model; :meth:`~nepali.model.transformer.GPT.parameter_count` reports the true count
        from the built module. The two should agree to within the bias terms.

        Returns:
            Mapping from component name to parameter count, including a
            ``"total"`` key.
        """
        d, v, L = self.n_embd, self.vocab_size, self.n_layer
        d_ff = self.d_ff or 4 * d
        token_emb = v * d
        pos_emb = self.block_size * d if self.pos_encoding == "learned" else 0
        attn_per_layer = 4 * d * d + (4 * d if self.bias else 0)
        ffn_per_layer = 2 * d * d_ff + ((d_ff + d) if self.bias else 0)
        norm_per_layer = 2 * (2 * d if self.bias else d)
        blocks = L * (attn_per_layer + ffn_per_layer + norm_per_layer)
        final_norm = 2 * d if self.bias else d
        head = 0 if self.tie_weights else v * d
        total = token_emb + pos_emb + blocks + final_norm + head
        return {
            "token_embedding": token_emb,
            "positional_embedding": pos_emb,
            "blocks": blocks,
            "final_norm": final_norm,
            "lm_head": head,
            "total": total,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return this config as a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GPTConfig":
        """Build a config from a dictionary, ignoring unknown keys.

        Args:
            data: Mapping of field names to values.

        Returns:
            The constructed :class:`GPTConfig`.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class TrainConfig:
    """Optimisation, schedule, checkpointing and data-location settings.

    Attributes:
        language: Language name, used for run naming and output paths.
        data_dir: Directory holding ``train.bin`` / ``val.bin`` / ``test.bin``.
        out_dir: Directory for checkpoints and the run's config snapshot.
        max_steps: Total optimiser steps.
        batch_size: Sequences per micro-batch (per forward pass).
        grad_accum_steps: Micro-batches accumulated per optimiser step. The
            effective batch is ``batch_size * grad_accum_steps * block_size``
            tokens; this is the knob that decouples the effective batch from
            available VRAM.
        learning_rate: Peak learning rate after warmup.
        min_lr: Floor the cosine schedule decays to.
        warmup_steps: Linear warmup length in steps.
        weight_decay: AdamW decoupled weight decay, applied only to matrices
            (not to biases, LayerNorm gains or embeddings).
        beta1: AdamW first-moment decay.
        beta2: AdamW second-moment decay.
        grad_clip: Global gradient-norm clip; ``0`` disables clipping.
        eval_interval: Steps between validation passes.
        eval_iters: Micro-batches averaged per validation pass.
        log_interval: Steps between training-loss log lines.
        checkpoint_interval: Steps between checkpoint writes.
        keep_last_n: How many rolling checkpoints to retain on disk.
        seed: Base RNG seed.
        device: ``"cuda"``, ``"cpu"``, or ``"auto"``.
        dtype: Autocast dtype: ``"bfloat16"``, ``"float16"`` or ``"float32"``.
        compile_model: Whether to wrap the model in ``torch.compile``.
    """

    language: str
    data_dir: str
    out_dir: str
    max_steps: int = 20_000
    batch_size: int = 16
    grad_accum_steps: int = 8
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 20
    checkpoint_interval: int = 500
    keep_last_n: int = 2
    seed: int = 20260902
    device: str = "auto"
    dtype: str = "bfloat16"
    compile_model: bool = False

    def tokens_per_step(self, block_size: int) -> int:
        """Return the number of tokens consumed by one optimiser step.

        Args:
            block_size: The model's context length.

        Returns:
            ``batch_size * grad_accum_steps * block_size``.
        """
        return self.batch_size * self.grad_accum_steps * block_size

    def to_dict(self) -> dict[str, Any]:
        """Return this config as a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        """Build a config from a dictionary, ignoring unknown keys.

        Args:
            data: Mapping of field names to values.

        Returns:
            The constructed :class:`TrainConfig`.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class RunConfig:
    """A complete run: one architecture plus one optimisation setup."""

    model: GPTConfig
    train: TrainConfig
    notes: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        """Read a run configuration from a YAML file.

        Args:
            path: Path to a YAML file with ``model:`` and ``train:`` sections.

        Returns:
            The parsed :class:`RunConfig`.
        """
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            model=GPTConfig.from_dict(data["model"]),
            train=TrainConfig.from_dict(data["train"]),
            notes=data.get("notes", ""),
        )

    def save(self, path: str | Path) -> None:
        """Write this run configuration to a YAML file.

        Args:
            path: Destination path; parent directories are created.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model.to_dict(),
            "train": self.train.to_dict(),
            "notes": self.notes,
        }
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    def to_json(self) -> str:
        """Return this run configuration as an indented JSON string."""
        return json.dumps(
            {
                "model": self.model.to_dict(),
                "train": self.train.to_dict(),
                "notes": self.notes,
            },
            indent=2,
        )

"""Finetune Model H on its own Hindi reasoning corpus.

The protocol the specification fixes is followed exactly:

* training starts from **this language's** pretrained checkpoint, never from
  the other model and never from scratch;
* the tokenizer and vocabulary are the ones Phase 1 produced and are not
  touched, so the embedding and output matrices keep their pretrained rows;
* every hyperparameter is recorded in a config file that is written next to the
  checkpoints;
* checkpoints carry model weights, optimizer state, scheduler state, the step
  counter and the config -- the same resume-capable format as pretraining, so
  the Phase 2 evaluation and attention tooling loads a finetuned checkpoint
  without modification. That reuse is what deliverable 4 asks for.

Loss is computed on answer tokens only; see ``dataset.py`` for why.

Usage
-----
    python -m hindi.reasoning.finetune --config hindi/configs/finetune.yaml
    python -m hindi.reasoning.finetune --config ... --no-resume --max-minutes 45
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from ..model.config import GPTConfig
from ..model.transformer import GPT
from ..train.train import CosineWarmupSchedule, resolve_device, resolve_dtype
from .dataset import ReasoningDataset, load_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]

LANGUAGE = "hindi"


@dataclass
class FinetuneConfig:
    """Every knob of the reasoning finetuning run, logged alongside the weights.

    Attributes:
        language: Language name, used for run naming and output paths.
        pretrained_checkpoint: Path to this language's Phase 2 checkpoint. The
            run starts from these weights.
        data_dir: Directory holding the reasoning JSONL splits.
        tokenizer_model: Phase 1 tokenizer, held fixed throughout.
        out_dir: Directory for finetuned checkpoints and the config snapshot.
        epochs: Passes over the training split.
        batch_size: Examples per optimiser step.
        learning_rate: Peak learning rate after warmup. Well below the
            pretraining peak: the model already has a language model worth
            preserving, and a large step would wash it out before it learns the
            task.
        min_lr: Floor the cosine schedule decays to.
        warmup_steps: Linear warmup length in steps.
        weight_decay: AdamW decoupled weight decay, matrices only.
        beta1: AdamW first-moment decay.
        beta2: AdamW second-moment decay.
        grad_clip: Global gradient-norm clip; ``0`` disables clipping.
        eval_interval: Steps between validation passes.
        log_interval: Steps between training-loss log lines.
        seed: Base RNG seed.
        device: ``"cuda"``, ``"cpu"``, or ``"auto"``.
        dtype: Autocast dtype: ``"bfloat16"``, ``"float16"`` or ``"float32"``.
        notes: Free-text description carried into the checkpoint.
    """

    language: str = LANGUAGE
    pretrained_checkpoint: str = ""
    data_dir: str = f"{LANGUAGE}/reasoning/data"
    tokenizer_model: str = ""
    out_dir: str = f"checkpoints/{LANGUAGE}_reasoning"
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 1e-4
    min_lr: float = 1e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 100
    log_interval: int = 25
    seed: int = 20260914
    device: str = "auto"
    dtype: str = "float16"
    notes: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "FinetuneConfig":
        """Read a finetuning config from YAML.

        Args:
            path: Path to the YAML file.

        Returns:
            The populated config.
        """
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str | Path) -> None:
        """Write this config to YAML next to the checkpoints.

        Args:
            path: Destination file.
        """
        Path(path).write_text(
            yaml.safe_dump(asdict(self), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


class FineTuner:
    """Runs one language's reasoning finetuning, with checkpointing and resume."""

    def __init__(self, cfg: FinetuneConfig, resume: bool = True) -> None:
        """Load the pretrained model and prepare the finetuning run.

        Args:
            cfg: The finetuning configuration.
            resume: Whether to continue from ``latest.pt`` in ``out_dir``.

        Raises:
            FileNotFoundError: If the pretrained checkpoint does not exist.
        """
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.dtype = resolve_dtype(cfg.dtype, self.device)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.out_dir = REPO_ROOT / cfg.out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "finetune_log.jsonl"

        pretrained = Path(cfg.pretrained_checkpoint)
        if not pretrained.is_absolute():
            pretrained = REPO_ROOT / pretrained
        if not pretrained.is_file():
            raise FileNotFoundError(
                f"pretrained checkpoint not found: {pretrained}\n"
                "Download this language's Phase 2 checkpoint first (README has "
                "the Drive link)."
            )
        ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)

        # The architecture is not a choice here -- it is whatever Phase 2
        # trained, read straight out of the checkpoint. Rebuilding it from the
        # YAML would risk a silent mismatch.
        self.model_config = GPTConfig.from_dict(ckpt["config"]["model"])
        self.pretrain_config = ckpt["config"]["train"]
        self.pretrain_step = ckpt.get("step", -1)

        self.model = GPT(self.model_config).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        print(
            f"  loaded pretrained weights from {pretrained.name} "
            f"(step {self.pretrain_step:,})"
        )

        # A fresh optimizer: the pretraining moments describe a different
        # objective on a different corpus and carrying them over would fight the
        # new, much smaller learning rate.
        self.optimizer = self.model.configure_optimizers(
            weight_decay=cfg.weight_decay,
            learning_rate=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            device_type="cuda" if self.device.startswith("cuda") else "cpu",
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.dtype == torch.float16)
        )
        self.autocast = (
            torch.amp.autocast(device_type="cuda", dtype=self.dtype)
            if self.device.startswith("cuda")
            else nullcontext()
        )

        tokenizer_model = cfg.tokenizer_model or None
        if tokenizer_model and not Path(tokenizer_model).is_absolute():
            tokenizer_model = REPO_ROOT / tokenizer_model
        self.tokenizer = load_tokenizer(tokenizer_model)
        data_dir = REPO_ROOT / cfg.data_dir
        self.train_data = ReasoningDataset(
            "train", self.tokenizer, data_dir, device=self.device
        )
        self.val_data = ReasoningDataset(
            "val", self.tokenizer, data_dir, device=self.device
        )
        if self.tokenizer.get_piece_size() != self.model_config.vocab_size:
            raise ValueError(
                f"tokenizer vocabulary ({self.tokenizer.get_piece_size()}) does "
                f"not match the pretrained model ({self.model_config.vocab_size}). "
                "The finetuning tokenizer must be the pretraining one."
            )

        steps_per_epoch = math.ceil(len(self.train_data) / cfg.batch_size)
        self.max_steps = steps_per_epoch * cfg.epochs
        self.schedule = CosineWarmupSchedule(
            self.optimizer,
            peak_lr=cfg.learning_rate,
            min_lr=cfg.min_lr,
            warmup_steps=min(cfg.warmup_steps, max(self.max_steps // 10, 1)),
            max_steps=self.max_steps,
        )
        self.val_batches = self.val_data.batches(
            cfg.batch_size, shuffle=False, seed=cfg.seed
        )

        self.step = 0
        self.examples_seen = 0
        self.best_val_loss = float("inf")
        if resume:
            self._maybe_resume()

        cfg.save(self.out_dir / "finetune_config.yaml")
        print(
            f"  {len(self.train_data):,} train / {len(self.val_data):,} val "
            f"examples | {steps_per_epoch:,} steps/epoch | "
            f"{self.max_steps:,} steps total"
        )

    # -- checkpointing ----------------------------------------------------

    def _checkpoint_payload(self) -> dict:
        """Assemble everything needed to resume this run exactly.

        The ``config.model`` and ``config.train`` keys are carried forward from
        the pretrained checkpoint verbatim. That is deliberate: the Phase 2
        ``Evaluator`` and attention toolkit read those keys to rebuild the model
        and find the packed corpus, so preserving them is what lets a finetuned
        checkpoint be dropped into the existing tooling unchanged.

        Returns:
            The dict that gets serialised to a checkpoint file.
        """
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.schedule.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.step,
            "examples_seen": self.examples_seen,
            "best_val_loss": self.best_val_loss,
            "config": {
                "model": self.model_config.to_dict(),
                "train": self.pretrain_config,
                "finetune": asdict(self.cfg),
                "notes": self.cfg.notes,
            },
            "stage": "reasoning_finetune",
            "pretrained_checkpoint": self.cfg.pretrained_checkpoint,
            "pretrained_step": self.pretrain_step,
            "parameter_count": self.model.parameter_count(),
            "torch_rng_state": torch.get_rng_state(),
        }

    def save_checkpoint(self, tag: str | None = None) -> Path:
        """Write a checkpoint.

        Args:
            tag: Optional filename tag; defaults to the step number.

        Returns:
            Path to the checkpoint that was written.
        """
        name = f"ckpt_{tag}.pt" if tag else f"ckpt_step{self.step:06d}.pt"
        path = self.out_dir / name
        payload = self._checkpoint_payload()
        torch.save(payload, path)
        torch.save(payload, self.out_dir / "latest.pt")
        return path

    def _maybe_resume(self) -> None:
        """Restore state from ``latest.pt`` in the output directory, if present."""
        path = self.out_dir / "latest.pt"
        if not path.exists():
            print("  no finetuning checkpoint yet; starting from the "
                  "pretrained weights")
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.schedule.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.step = ckpt["step"]
        self.examples_seen = ckpt.get("examples_seen", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        print(f"  resumed finetuning from {path} at step {self.step:,}")

    # -- evaluation -------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> float:
        """Average the answer-token loss over the validation split.

        Returns:
            Mean validation cross-entropy in nats over answer tokens only.
        """
        self.model.eval()
        losses = []
        for x, y in self.val_batches:
            with self.autocast:
                _, loss, _ = self.model(x, targets=y)
            losses.append(loss.item())
        self.model.train()
        return float(np.mean(losses)) if losses else float("nan")

    # -- logging ----------------------------------------------------------

    def _log(self, record: dict) -> None:
        """Append one record to the run's JSONL finetuning log.

        Args:
            record: The fields to write.
        """
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    # -- training ---------------------------------------------------------

    def train(self, max_minutes: float | None = None) -> None:
        """Run the finetuning loop to completion.

        Args:
            max_minutes: Optional wall-clock budget; the run stops cleanly and
                saves once it is exceeded.
        """
        self.model.train()
        started = time.time()
        print(f"  finetuning on {self.device} ({self.dtype}) "
              f"for {self.max_steps:,} steps")

        # A resumed run must not replay the batches it already trained on. Each
        # epoch is regenerated from a seed derived from the epoch number, so the
        # batch order is reproducible, and the position within it is simply the
        # step counter -- skipping forward to it lands exactly where the
        # interrupted run stopped.
        consumed = 0
        for epoch in range(self.cfg.epochs):
            batches = self.train_data.batches(
                self.cfg.batch_size, shuffle=True, seed=self.cfg.seed + epoch
            )
            if consumed + len(batches) <= self.step:
                consumed += len(batches)
                continue  # this whole epoch was already trained on
            for x, y in batches:
                if consumed < self.step:
                    consumed += 1
                    continue
                consumed += 1
                if self.step >= self.max_steps:
                    break
                lr = self.schedule.step(self.step)

                with self.autocast:
                    _, loss, _ = self.model(x, targets=y)
                self.scaler.scale(loss).backward()
                if self.cfg.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                self.step += 1
                self.examples_seen += x.size(0)

                if self.step % self.cfg.log_interval == 0:
                    record = {
                        "step": self.step,
                        "epoch": epoch,
                        "train_loss": round(loss.item(), 4),
                        "lr": lr,
                        "examples_seen": self.examples_seen,
                        "elapsed_s": round(time.time() - started, 1),
                    }
                    self._log(record)
                    print(
                        f"    step {self.step:,}/{self.max_steps:,} "
                        f"loss {record['train_loss']:.4f} lr {lr:.2e}"
                    )

                if self.step % self.cfg.eval_interval == 0:
                    val = self.evaluate()
                    self._log({
                        "step": self.step,
                        "epoch": epoch,
                        "val_loss": round(val, 4),
                        "val_perplexity": round(math.exp(min(val, 20)), 3),
                    })
                    print(f"    step {self.step:,} val loss {val:.4f} "
                          f"(ppl {math.exp(min(val, 20)):.2f})")
                    if val < self.best_val_loss:
                        self.best_val_loss = val
                        self.save_checkpoint("best")
                        print(f"      new best; saved ckpt_best.pt")

                if max_minutes and (time.time() - started) / 60 >= max_minutes:
                    print("  wall-clock budget reached; saving and stopping")
                    self.save_checkpoint("final")
                    return

        val = self.evaluate()
        if val < self.best_val_loss:
            self.best_val_loss = val
            self.save_checkpoint("best")
        self._log({
            "step": self.step,
            "val_loss": round(val, 4),
            "final": True,
            "elapsed_s": round(time.time() - started, 1),
        })
        self.save_checkpoint("final")
        print(f"  done: {self.step:,} steps in "
              f"{(time.time() - started) / 60:.1f} min, "
              f"best val loss {self.best_val_loss:.4f}")


def main() -> None:
    """Command-line entry point for reasoning finetuning."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=f"{LANGUAGE}/configs/finetune.yaml")
    parser.add_argument("--pretrained-checkpoint", default=None,
                        help="override the checkpoint named in the config")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    cfg = FinetuneConfig.load(args.config)
    for name in ("pretrained_checkpoint", "out_dir", "device", "dtype", "epochs"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg, name, value)

    tuner = FineTuner(cfg, resume=not args.no_resume)
    tuner.train(max_minutes=args.max_minutes)


if __name__ == "__main__":
    main()

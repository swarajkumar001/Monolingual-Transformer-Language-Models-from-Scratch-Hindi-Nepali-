"""Pretraining loop with resumable checkpointing.

Checkpoint resume is mandatory for this project: Colab sessions terminate
without warning, and a run that cannot pick up where it stopped loses marks. A
checkpoint written here contains model weights, optimizer state, learning-rate
scheduler state, the current step, the token count consumed so far, the RNG
state, and the full run configuration — everything needed to continue a run
bit-for-bit rather than approximately.

Model H and Model L are trained by separate invocations of this module against
separate configs, corpora and output directories. Nothing is shared at runtime,
and neither model is ever initialised from the other.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from ..model.config import RunConfig
from ..data.dataset import BatchSampler
from ..model.transformer import GPT

REPO_ROOT = Path(__file__).resolve().parents[2]


class CosineWarmupSchedule:
    """Linear warmup followed by cosine decay to a floor learning rate.

    Warmup matters because AdamW's second-moment estimate is unreliable for the
    first few hundred steps; stepping at full learning rate before it settles is
    a common cause of early divergence. Cosine decay then spends most of the
    budget at a high rate and anneals smoothly at the end, which tends to beat a
    step schedule at fixed budget.

    Implemented explicitly rather than via ``torch.optim.lr_scheduler`` so that
    its state is a single integer and checkpoint resume is unambiguous.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        peak_lr: float,
        min_lr: float,
        warmup_steps: int,
        max_steps: int,
    ) -> None:
        """Configure the schedule.

        Args:
            optimizer: Optimizer whose param groups will be updated.
            peak_lr: Learning rate reached at the end of warmup.
            min_lr: Floor reached at ``max_steps``.
            warmup_steps: Number of linear warmup steps.
            max_steps: Total steps in the run.
        """
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.last_step = 0

    def lr_at(self, step: int) -> float:
        """Return the learning rate for a given step.

        Args:
            step: Zero-based optimiser step.

        Returns:
            The learning rate to use at that step.
        """
        if step < self.warmup_steps:
            return self.peak_lr * (step + 1) / max(self.warmup_steps, 1)
        progress = (step - self.warmup_steps) / max(
            self.max_steps - self.warmup_steps, 1
        )
        progress = min(progress, 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + coeff * (self.peak_lr - self.min_lr)

    def step(self, step: int) -> float:
        """Set the learning rate on every param group for ``step``.

        Args:
            step: Zero-based optimiser step.

        Returns:
            The learning rate that was applied.
        """
        lr = self.lr_at(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.last_step = step
        return lr

    def state_dict(self) -> dict:
        """Return the schedule's resumable state."""
        return {
            "last_step": self.last_step,
            "peak_lr": self.peak_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore schedule *progress* from a checkpoint, keeping current shape.

        Only ``last_step`` is restored. The schedule's shape (peak rate, floor,
        warmup length, horizon) is deliberately taken from the live config
        instead of the checkpoint, because a resume that extends the run — the
        normal case when a Colab session dies and training is restarted with a
        larger ``max_steps`` — must follow the *new* horizon. Restoring the old
        one would leave ``step > max_steps``, pinning the learning rate at
        ``min_lr`` for the entire extension and making training look stalled.

        Any difference is reported rather than applied silently, since a
        changed schedule mid-run is something the training log should show.

        Args:
            state: A dict previously returned by :meth:`state_dict`.
        """
        self.last_step = state["last_step"]
        changed = {
            key: (state[key], getattr(self, key))
            for key in ("peak_lr", "min_lr", "warmup_steps", "max_steps")
            if key in state and state[key] != getattr(self, key)
        }
        for key, (old, new) in changed.items():
            print(f"  schedule {key}: checkpoint had {old}, using config's {new}")


def resolve_device(requested: str) -> str:
    """Resolve ``"auto"`` to the best available device.

    Args:
        requested: ``"auto"``, ``"cuda"`` or ``"cpu"``.

    Returns:
        A concrete device string.
    """
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    """Pick an autocast dtype the current device actually supports.

    bfloat16 is preferred over float16 where available: it has float32's
    exponent range, so it does not need loss scaling and cannot silently
    overflow to inf during training.

    Args:
        requested: ``"bfloat16"``, ``"float16"`` or ``"float32"``.
        device: Resolved device string.

    Returns:
        The dtype to autocast with.
    """
    if device == "cpu":
        return torch.float32
    if requested == "bfloat16" and not torch.cuda.is_bf16_supported():
        print("  bfloat16 unsupported on this GPU; falling back to float16")
        return torch.float16
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        requested, torch.float32
    )


class Trainer:
    """Runs one language's pretraining, with checkpointing and resume."""

    def __init__(self, run: RunConfig, resume: bool = True) -> None:
        """Set up the model, optimizer, schedule, data and output directory.

        Args:
            run: The full run configuration.
            resume: Whether to continue from the latest checkpoint if one exists.
        """
        self.run = run
        cfg, tcfg = run.model, run.train

        self.device = resolve_device(tcfg.device)
        self.dtype = resolve_dtype(tcfg.dtype, self.device)
        torch.manual_seed(tcfg.seed)
        np.random.seed(tcfg.seed)

        self.out_dir = REPO_ROOT / tcfg.out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "training_log.jsonl"

        self.model = GPT(cfg).to(self.device)
        self.optimizer = self.model.configure_optimizers(
            weight_decay=tcfg.weight_decay,
            learning_rate=tcfg.learning_rate,
            betas=(tcfg.beta1, tcfg.beta2),
            device_type="cuda" if self.device.startswith("cuda") else "cpu",
        )
        self.schedule = CosineWarmupSchedule(
            self.optimizer,
            peak_lr=tcfg.learning_rate,
            min_lr=tcfg.min_lr,
            warmup_steps=tcfg.warmup_steps,
            max_steps=tcfg.max_steps,
        )
        # GradScaler is only needed for float16; bfloat16 has enough exponent
        # range that gradients do not underflow.
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.dtype == torch.float16)
        )
        self.autocast = (
            torch.amp.autocast(device_type="cuda", dtype=self.dtype)
            if self.device.startswith("cuda")
            else nullcontext()
        )

        self.data = BatchSampler(
            REPO_ROOT / tcfg.data_dir,
            block_size=cfg.block_size,
            device=self.device,
            seed=tcfg.seed,
        )
        self.eval_batches = self.data.fixed_batches(
            "val", tcfg.batch_size, tcfg.eval_iters
        ) if "val" in self.data.streams else []

        self.step = 0
        self.tokens_seen = 0
        self.best_val_loss = float("inf")
        if resume:
            self._maybe_resume()

        run.save(self.out_dir / "config.yaml")

    # -- checkpointing ----------------------------------------------------

    def _checkpoint_payload(self) -> dict:
        """Assemble everything needed to resume this run exactly.

        Returns:
            The dict that gets serialised to a checkpoint file.
        """
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.schedule.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "best_val_loss": self.best_val_loss,
            "config": {
                "model": self.run.model.to_dict(),
                "train": self.run.train.to_dict(),
                "notes": self.run.notes,
            },
            "parameter_count": self.model.parameter_count(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": self.data.rng.bit_generator.state,
        }

    def save_checkpoint(self, tag: str | None = None) -> Path:
        """Write a checkpoint and prune old rolling checkpoints.

        Args:
            tag: Optional filename tag; defaults to the step number.

        Returns:
            Path to the checkpoint that was written.
        """
        name = f"ckpt_{tag}.pt" if tag else f"ckpt_step{self.step:06d}.pt"
        path = self.out_dir / name
        torch.save(self._checkpoint_payload(), path)
        # "latest.pt" is what resume looks for, so an interrupted run always has
        # an unambiguous entry point.
        torch.save(self._checkpoint_payload(), self.out_dir / "latest.pt")
        if tag is None:
            rolling = sorted(self.out_dir.glob("ckpt_step*.pt"))
            for stale in rolling[: -self.run.train.keep_last_n]:
                stale.unlink(missing_ok=True)
        return path

    def _maybe_resume(self) -> None:
        """Restore state from ``latest.pt`` in the output directory, if present."""
        path = self.out_dir / "latest.pt"
        if not path.exists():
            print(f"  no checkpoint at {path}; starting from scratch")
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.schedule.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.step = ckpt["step"]
        self.tokens_seen = ckpt["tokens_seen"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        if "numpy_rng_state" in ckpt:
            self.data.rng.bit_generator.state = ckpt["numpy_rng_state"]
        print(
            f"  resumed from {path} at step {self.step:,} "
            f"({self.tokens_seen/1e6:.1f}M tokens seen)"
        )

    # -- evaluation -------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> float:
        """Average the loss over the fixed validation batches.

        Returns:
            Mean validation cross-entropy in nats, or ``nan`` if no validation
            split is available.
        """
        if not self.eval_batches:
            return float("nan")
        self.model.eval()
        losses = []
        for x, y in self.eval_batches:
            with self.autocast:
                _, loss, _ = self.model(x, targets=y)
            losses.append(loss.item())
        self.model.train()
        return float(np.mean(losses))

    # -- logging ----------------------------------------------------------

    def _log(self, record: dict) -> None:
        """Append one record to the run's JSONL training log.

        The log is what the loss curves in the report are plotted from, so it is
        appended to rather than overwritten across resumes.

        Args:
            record: The record to append.
        """
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    # -- the loop ---------------------------------------------------------

    def train(self, max_minutes: float | None = None) -> None:
        """Run the training loop until the step budget or a time limit is hit.

        Args:
            max_minutes: Optional wall-clock budget. When set, the loop exits
                cleanly (writing a checkpoint) before the limit, which is how a
                Colab session is kept from being killed mid-write.
        """
        tcfg = self.run.train
        counts = self.model.parameter_count()
        tokens_per_step = tcfg.tokens_per_step(self.run.model.block_size)

        print(f"\n=== pretraining Model for {tcfg.language} ===")
        print(f"  device={self.device} dtype={self.dtype}")
        print(f"  parameters: {counts['total']:,} total, "
              f"{counts['non_embedding']:,} non-embedding")
        print(f"  data: {json.dumps(self.data.describe())}")
        print(f"  {tokens_per_step:,} tokens/step x {tcfg.max_steps:,} steps = "
              f"{tokens_per_step * tcfg.max_steps / 1e6:.1f}M tokens")
        print(f"  checkpoints -> {self.out_dir}\n")

        self.model.train()
        started = time.time()
        window_started = started
        window_loss = 0.0
        window_steps = 0

        while self.step < tcfg.max_steps:
            lr = self.schedule.step(self.step)

            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(tcfg.grad_accum_steps):
                x, y = self.data.sample("train", tcfg.batch_size)
                with self.autocast:
                    _, loss, _ = self.model(x, targets=y)
                    # Scale so that accumulated gradients average, rather than
                    # sum, across the micro-batches.
                    loss = loss / tcfg.grad_accum_steps
                self.scaler.scale(loss).backward()
                accum_loss += loss.item()

            if tcfg.grad_clip > 0:
                # Unscale before clipping, or the clip threshold would be
                # applied to loss-scaled gradients.
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), tcfg.grad_clip
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.step += 1
            self.tokens_seen += tokens_per_step
            window_loss += accum_loss
            window_steps += 1

            if self.step % tcfg.log_interval == 0:
                now = time.time()
                mean_loss = window_loss / max(window_steps, 1)
                elapsed = now - window_started
                tps = tokens_per_step * window_steps / max(elapsed, 1e-9)
                print(
                    f"  step {self.step:>6,}/{tcfg.max_steps:,}  "
                    f"loss {mean_loss:.4f}  ppl {math.exp(min(mean_loss, 20)):>8.1f}  "
                    f"lr {lr:.2e}  {tps/1e3:.1f}k tok/s  "
                    f"{self.tokens_seen/1e6:.0f}M seen",
                    flush=True,
                )
                self._log({
                    "step": self.step,
                    "train_loss": mean_loss,
                    "lr": lr,
                    "tokens_seen": self.tokens_seen,
                    "tokens_per_sec": tps,
                    "wall_seconds": now - started,
                })
                window_loss, window_steps = 0.0, 0
                window_started = now

            if self.step % tcfg.eval_interval == 0:
                val_loss = self.evaluate()
                print(
                    f"  step {self.step:>6,}  VAL loss {val_loss:.4f}  "
                    f"ppl {math.exp(min(val_loss, 20)):.1f}",
                    flush=True,
                )
                self._log({
                    "step": self.step,
                    "val_loss": val_loss,
                    "val_ppl": math.exp(min(val_loss, 20)),
                    "tokens_seen": self.tokens_seen,
                })
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(tag="best")

            if self.step % tcfg.checkpoint_interval == 0:
                self.save_checkpoint()

            if max_minutes is not None and (time.time() - started) / 60 >= max_minutes:
                print(f"\n  wall-clock budget of {max_minutes} min reached")
                break

        self.save_checkpoint(tag="final")
        print(
            f"\n=== done: step {self.step:,}, "
            f"{self.tokens_seen/1e6:.1f}M tokens, "
            f"best val loss {self.best_val_loss:.4f} "
            f"({time.time() - started:.0f}s this session) ==="
        )


def main() -> None:
    """Parse arguments and run pretraining for one language."""
    parser = argparse.ArgumentParser(description="Pretrain one monolingual LM.")
    parser.add_argument("--config", required=True, help="Path to a run YAML.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing checkpoint and start over.",
    )
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="Stop cleanly after this many minutes (for Colab session limits).",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Override max_steps.")
    parser.add_argument("--device", default=None, help="Override the device.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override data_dir. Needed on Kaggle, where the packed streams are "
        "mounted read-only under /kaggle/input/<dataset>/ rather than in the repo.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Override out_dir. On Kaggle this should point inside "
        "/kaggle/working/, the only writable location that survives the session.",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["bfloat16", "float16", "float32"],
        help="Override the autocast dtype. Kaggle's P100 and T4 GPUs predate "
        "Ampere and have no bfloat16 support; the trainer falls back to float16 "
        "automatically, but this makes the choice explicit in the run config.",
    )
    args = parser.parse_args()

    run = RunConfig.load(args.config)
    if args.max_steps is not None:
        run.train.max_steps = args.max_steps
    if args.device is not None:
        run.train.device = args.device
    if args.data_dir is not None:
        run.train.data_dir = args.data_dir
    if args.out_dir is not None:
        run.train.out_dir = args.out_dir
    if args.dtype is not None:
        run.train.dtype = args.dtype

    Trainer(run, resume=not args.no_resume).train(max_minutes=args.max_minutes)


if __name__ == "__main__":
    main()

"""Language-modelling and generation evaluation for one trained model.

Covers the Phase 2 metric list: cross-entropy and perplexity, bits-per-byte,
BLEU / chrF++ / ROUGE-L against held-out references at several decoding
temperatures, and repetition / Distinct-n diversity diagnostics.

A note on metric choice, since the spec asks for it explicitly:

* **Perplexity** is tokenizer-dependent. Hindi's 8,000-token Unigram vocabulary
  and Nepali's 10,000-token BPE vocabulary segment text at different rates, so
  their perplexities are *not* directly comparable — a model with a coarser
  tokenizer predicts fewer, harder tokens and looks worse than it is.
* **Bits-per-byte** normalises the same total log-likelihood by the UTF-8 byte
  length of the underlying text rather than by token count, which removes the
  tokenizer from the denominator and makes the two models comparable. It is the
  number to trust when contrasting Model H and Model L.
* **BLEU** rewards exact n-gram overlap with a single reference. Open-ended
  continuation has no single correct answer, so absolute BLEU here will be very
  low and is only meaningful as a *relative* signal across decoding settings of
  the same model. It is also tokenization-sensitive; ``intl`` tokenization is
  used rather than the Latin-centric default.
* **chrF++** operates on character n-grams, so it degrades gracefully with the
  rich inflectional morphology of both languages and does not depend on a word
  tokenizer. It is the most trustworthy of the three overlap metrics here.
* **ROUGE-L** is computed with a local longest-common-subsequence implementation
  over whitespace tokens rather than via ``rouge_score``, whose default
  tokenizer lowercases and strips non-ASCII-alphanumeric characters — which
  would delete Devanagari text entirely and silently report zero.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ..model.config import GPTConfig
from ..data.dataset import BatchSampler
from ..model.transformer import GPT

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# Overlap metrics
# --------------------------------------------------------------------------


def lcs_length(a: list[str], b: list[str]) -> int:
    """Return the length of the longest common subsequence of two token lists.

    Args:
        a: First token sequence.
        b: Second token sequence.

    Returns:
        Length of the LCS.
    """
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b):
            curr.append(
                prev[j] + 1 if token_a == token_b else max(curr[j], prev[j + 1])
            )
        prev = curr
    return prev[-1]


def rouge_l(hypotheses: list[str], references: list[str]) -> float:
    """Compute corpus ROUGE-L as the mean per-pair F1 over whitespace tokens.

    Args:
        hypotheses: Generated continuations.
        references: Reference continuations, aligned with ``hypotheses``.

    Returns:
        Mean ROUGE-L F1 in ``[0, 1]``.
    """
    scores = []
    for hyp, ref in zip(hypotheses, references):
        h, r = hyp.split(), ref.split()
        if not h or not r:
            scores.append(0.0)
            continue
        lcs = lcs_length(h, r)
        if lcs == 0:
            scores.append(0.0)
            continue
        precision, recall = lcs / len(h), lcs / len(r)
        scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Return the list of ``n``-grams in a token sequence.

    Args:
        tokens: Token sequence.
        n: N-gram order.

    Returns:
        List of n-gram tuples; empty if the sequence is shorter than ``n``.
    """
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def diversity_metrics(texts: list[str], rep_n: int = 4) -> dict[str, float]:
    """Compute repetition rate and Distinct-1 / Distinct-2 over generated text.

    Degenerate repetition is the characteristic failure of a small language
    model under low-temperature decoding, and these are the cheapest diagnostics
    that surface it.

    Args:
        texts: Generated strings.
        rep_n: N-gram order used for the repetition rate.

    Returns:
        Mapping with ``repetition_rate``, ``distinct_1``, ``distinct_2`` and the
        total generated token count.
    """
    total_tokens = 0
    unigrams: Counter = Counter()
    bigrams: Counter = Counter()
    repeated, total_ngrams = 0, 0
    for text in texts:
        tokens = text.split()
        total_tokens += len(tokens)
        unigrams.update(tokens)
        bigrams.update(ngrams(tokens, 2))
        grams = ngrams(tokens, rep_n)
        total_ngrams += len(grams)
        # Every occurrence beyond the first is a repeat.
        repeated += len(grams) - len(set(grams))
    return {
        "repetition_rate": (repeated / total_ngrams) if total_ngrams else 0.0,
        "repetition_ngram_order": rep_n,
        "distinct_1": (len(unigrams) / total_tokens) if total_tokens else 0.0,
        "distinct_2": (sum(bigrams.values()) and len(bigrams) / sum(bigrams.values()))
        or 0.0,
        "generated_tokens": total_tokens,
    }


def overlap_metrics(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    """Compute BLEU, chrF, chrF++ and ROUGE-L for aligned hypothesis/reference lists.

    Args:
        hypotheses: Generated continuations.
        references: Reference continuations.

    Returns:
        Mapping of metric name to score. BLEU and chrF are on a 0-100 scale as
        sacrebleu reports them; ROUGE-L is on 0-1.
    """
    from sacrebleu.metrics import BLEU, CHRF

    # The metric objects (rather than the one-shot corpus_* helpers) are used so
    # that get_signature() is available: recording the exact tokenizer, smoothing
    # and version alongside the score is what makes a BLEU number reproducible.
    bleu_metric = BLEU(tokenize="intl")
    chrf_metric = CHRF()
    chrf_pp_metric = CHRF(word_order=2)  # chrF++ adds word bigrams to chrF

    bleu = bleu_metric.corpus_score(hypotheses, [references])
    return {
        "bleu": bleu.score,
        "bleu_order": bleu_metric.max_ngram_order,
        "bleu_signature": str(bleu_metric.get_signature()),
        "chrf": chrf_metric.corpus_score(hypotheses, [references]).score,
        "chrf_pp": chrf_pp_metric.corpus_score(hypotheses, [references]).score,
        "chrf_pp_signature": str(chrf_pp_metric.get_signature()),
        "rouge_l": rouge_l(hypotheses, references),
    }


# --------------------------------------------------------------------------
# Evaluator
# --------------------------------------------------------------------------


class Evaluator:
    """Loads a checkpoint and runs the full Phase 2 evaluation suite on it."""

    def __init__(
        self,
        checkpoint: str | Path,
        data_dir: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        """Restore a model from a checkpoint and open its corpus.

        Args:
            checkpoint: Path to a ``.pt`` file written by the trainer.
            data_dir: Override for the packed-data directory; defaults to the
                one recorded in the checkpoint's config.
            device: ``"auto"``, ``"cuda"`` or ``"cpu"``.
        """
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model_config = GPTConfig.from_dict(ckpt["config"]["model"])
        self.train_config = ckpt["config"]["train"]
        self.step = ckpt.get("step", -1)
        self.language = self.train_config.get("language", "unknown")

        self.model = GPT(self.model_config).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        resolved = Path(data_dir or (REPO_ROOT / self.train_config["data_dir"]))
        self.data = BatchSampler(
            resolved, block_size=self.model_config.block_size, device=self.device
        )

        # The tokenizer is needed to turn token ids back into text, both for
        # generation samples and for the byte count that bits-per-byte needs.
        self.tokenizer = None
        meta = next(iter(self.data.streams.values())).meta
        model_rel = meta.get("tokenizer_model")
        if model_rel:
            import sentencepiece as spm

            # The sidecar records a repo-relative path, which is correct when
            # running from a checkout but not when the corpus is mounted
            # elsewhere — on Kaggle the packed streams and their tokenizer sit
            # side by side under /kaggle/input/<dataset>/. Look next to the data
            # first, then fall back to the repository layout.
            candidates = [
                resolved / Path(model_rel).name,
                REPO_ROOT / model_rel,
            ]
            found = next((c for c in candidates if c.exists()), None)
            if found is None:
                raise FileNotFoundError(
                    f"tokenizer {model_rel} not found next to the data "
                    f"({candidates[0]}) or in the repo ({candidates[1]}). "
                    "Perplexity works without it but bits-per-byte and decoded "
                    "samples do not."
                )
            self.tokenizer = spm.SentencePieceProcessor(model_file=str(found))

    def decode(self, ids) -> str:
        """Decode token ids back into text.

        Args:
            ids: An iterable of integer token ids.

        Returns:
            The decoded string, or a space-joined id list if no tokenizer is
            available.
        """
        ids = [int(i) for i in ids]
        if self.tokenizer is None:
            return " ".join(map(str, ids))
        return self.tokenizer.decode(ids)

    @torch.no_grad()
    def perplexity_and_bpb(
        self, split: str = "test", max_tokens: int | None = 2_000_000
    ) -> dict:
        """Score a held-out split for cross-entropy, perplexity and bits-per-byte.

        The split is walked once in non-overlapping windows so every token is
        scored exactly once. Bits-per-byte divides the same total negative log
        likelihood by the UTF-8 byte length of the decoded target text, which is
        what makes the number comparable across the two models' different
        tokenizers.

        Args:
            split: Which split to score.
            max_tokens: Optional cap on tokens scored, for a faster pass.

        Returns:
            Mapping with loss, perplexity, bits-per-byte, and the token and byte
            counts the numbers were computed over.
        """
        total_nll = 0.0
        total_tokens = 0
        total_bytes = 0
        batch_size = 16

        for x, y in self.data.sequential_batches(split, batch_size, max_tokens):
            logits, _, _ = self.model(x)
            # reduction="sum" so batches of unequal size weight correctly.
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum",
            )
            total_nll += nll.item()
            total_tokens += y.numel()
            for row in y.cpu().numpy():
                total_bytes += len(self.decode(row).encode("utf-8"))

        mean_nll = total_nll / max(total_tokens, 1)
        bits = total_nll / math.log(2)
        return {
            "split": split,
            "cross_entropy_nats": mean_nll,
            "perplexity": math.exp(min(mean_nll, 20)),
            "bits_per_token": bits / max(total_tokens, 1),
            "bits_per_byte": bits / max(total_bytes, 1),
            "tokens_scored": total_tokens,
            "bytes_scored": total_bytes,
        }

    @torch.no_grad()
    def generation_suite(
        self,
        split: str = "test",
        n_prompts: int = 64,
        prompt_tokens: int = 64,
        continuation_tokens: int = 64,
        temperatures: tuple[float, ...] = (0.5, 1.0, 1.5),
        seed: int = 7,
    ) -> dict:
        """Generate continuations from held-out prefixes and score them.

        Each held-out window is split into a prompt prefix and a reference
        continuation; the model is asked to continue the prefix, and its output
        is compared against the true continuation under greedy decoding and each
        requested temperature.

        Args:
            split: Split to draw prompts from.
            n_prompts: Number of prompts.
            prompt_tokens: Prefix length in tokens.
            continuation_tokens: Number of tokens to generate and to use as the
                reference length.
            temperatures: Sampling temperatures to evaluate.
            seed: Seed fixing which windows become prompts.

        Returns:
            Mapping from decoding setting name to its metrics, plus a handful of
            decoded samples for the report.
        """
        # Prompt plus continuation must fit inside one context window, since the
        # reference continuation is read out of the same window. Rather than
        # failing on a model with a short context (the smoke configs, or a
        # reduced-context ablation), fall back to splitting the window evenly.
        window = prompt_tokens + continuation_tokens
        if window > self.model_config.block_size:
            prompt_tokens = self.model_config.block_size // 2
            continuation_tokens = self.model_config.block_size - prompt_tokens
            window = self.model_config.block_size
            print(
                f"  note: prompt+continuation exceeded block_size "
                f"({self.model_config.block_size}); using "
                f"{prompt_tokens}+{continuation_tokens} instead"
            )
        batches = self.data.fixed_batches(split, n_prompts, 1, seed=seed)
        x, _ = batches[0]
        prompts = x[:, :prompt_tokens]
        references = [self.decode(row) for row in x[:, prompt_tokens:window].cpu().numpy()]

        eos_id = self.tokenizer.eos_id() if self.tokenizer else None
        settings: dict[str, dict] = {}
        samples: list[dict] = []

        configs = [("greedy", dict(greedy=True))] + [
            (f"temperature_{t}", dict(temperature=t, greedy=False))
            for t in temperatures
        ]
        for name, kwargs in configs:
            torch.manual_seed(seed)
            out = self.model.generate(
                prompts, max_new_tokens=continuation_tokens, eos_id=eos_id, **kwargs
            )
            generated = out[:, prompt_tokens:].cpu().numpy()
            hypotheses = [self.decode(row) for row in generated]
            metrics = overlap_metrics(hypotheses, references)
            metrics.update(diversity_metrics(hypotheses))
            settings[name] = metrics
            for i in range(min(3, len(hypotheses))):
                samples.append(
                    {
                        "setting": name,
                        "prompt": self.decode(prompts[i].cpu().numpy()),
                        "generated": hypotheses[i],
                        "reference": references[i],
                    }
                )

        return {
            "settings": settings,
            "samples": samples,
            "n_prompts": n_prompts,
            "prompt_tokens": prompt_tokens,
            "continuation_tokens": continuation_tokens,
        }

    def attention_report(self, split: str = "test", n_sequences: int = 8) -> dict:
        """Summarise attention entropy and distance on held-out text.

        Args:
            split: Split to draw sequences from.
            n_sequences: Number of sequences to average over.

        Returns:
            The summary produced by :func:`hindi.eval.attention.summarise_model_attention`,
            with the uniform-attention baseline attached for reference.
        """
        from .attention import summarise_model_attention, uniform_baseline_entropy

        x, _ = self.data.fixed_batches(split, n_sequences, 1, seed=11)[0]
        summary = summarise_model_attention(self.model, x)
        summary["uniform_baseline_entropy"] = uniform_baseline_entropy(
            summary["sequence_length"]
        )
        return summary

    def full_report(self, split: str = "test") -> dict:
        """Run every evaluation and return one consolidated result dictionary.

        Args:
            split: Held-out split to evaluate on.

        Returns:
            A dict with model metadata, intrinsic metrics, generation metrics
            and attention statistics.
        """
        return {
            "language": self.language,
            "checkpoint_step": self.step,
            "parameter_count": self.model.parameter_count(),
            "model_config": self.model_config.to_dict(),
            "corpus": self.data.describe(),
            "intrinsic": self.perplexity_and_bpb(split),
            "generation": self.generation_suite(split),
            "attention": self.attention_report(split),
        }


def main() -> None:
    """Evaluate one checkpoint and write the results to JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None, help="Where to write results JSON.")
    args = parser.parse_args()

    ev = Evaluator(args.checkpoint, data_dir=args.data_dir, device=args.device)
    report = ev.full_report(split=args.split)

    out = Path(args.out) if args.out else (
        REPO_ROOT / f"report/phase2/{ev.language}_eval.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    intrinsic = report["intrinsic"]
    print(f"\n=== {ev.language} @ step {ev.step:,} ===")
    print(f"  loss {intrinsic['cross_entropy_nats']:.4f}  "
          f"ppl {intrinsic['perplexity']:.2f}  "
          f"bpb {intrinsic['bits_per_byte']:.4f}")
    for name, metrics in report["generation"]["settings"].items():
        print(f"  {name:>18}: BLEU {metrics['bleu']:.2f}  "
              f"chrF++ {metrics['chrf_pp']:.2f}  "
              f"ROUGE-L {metrics['rouge_l']:.3f}  "
              f"rep {metrics['repetition_rate']:.3f}  "
              f"d1 {metrics['distinct_1']:.3f}")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()

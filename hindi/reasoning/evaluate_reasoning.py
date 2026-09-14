"""Score a checkpoint on Model H's Hindi reasoning test set.

The specification asks for *pretrained vs. finetuned* accuracy, so this script
must work on both, and the comparison has to be fair. That turns out to need
two different measurements, because a pretrained language model has never seen
this prompt format and will not spontaneously answer with a bare name:

**Exact match after greedy decoding.** The model is given the question and
generates freely; the continuation is compared to the gold answer. This is the
honest end-to-end number and the headline metric, but on a pretrained
checkpoint it mostly measures "does it know the output format", which is not
the thing under study, and it bottoms out near zero.

**Forced choice among the entities named in the prompt.** The candidates are
exactly the people the question mentions. For each, the model scores the
continuation, and the highest-scoring candidate is taken as its answer. A model
that cannot format an answer but *has* learned something about comparatives can
still show it here, and chance is a known 1/2 or 1/3, so the pretrained
baseline is interpretable rather than a floor artefact.

Scores are length-normalised (mean log-probability per answer token). Without
that, short names win on total log-probability for no reason connected to the
reasoning, which would silently bias every comparison toward whichever entity
happened to tokenize shortest.

Both numbers are broken down by task family and by hop count, because the
interesting question is not the average -- it is whether two-hop transitive
chains behave differently from one-hop comparisons.

Usage
-----
    python -m hindi.reasoning.evaluate_reasoning \\
        --checkpoint checkpoints/hindi/ckpt_best.pt --tag pretrained
    python -m hindi.reasoning.evaluate_reasoning \\
        --checkpoint checkpoints/hindi_reasoning/ckpt_best.pt --tag finetuned
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from ..model.config import GPTConfig
from ..model.transformer import GPT
from .dataset import ReasoningDataset, load_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]

LANGUAGE = "hindi"

# Devanagari danda and double danda, plus ASCII sentence punctuation. A model
# that answers "राम।" has got the answer right and added terminal punctuation;
# counting that as wrong would understate every model equally but pointlessly.
_TRIM = re.compile(r"^[\s।॥.,!?;:\"'()\[\]]+|[\s।॥.,!?;:\"'()\[\]]+$")


def normalise(text: str) -> str:
    """Strip punctuation and normalise Unicode for answer comparison.

    NFC normalisation matters for Devanagari: the same visible string can be
    encoded with precomposed or decomposed forms, and a byte comparison would
    call two identical-looking names different.

    Args:
        text: Raw generated or gold text.

    Returns:
        The trimmed, NFC-normalised string.
    """
    return _TRIM.sub("", unicodedata.normalize("NFC", text)).strip()


class ReasoningEvaluator:
    """Loads a checkpoint and scores it on the reasoning splits."""

    def __init__(
        self,
        checkpoint: str | Path,
        data_dir: str | Path | None = None,
        tokenizer_model: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        """Restore a model and open the reasoning corpus.

        Args:
            checkpoint: Path to a ``.pt`` file from pretraining or finetuning.
            data_dir: Override for the reasoning data directory.
            tokenizer_model: Override for the tokenizer model path.
            device: ``"auto"``, ``"cuda"`` or ``"cpu"``.
        """
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        path = Path(checkpoint)
        if not path.is_absolute():
            path = REPO_ROOT / path
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.model_config = GPTConfig.from_dict(ckpt["config"]["model"])
        self.model = GPT(self.model_config).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        # "stage" is written only by the finetuner, so its presence is what
        # distinguishes a finetuned checkpoint from a pretrained one.
        self.stage = ckpt.get("stage", "pretrained")
        self.step = ckpt.get("step", -1)
        self.checkpoint_path = str(path)

        tok = Path(tokenizer_model) if tokenizer_model else None
        if tok and not tok.is_absolute():
            tok = REPO_ROOT / tok
        self.tokenizer = load_tokenizer(tok)
        self.data_dir = Path(data_dir) if data_dir else None
        self.eos_id = self.tokenizer.eos_id()

    def load_split(self, split: str) -> ReasoningDataset:
        """Open one split of the reasoning corpus.

        Args:
            split: ``"train"``, ``"val"`` or ``"test"``.

        Returns:
            The tokenized split.
        """
        return ReasoningDataset(
            split, self.tokenizer, self.data_dir, device=self.device
        )

    # -- metric 1: free generation ---------------------------------------

    @torch.no_grad()
    def greedy_answers(
        self, data: ReasoningDataset, batch_size: int = 64
    ) -> list[dict]:
        """Greedily decode an answer for every example.

        Args:
            data: The split to score.
            batch_size: Prompts per generation batch.

        Returns:
            One record per example with the generated text and whether it
            exactly matches the gold answer.
        """
        budget = max(len(r["answer_ids"]) for r in data.rows) + 2
        results = []
        for prompt_ids, rows in data.prompt_batches(batch_size):
            out = self.model.generate(
                prompt_ids, max_new_tokens=budget, greedy=True,
                eos_id=self.eos_id,
            )
            generated = out[:, prompt_ids.size(1):].tolist()
            for row, ids in zip(rows, generated):
                # Everything after the first EOS is the model padding out the
                # batch for other sequences, not part of this answer.
                if self.eos_id in ids:
                    ids = ids[: ids.index(self.eos_id)]
                text = normalise(self.tokenizer.decode(ids))
                entities = [normalise(e) for e in row["entities"]]
                # Which of the entities the question named did the model
                # produce, by position in the prompt? -1 means it produced
                # something that is not in the prompt at all -- typically a
                # training name, which is the signature of having memorised the
                # answer prior instead of reading the question.
                model_index = entities.index(text) if text in entities else -1
                results.append({
                    "task": row["task"], "hops": row["hops"],
                    "attribute": row["attribute"], "prompt": row["prompt"],
                    "gold": row["answer"], "generated": text,
                    "correct": text == normalise(row["answer"]),
                    "entities": row["entities"],
                    "n_candidates": len(entities),
                    "gold_index": entities.index(normalise(row["answer"])),
                    "model_index": model_index,
                })
        return results

    # -- metric 2: forced choice -----------------------------------------

    @torch.no_grad()
    def forced_choice(self, data: ReasoningDataset) -> list[dict]:
        """Pick, among the entities the prompt names, the best-scoring answer.

        Each candidate is appended to the prompt and scored by its mean
        log-probability per answer token. All candidates for one example share a
        prompt, so they are evaluated in a single batched forward pass.

        Args:
            data: The split to score.

        Returns:
            One record per example with the chosen entity and correctness.
        """
        results = []
        for row in data.rows:
            prompt_ids = row["prompt_ids"]
            candidates = row["entities"]
            encoded = [
                self.tokenizer.encode(" " + name) + [self.eos_id]
                for name in candidates
            ]
            width = len(prompt_ids) + max(len(e) for e in encoded)
            batch = torch.full(
                (len(candidates), width), self.eos_id,
                dtype=torch.long, device=self.device,
            )
            for i, answer_ids in enumerate(encoded):
                seq = prompt_ids + answer_ids
                batch[i, : len(seq)] = torch.tensor(
                    seq, dtype=torch.long, device=self.device
                )

            logits, _, _ = self.model(batch[:, :-1])
            log_probs = F.log_softmax(logits.float(), dim=-1)
            targets = batch[:, 1:]
            gathered = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

            scores = []
            for i, answer_ids in enumerate(encoded):
                start = len(prompt_ids) - 1
                end = start + len(answer_ids)
                span = gathered[i, start:end]
                # Length-normalised: see the module docstring. Without this a
                # one-token name beats a three-token one on total log-prob
                # regardless of what the question asked.
                scores.append(float(span.mean()))

            best = int(max(range(len(scores)), key=lambda i: scores[i]))
            chosen = candidates[best]
            results.append({
                "task": row["task"], "hops": row["hops"],
                "attribute": row["attribute"], "prompt": row["prompt"],
                "gold": row["answer"], "chosen": chosen,
                "n_candidates": len(candidates),
                "correct": normalise(chosen) == normalise(row["answer"]),
                "scores": [round(s, 4) for s in scores],
            })
        return results

    # -- metric 3: answer-token loss --------------------------------------

    @torch.no_grad()
    def answer_loss(self, data: ReasoningDataset, batch_size: int = 32) -> dict:
        """Average cross-entropy over gold answer tokens.

        This is the quantity finetuning actually minimises, so it is the
        cleanest before/after number -- unlike accuracy, it moves smoothly and
        is not floored by output-format problems.

        Args:
            data: The split to score.
            batch_size: Examples per batch.

        Returns:
            Mean loss in nats and the corresponding perplexity.
        """
        totals, counts = 0.0, 0
        for x, y in data.batches(batch_size, shuffle=False):
            logits, _, _ = self.model(x)
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                y.reshape(-1), ignore_index=-1, reduction="sum",
            )
            totals += float(loss)
            counts += int((y != -1).sum())
        mean = totals / max(counts, 1)
        return {
            "answer_token_loss": round(mean, 4),
            "answer_token_perplexity": round(math.exp(min(mean, 20)), 3),
            "answer_tokens_scored": counts,
        }


def breakdown(records: list[dict], key: str) -> dict:
    """Accuracy grouped by one field.

    Args:
        records: Scored records carrying ``correct`` and the grouping field.
        key: Field to group by, e.g. ``"task"`` or ``"hops"``.

    Returns:
        Mapping from group value to accuracy, count and number correct.
    """
    groups: dict = defaultdict(lambda: [0, 0])
    for record in records:
        bucket = groups[record[key]]
        bucket[0] += int(record["correct"])
        bucket[1] += 1
    return {
        str(name): {
            "accuracy": round(hits / total, 4),
            "correct": hits,
            "total": total,
        }
        for name, (hits, total) in sorted(groups.items(), key=lambda kv: str(kv[0]))
    }


def position_analysis(records: list[dict]) -> dict:
    """Explain *where* a model's answers come from, not just whether they match.

    Accuracy alone cannot distinguish a model guessing uniformly among the
    entities in the question from one that has latched onto a positional habit
    -- "always answer with the last name mentioned" -- or from one ignoring the
    question and emitting a memorised training name. Those have very different
    implications and this separates them.

    For each task family it reports how often the model produced the entity at
    each prompt position, how often the gold answer sits at each position, and
    how often the model produced a name the question never mentioned.

    Args:
        records: Greedy-generation records carrying ``model_index``,
            ``gold_index`` and ``n_candidates``.

    Returns:
        Per-task distributions over answer position, plus the off-prompt rate.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["task"]].append(record)

    out = {}
    for task, rows in sorted(groups.items()):
        width = max(r["n_candidates"] for r in rows)
        model = [0] * width
        gold = [0] * width
        off_prompt = 0
        for row in rows:
            gold[row["gold_index"]] += 1
            if row["model_index"] < 0:
                off_prompt += 1
            else:
                model[row["model_index"]] += 1
        total = len(rows)
        out[task] = {
            "examples": total,
            "model_answer_position": [round(c / total, 4) for c in model],
            "gold_answer_position": [round(c / total, 4) for c in gold],
            "answered_off_prompt": round(off_prompt / total, 4),
            "candidates": width,
        }
    return out


def qualitative(records: list[dict], limit: int = 8) -> dict:
    """Collect example successes and failures for the report.

    Args:
        records: Scored records.
        limit: How many of each to keep.

    Returns:
        Two lists of examples, the first correct and the second not.
    """
    field = "generated" if "generated" in records[0] else "chosen"
    hits = [r for r in records if r["correct"]][:limit]
    misses = [r for r in records if not r["correct"]][:limit]
    keep = lambda r: {
        "task": r["task"], "prompt": r["prompt"],
        "gold": r["gold"], "model": r[field],
    }
    return {
        "correct": [keep(r) for r in hits],
        "incorrect": [keep(r) for r in misses],
    }


def build_report(
    ev: ReasoningEvaluator,
    data: ReasoningDataset,
    tag: str,
    split: str,
    batch_size: int = 64,
) -> dict:
    """Score a checkpoint on one split and assemble the full JSON report.

    Factored out of :func:`main` so the Kaggle notebook and the command line
    produce byte-identical reports. Two code paths computing "the same" metrics
    is how a side-by-side table stops being comparable.

    Args:
        ev: An evaluator with the checkpoint already loaded.
        data: The tokenized split to score.
        tag: Label for this run, e.g. ``"pretrained"`` or ``"finetuned"``.
        split: Name of the split, recorded in the report.
        batch_size: Batch size for generation and loss.

    Returns:
        The report dict.
    """
    generated = ev.greedy_answers(data, batch_size=batch_size)
    chosen = ev.forced_choice(data)
    losses = ev.answer_loss(data, batch_size=batch_size)

    gen_accuracy = sum(r["correct"] for r in generated) / len(generated)
    fc_accuracy = sum(r["correct"] for r in chosen) / len(chosen)
    # Chance depends on how many entities each question names, which varies by
    # task, so it is computed rather than assumed to be one half.
    chance = sum(1 / r["n_candidates"] for r in chosen) / len(chosen)

    return {
        "language": LANGUAGE,
        "tag": tag,
        "stage": ev.stage,
        "checkpoint": ev.checkpoint_path,
        "checkpoint_step": ev.step,
        "split": split,
        "examples": len(data),
        "sequence_stats": data.sequence_stats(),
        "exact_match": {
            "accuracy": round(gen_accuracy, 4),
            "correct": sum(r["correct"] for r in generated),
            "by_task": breakdown(generated, "task"),
            "by_hops": breakdown(generated, "hops"),
            "by_attribute": breakdown(generated, "attribute"),
        },
        "forced_choice": {
            "accuracy": round(fc_accuracy, 4),
            "correct": sum(r["correct"] for r in chosen),
            "chance_accuracy": round(chance, 4),
            "by_task": breakdown(chosen, "task"),
            "by_hops": breakdown(chosen, "hops"),
            "by_attribute": breakdown(chosen, "attribute"),
        },
        "answer_token_loss": losses,
        "error_analysis": position_analysis(generated),
        "qualitative_generation": qualitative(generated),
        "qualitative_forced_choice": qualitative(chosen),
    }


def main() -> None:
    """Command-line entry point: score one checkpoint and write a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--tag", default=None,
                        help="label for the output file, e.g. pretrained")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--tokenizer-model", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ev = ReasoningEvaluator(
        args.checkpoint, data_dir=args.data_dir,
        tokenizer_model=args.tokenizer_model, device=args.device,
    )
    data = ev.load_split(args.split)
    tag = args.tag or ev.stage
    print(f"  {LANGUAGE} | {tag} | {len(data):,} {args.split} examples "
          f"on {ev.device}")

    report = build_report(ev, data, tag, args.split, args.batch_size)

    out = Path(args.out) if args.out else (
        REPO_ROOT / f"report/phase3/{LANGUAGE}_reasoning_{tag}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    print(f"    exact match   {report['exact_match']['accuracy']:6.2%}")
    print(f"    forced choice {report['forced_choice']['accuracy']:6.2%}  "
          f"(chance {report['forced_choice']['chance_accuracy']:.2%})")
    print(f"    answer loss   "
          f"{report['answer_token_loss']['answer_token_loss']:.4f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()

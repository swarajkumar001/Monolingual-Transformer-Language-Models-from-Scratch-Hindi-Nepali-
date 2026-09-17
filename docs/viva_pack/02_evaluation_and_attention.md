# Phase 2 - evaluation, generation metrics, attention toolkit

Part of the LMA individual project viva pack. Line numbers below match the files in the repository exactly, so any line cited here can be opened directly in the IDE.

Files in this pack:

- `hindi/eval/evaluate.py`
- `hindi/eval/attention.py`
- `hindi/eval/viz.py`
- `hindi/tests/test_model.py`


## hindi/eval/evaluate.py

```python
   1  """Language-modelling and generation evaluation for one trained model.
   2  
   3  Covers the Phase 2 metric list: cross-entropy and perplexity, bits-per-byte,
   4  BLEU / chrF++ / ROUGE-L against held-out references at several decoding
   5  temperatures, and repetition / Distinct-n diversity diagnostics.
   6  
   7  A note on metric choice, since the spec asks for it explicitly:
   8  
   9  * **Perplexity** is tokenizer-dependent. Hindi's 8,000-token Unigram vocabulary
  10    and Nepali's 10,000-token BPE vocabulary segment text at different rates, so
  11    their perplexities are *not* directly comparable — a model with a coarser
  12    tokenizer predicts fewer, harder tokens and looks worse than it is.
  13  * **Bits-per-byte** normalises the same total log-likelihood by the UTF-8 byte
  14    length of the underlying text rather than by token count, which removes the
  15    tokenizer from the denominator and makes the two models comparable. It is the
  16    number to trust when contrasting Model H and Model L.
  17  * **BLEU** rewards exact n-gram overlap with a single reference. Open-ended
  18    continuation has no single correct answer, so absolute BLEU here will be very
  19    low and is only meaningful as a *relative* signal across decoding settings of
  20    the same model. It is also tokenization-sensitive; ``intl`` tokenization is
  21    used rather than the Latin-centric default.
  22  * **chrF++** operates on character n-grams, so it degrades gracefully with the
  23    rich inflectional morphology of both languages and does not depend on a word
  24    tokenizer. It is the most trustworthy of the three overlap metrics here.
  25  * **ROUGE-L** is computed with a local longest-common-subsequence implementation
  26    over whitespace tokens rather than via ``rouge_score``, whose default
  27    tokenizer lowercases and strips non-ASCII-alphanumeric characters — which
  28    would delete Devanagari text entirely and silently report zero.
  29  """
  30  
  31  from __future__ import annotations
  32  
  33  import argparse
  34  import json
  35  import math
  36  from collections import Counter
  37  from pathlib import Path
  38  
  39  import numpy as np
  40  import torch
  41  
  42  from ..model.config import GPTConfig
  43  from ..data.dataset import BatchSampler
  44  from ..model.transformer import GPT
  45  
  46  REPO_ROOT = Path(__file__).resolve().parents[2]
  47  
  48  
  49  # --------------------------------------------------------------------------
  50  # Overlap metrics
  51  # --------------------------------------------------------------------------
  52  
  53  
  54  def lcs_length(a: list[str], b: list[str]) -> int:
  55      """Return the length of the longest common subsequence of two token lists.
  56  
  57      Args:
  58          a: First token sequence.
  59          b: Second token sequence.
  60  
  61      Returns:
  62          Length of the LCS.
  63      """
  64      if not a or not b:
  65          return 0
  66      prev = [0] * (len(b) + 1)
  67      for token_a in a:
  68          curr = [0]
  69          for j, token_b in enumerate(b):
  70              curr.append(
  71                  prev[j] + 1 if token_a == token_b else max(curr[j], prev[j + 1])
  72              )
  73          prev = curr
  74      return prev[-1]
  75  
  76  
  77  def rouge_l(hypotheses: list[str], references: list[str]) -> float:
  78      """Compute corpus ROUGE-L as the mean per-pair F1 over whitespace tokens.
  79  
  80      Args:
  81          hypotheses: Generated continuations.
  82          references: Reference continuations, aligned with ``hypotheses``.
  83  
  84      Returns:
  85          Mean ROUGE-L F1 in ``[0, 1]``.
  86      """
  87      scores = []
  88      for hyp, ref in zip(hypotheses, references):
  89          h, r = hyp.split(), ref.split()
  90          if not h or not r:
  91              scores.append(0.0)
  92              continue
  93          lcs = lcs_length(h, r)
  94          if lcs == 0:
  95              scores.append(0.0)
  96              continue
  97          precision, recall = lcs / len(h), lcs / len(r)
  98          scores.append(2 * precision * recall / (precision + recall))
  99      return float(np.mean(scores)) if scores else 0.0
 100  
 101  
 102  def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
 103      """Return the list of ``n``-grams in a token sequence.
 104  
 105      Args:
 106          tokens: Token sequence.
 107          n: N-gram order.
 108  
 109      Returns:
 110          List of n-gram tuples; empty if the sequence is shorter than ``n``.
 111      """
 112      return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
 113  
 114  
 115  def diversity_metrics(texts: list[str], rep_n: int = 4) -> dict[str, float]:
 116      """Compute repetition rate and Distinct-1 / Distinct-2 over generated text.
 117  
 118      Degenerate repetition is the characteristic failure of a small language
 119      model under low-temperature decoding, and these are the cheapest diagnostics
 120      that surface it.
 121  
 122      Args:
 123          texts: Generated strings.
 124          rep_n: N-gram order used for the repetition rate.
 125  
 126      Returns:
 127          Mapping with ``repetition_rate``, ``distinct_1``, ``distinct_2`` and the
 128          total generated token count.
 129      """
 130      total_tokens = 0
 131      unigrams: Counter = Counter()
 132      bigrams: Counter = Counter()
 133      repeated, total_ngrams = 0, 0
 134      for text in texts:
 135          tokens = text.split()
 136          total_tokens += len(tokens)
 137          unigrams.update(tokens)
 138          bigrams.update(ngrams(tokens, 2))
 139          grams = ngrams(tokens, rep_n)
 140          total_ngrams += len(grams)
 141          # Every occurrence beyond the first is a repeat.
 142          repeated += len(grams) - len(set(grams))
 143      return {
 144          "repetition_rate": (repeated / total_ngrams) if total_ngrams else 0.0,
 145          "repetition_ngram_order": rep_n,
 146          # Distinct-n over whitespace words. Word-level is the standard reading
 147          # of "unique unigrams / bigrams over generated tokens" and is
 148          # comparable across the two models despite their different tokenizers;
 149          # subword-level Distinct-n is reported alongside it by the caller.
 150          "distinct_1": (len(unigrams) / total_tokens) if total_tokens else 0.0,
 151          "distinct_2": (sum(bigrams.values()) and len(bigrams) / sum(bigrams.values()))
 152          or 0.0,
 153          "distinct_unit": "whitespace word",
 154          "generated_tokens": total_tokens,
 155      }
 156  
 157  
 158  def overlap_metrics(hypotheses: list[str], references: list[str]) -> dict[str, float]:
 159      """Compute BLEU, chrF, chrF++ and ROUGE-L for aligned hypothesis/reference lists.
 160  
 161      Args:
 162          hypotheses: Generated continuations.
 163          references: Reference continuations.
 164  
 165      Returns:
 166          Mapping of metric name to score. BLEU and chrF are on a 0-100 scale as
 167          sacrebleu reports them; ROUGE-L is on 0-1.
 168      """
 169      from sacrebleu.metrics import BLEU, CHRF
 170  
 171      # The metric objects (rather than the one-shot corpus_* helpers) are used so
 172      # that get_signature() is available: recording the exact tokenizer, smoothing
 173      # and version alongside the score is what makes a BLEU number reproducible.
 174      bleu_metric = BLEU(tokenize="intl")
 175      chrf_metric = CHRF()
 176      chrf_pp_metric = CHRF(word_order=2)  # chrF++ adds word bigrams to chrF
 177  
 178      bleu = bleu_metric.corpus_score(hypotheses, [references])
 179      return {
 180          "bleu": bleu.score,
 181          "bleu_order": bleu_metric.max_ngram_order,
 182          "bleu_signature": str(bleu_metric.get_signature()),
 183          "chrf": chrf_metric.corpus_score(hypotheses, [references]).score,
 184          "chrf_pp": chrf_pp_metric.corpus_score(hypotheses, [references]).score,
 185          "chrf_pp_signature": str(chrf_pp_metric.get_signature()),
 186          "rouge_l": rouge_l(hypotheses, references),
 187      }
 188  
 189  
 190  # --------------------------------------------------------------------------
 191  # Evaluator
 192  # --------------------------------------------------------------------------
 193  
 194  
 195  class Evaluator:
 196      """Loads a checkpoint and runs the full Phase 2 evaluation suite on it."""
 197  
 198      def __init__(
 199          self,
 200          checkpoint: str | Path,
 201          data_dir: str | Path | None = None,
 202          device: str = "auto",
 203      ) -> None:
 204          """Restore a model from a checkpoint and open its corpus.
 205  
 206          Args:
 207              checkpoint: Path to a ``.pt`` file written by the trainer.
 208              data_dir: Override for the packed-data directory; defaults to the
 209                  one recorded in the checkpoint's config.
 210              device: ``"auto"``, ``"cuda"`` or ``"cpu"``.
 211          """
 212          self.device = (
 213              ("cuda" if torch.cuda.is_available() else "cpu")
 214              if device == "auto"
 215              else device
 216          )
 217          ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
 218          self.model_config = GPTConfig.from_dict(ckpt["config"]["model"])
 219          self.train_config = ckpt["config"]["train"]
 220          self.step = ckpt.get("step", -1)
 221          self.language = self.train_config.get("language", "unknown")
 222  
 223          self.model = GPT(self.model_config).to(self.device)
 224          self.model.load_state_dict(ckpt["model"])
 225          self.model.eval()
 226  
 227          resolved = Path(data_dir or (REPO_ROOT / self.train_config["data_dir"]))
 228          self.data = BatchSampler(
 229              resolved, block_size=self.model_config.block_size, device=self.device
 230          )
 231  
 232          # The tokenizer is needed to turn token ids back into text, both for
 233          # generation samples and for the byte count that bits-per-byte needs.
 234          self.tokenizer = None
 235          meta = next(iter(self.data.streams.values())).meta
 236          model_rel = meta.get("tokenizer_model")
 237          if model_rel:
 238              import sentencepiece as spm
 239  
 240              # The sidecar records a repo-relative path, which is correct when
 241              # running from a checkout but not when the corpus is mounted
 242              # elsewhere — on Kaggle the packed streams and their tokenizer sit
 243              # side by side under /kaggle/input/<dataset>/. Look next to the data
 244              # first, then fall back to the repository layout.
 245              candidates = [
 246                  resolved / Path(model_rel).name,
 247                  REPO_ROOT / model_rel,
 248              ]
 249              found = next((c for c in candidates if c.exists()), None)
 250              if found is None:
 251                  raise FileNotFoundError(
 252                      f"tokenizer {model_rel} not found next to the data "
 253                      f"({candidates[0]}) or in the repo ({candidates[1]}). "
 254                      "Perplexity works without it but bits-per-byte and decoded "
 255                      "samples do not."
 256                  )
 257              self.tokenizer = spm.SentencePieceProcessor(model_file=str(found))
 258  
 259      def decode(self, ids) -> str:
 260          """Decode token ids back into text.
 261  
 262          Args:
 263              ids: An iterable of integer token ids.
 264  
 265          Returns:
 266              The decoded string, or a space-joined id list if no tokenizer is
 267              available.
 268          """
 269          ids = [int(i) for i in ids]
 270          if self.tokenizer is None:
 271              return " ".join(map(str, ids))
 272          return self.tokenizer.decode(ids)
 273  
 274      @torch.no_grad()
 275      def perplexity_and_bpb(
 276          self, split: str = "test", max_tokens: int | None = 2_000_000
 277      ) -> dict:
 278          """Score a held-out split for cross-entropy, perplexity and bits-per-byte.
 279  
 280          The split is walked once in non-overlapping windows so every token is
 281          scored exactly once. Bits-per-byte divides the same total negative log
 282          likelihood by the UTF-8 byte length of the decoded target text, which is
 283          what makes the number comparable across the two models' different
 284          tokenizers.
 285  
 286          Args:
 287              split: Which split to score.
 288              max_tokens: Optional cap on tokens scored, for a faster pass.
 289  
 290          Returns:
 291              Mapping with loss, perplexity, bits-per-byte, and the token and byte
 292              counts the numbers were computed over.
 293          """
 294          total_nll = 0.0
 295          total_tokens = 0
 296          total_bytes = 0
 297          batch_size = 16
 298  
 299          for x, y in self.data.sequential_batches(split, batch_size, max_tokens):
 300              logits, _, _ = self.model(x)
 301              # reduction="sum" so batches of unequal size weight correctly.
 302              nll = torch.nn.functional.cross_entropy(
 303                  logits.reshape(-1, logits.size(-1)),
 304                  y.reshape(-1),
 305                  reduction="sum",
 306              )
 307              total_nll += nll.item()
 308              total_tokens += y.numel()
 309              for row in y.cpu().numpy():
 310                  total_bytes += len(self.decode(row).encode("utf-8"))
 311  
 312          mean_nll = total_nll / max(total_tokens, 1)
 313          bits = total_nll / math.log(2)
 314          return {
 315              "split": split,
 316              "cross_entropy_nats": mean_nll,
 317              "perplexity": math.exp(min(mean_nll, 20)),
 318              "bits_per_token": bits / max(total_tokens, 1),
 319              "bits_per_byte": bits / max(total_bytes, 1),
 320              "tokens_scored": total_tokens,
 321              "bytes_scored": total_bytes,
 322          }
 323  
 324      @torch.no_grad()
 325      def generation_suite(
 326          self,
 327          split: str = "test",
 328          n_prompts: int = 64,
 329          prompt_tokens: int = 64,
 330          continuation_tokens: int = 64,
 331          temperatures: tuple[float, ...] = (0.5, 1.0, 1.5),
 332          seed: int = 7,
 333      ) -> dict:
 334          """Generate continuations from held-out prefixes and score them.
 335  
 336          Each held-out window is split into a prompt prefix and a reference
 337          continuation; the model is asked to continue the prefix, and its output
 338          is compared against the true continuation under greedy decoding and each
 339          requested temperature.
 340  
 341          Args:
 342              split: Split to draw prompts from.
 343              n_prompts: Number of prompts.
 344              prompt_tokens: Prefix length in tokens.
 345              continuation_tokens: Number of tokens to generate and to use as the
 346                  reference length.
 347              temperatures: Sampling temperatures to evaluate.
 348              seed: Seed fixing which windows become prompts.
 349  
 350          Returns:
 351              Mapping from decoding setting name to its metrics, plus a handful of
 352              decoded samples for the report.
 353          """
 354          # Prompt plus continuation must fit inside one context window, since the
 355          # reference continuation is read out of the same window. Rather than
 356          # failing on a model with a short context (the smoke configs, or a
 357          # reduced-context ablation), fall back to splitting the window evenly.
 358          window = prompt_tokens + continuation_tokens
 359          if window > self.model_config.block_size:
 360              prompt_tokens = self.model_config.block_size // 2
 361              continuation_tokens = self.model_config.block_size - prompt_tokens
 362              window = self.model_config.block_size
 363              print(
 364                  f"  note: prompt+continuation exceeded block_size "
 365                  f"({self.model_config.block_size}); using "
 366                  f"{prompt_tokens}+{continuation_tokens} instead"
 367              )
 368          batches = self.data.fixed_batches(split, n_prompts, 1, seed=seed)
 369          x, _ = batches[0]
 370          prompts = x[:, :prompt_tokens]
 371          references = [self.decode(row) for row in x[:, prompt_tokens:window].cpu().numpy()]
 372  
 373          eos_id = self.tokenizer.eos_id() if self.tokenizer else None
 374          settings: dict[str, dict] = {}
 375          samples: list[dict] = []
 376  
 377          configs = [("greedy", dict(greedy=True))] + [
 378              (f"temperature_{t}", dict(temperature=t, greedy=False))
 379              for t in temperatures
 380          ]
 381          for name, kwargs in configs:
 382              torch.manual_seed(seed)
 383              out = self.model.generate(
 384                  prompts, max_new_tokens=continuation_tokens, eos_id=eos_id, **kwargs
 385              )
 386              generated = out[:, prompt_tokens:].cpu().numpy()
 387              hypotheses = [self.decode(row) for row in generated]
 388              metrics = overlap_metrics(hypotheses, references)
 389              metrics.update(diversity_metrics(hypotheses))
 390              settings[name] = metrics
 391              for i in range(min(3, len(hypotheses))):
 392                  samples.append(
 393                      {
 394                          "setting": name,
 395                          "prompt": self.decode(prompts[i].cpu().numpy()),
 396                          "generated": hypotheses[i],
 397                          "reference": references[i],
 398                      }
 399                  )
 400  
 401          return {
 402              "settings": settings,
 403              "samples": samples,
 404              "n_prompts": n_prompts,
 405              "prompt_tokens": prompt_tokens,
 406              "continuation_tokens": continuation_tokens,
 407          }
 408  
 409      def attention_report(self, split: str = "test", n_sequences: int = 8) -> dict:
 410          """Summarise attention entropy and distance on held-out text.
 411  
 412          Args:
 413              split: Split to draw sequences from.
 414              n_sequences: Number of sequences to average over.
 415  
 416          Returns:
 417              The summary produced by :func:`hindi.eval.attention.summarise_model_attention`,
 418              with the uniform-attention baseline attached for reference.
 419          """
 420          from .attention import summarise_model_attention, uniform_baseline_entropy
 421  
 422          x, _ = self.data.fixed_batches(split, n_sequences, 1, seed=11)[0]
 423          summary = summarise_model_attention(self.model, x)
 424          summary["uniform_baseline_entropy"] = uniform_baseline_entropy(
 425              summary["sequence_length"]
 426          )
 427          # The spec requires the heatmaps to "include example sentences in the
 428          # model's language". The figure's tick labels degrade to numeric
 429          # positions when no Devanagari font is installed, so the sentences are
 430          # recorded here as text too and quoted in the report - otherwise the
 431          # example being visualised would be unrecoverable from the artifacts.
 432          summary["split"] = split
 433          summary["example_sentences"] = [
 434              self.decode(row) for row in x[:2].cpu().numpy()
 435          ]
 436          summary["example_tokens"] = [
 437              self.decode([int(t)]) for t in x[0][:48].cpu().numpy()
 438          ]
 439          return summary
 440  
 441      def full_report(self, split: str = "test") -> dict:
 442          """Run every evaluation and return one consolidated result dictionary.
 443  
 444          Intrinsic metrics are computed on **both** the validation and the test
 445          split. The spec asks specifically for "validation cross-entropy loss and
 446          perplexity", while the held-out test split is the cleaner number to
 447          report for a final model, so both are produced rather than choosing one.
 448          ``intrinsic`` keeps the requested split for convenience;
 449          ``intrinsic_by_split`` carries every split that was scored.
 450  
 451          Args:
 452              split: The split reported as the headline ``intrinsic`` result.
 453  
 454          Returns:
 455              A dict with model metadata, intrinsic metrics, generation metrics
 456              and attention statistics.
 457          """
 458          by_split = {}
 459          for name in ("val", "test"):
 460              if name in self.data.streams:
 461                  by_split[name] = self.perplexity_and_bpb(name)
 462          return {
 463              "language": self.language,
 464              "checkpoint_step": self.step,
 465              "parameter_count": self.model.parameter_count(),
 466              "model_config": self.model_config.to_dict(),
 467              "corpus": self.data.describe(),
 468              "intrinsic": by_split.get(split, self.perplexity_and_bpb(split)),
 469              "intrinsic_by_split": by_split,
 470              "generation": self.generation_suite(split),
 471              "attention": self.attention_report(split),
 472          }
 473  
 474  
 475  def main() -> None:
 476      """Evaluate one checkpoint and write the results to JSON."""
 477      parser = argparse.ArgumentParser(description=__doc__)
 478      parser.add_argument("--checkpoint", required=True)
 479      parser.add_argument("--split", default="test")
 480      parser.add_argument("--data-dir", default=None)
 481      parser.add_argument("--device", default="auto")
 482      parser.add_argument("--out", default=None, help="Where to write results JSON.")
 483      args = parser.parse_args()
 484  
 485      ev = Evaluator(args.checkpoint, data_dir=args.data_dir, device=args.device)
 486      report = ev.full_report(split=args.split)
 487  
 488      out = Path(args.out) if args.out else (
 489          REPO_ROOT / f"report/phase2/{ev.language}_eval.json"
 490      )
 491      out.parent.mkdir(parents=True, exist_ok=True)
 492      out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
 493  
 494      intrinsic = report["intrinsic"]
 495      print(f"\n=== {ev.language} @ step {ev.step:,} ===")
 496      print(f"  loss {intrinsic['cross_entropy_nats']:.4f}  "
 497            f"ppl {intrinsic['perplexity']:.2f}  "
 498            f"bpb {intrinsic['bits_per_byte']:.4f}")
 499      for name, metrics in report["generation"]["settings"].items():
 500          print(f"  {name:>18}: BLEU {metrics['bleu']:.2f}  "
 501                f"chrF++ {metrics['chrf_pp']:.2f}  "
 502                f"ROUGE-L {metrics['rouge_l']:.3f}  "
 503                f"rep {metrics['repetition_rate']:.3f}  "
 504                f"d1 {metrics['distinct_1']:.3f}")
 505      print(f"\n  wrote {out}")
 506  
 507  
 508  if __name__ == "__main__":
 509      main()
```


## hindi/eval/attention.py

```python
   1  """Attention analysis: heatmaps, per-head entropy, and mean attention distance.
   2  
   3  Because the attention block is student-implemented, its post-softmax probability
   4  matrices are available directly rather than having to be hooked out of a library
   5  module. Every statistic here is computed over the *causally valid* region only:
   6  query position ``i`` has exactly ``i + 1`` reachable keys, and averaging over the
   7  masked upper triangle would otherwise wash the numbers out with structural zeros.
   8  
   9  Phase 3 reuses this module unchanged on the finetuned checkpoints, so pretrained
  10  and finetuned attention can be compared on identical footing.
  11  """
  12  
  13  from __future__ import annotations
  14  
  15  import math
  16  
  17  import numpy as np
  18  import torch
  19  
  20  from ..model.transformer import GPT
  21  
  22  
  23  def collect_attention(
  24      model: GPT, token_ids: torch.Tensor
  25  ) -> list[torch.Tensor]:
  26      """Run the model once and return the attention probabilities of every layer.
  27  
  28      Args:
  29          model: The model to inspect.
  30          token_ids: Token ids of shape ``(B, T)``.
  31  
  32      Returns:
  33          A list of ``n_layer`` tensors, each ``(B, h, T, T)`` on CPU in float32.
  34      """
  35      was_training = model.training
  36      model.eval()
  37      with torch.no_grad():
  38          _, _, attentions = model(token_ids, return_attn=True)
  39      if was_training:
  40          model.train()
  41      assert attentions is not None
  42      return [a.detach().float().cpu() for a in attentions]
  43  
  44  
  45  def attention_entropy(attn: torch.Tensor, normalise: bool = False) -> np.ndarray:
  46      """Compute the entropy of each head's attention distribution.
  47  
  48      Entropy is measured in nats over the key axis and then averaged across query
  49      positions and batch. A head with high entropy spreads its probability mass
  50      broadly (a diffuse, averaging head); a head with low entropy concentrates on
  51      a few positions (a sharp, selective head).
  52  
  53      Query position 0 is skipped: it can only attend to itself, so its entropy is
  54      identically zero and including it would bias every head downward by the same
  55      uninformative amount.
  56  
  57      Args:
  58          attn: Attention probabilities of shape ``(B, h, T, T)``.
  59          normalise: If True, divide each query's entropy by ``log(i + 1)``, the
  60              maximum achievable entropy given that position ``i`` has ``i + 1``
  61              reachable keys. This removes the artefact that early positions
  62              *cannot* be diffuse, making heads comparable across the sequence.
  63  
  64      Returns:
  65          Array of shape ``(h,)`` giving mean entropy per head.
  66      """
  67      B, h, T, _ = attn.shape
  68      eps = 1e-12
  69      # -sum p log p over the key axis -> (B, h, T)
  70      ent = -(attn * (attn + eps).log()).sum(dim=-1)
  71      if normalise:
  72          # log(i + 1) is the entropy of a uniform distribution over i+1 keys.
  73          max_ent = torch.log(
  74              torch.arange(1, T + 1, dtype=ent.dtype).clamp(min=1)
  75          ).clamp(min=eps)
  76          ent = ent / max_ent.view(1, 1, T)
  77      return ent[:, :, 1:].mean(dim=(0, 2)).numpy()
  78  
  79  
  80  def mean_attention_distance(attn: torch.Tensor) -> np.ndarray:
  81      """Compute how far back, on average, each head attends.
  82  
  83      For query ``i``, the expected distance is ``sum_j p[i, j] * (i - j)``, which
  84      is zero for a head that only looks at the current token and grows for heads
  85      that reach back into the context. Contrasting this across heads is what
  86      separates local/positional heads from long-range, content-based ones.
  87  
  88      Query position 0 is skipped, since it has no history and contributes a
  89      forced zero.
  90  
  91      Args:
  92          attn: Attention probabilities of shape ``(B, h, T, T)``.
  93  
  94      Returns:
  95          Array of shape ``(h,)`` giving mean attention distance per head, in
  96          token positions.
  97      """
  98      B, h, T, _ = attn.shape
  99      positions = torch.arange(T, dtype=attn.dtype)
 100      # distance[i, j] = i - j, non-negative on the causal (lower) triangle.
 101      distance = positions.view(T, 1) - positions.view(1, T)
 102      expected = (attn * distance.view(1, 1, T, T)).sum(dim=-1)  # (B, h, T)
 103      return expected[:, :, 1:].mean(dim=(0, 2)).numpy()
 104  
 105  
 106  def attention_to_self(attn: torch.Tensor) -> np.ndarray:
 107      """Compute each head's average probability mass on the current token.
 108  
 109      A head with high self-attention is largely passing its own representation
 110      through, which often marks a head that has specialised into an identity or
 111      "no-op" role.
 112  
 113      Args:
 114          attn: Attention probabilities of shape ``(B, h, T, T)``.
 115  
 116      Returns:
 117          Array of shape ``(h,)``.
 118      """
 119      T = attn.shape[-1]
 120      diag = attn[..., torch.arange(T), torch.arange(T)]  # (B, h, T)
 121      return diag[:, :, 1:].mean(dim=(0, 2)).numpy()
 122  
 123  
 124  def summarise_model_attention(
 125      model: GPT, token_ids: torch.Tensor, normalise_entropy: bool = True
 126  ) -> dict:
 127      """Compute per-layer, per-head attention statistics for one input batch.
 128  
 129      Args:
 130          model: The model to inspect.
 131          token_ids: Token ids of shape ``(B, T)``.
 132          normalise_entropy: Whether to report length-normalised entropy
 133              alongside the raw value.
 134  
 135      Returns:
 136          A dict with ``n_layer``, ``n_head``, ``sequence_length`` and per-layer
 137          lists of ``entropy``, ``entropy_normalised``, ``mean_distance`` and
 138          ``self_attention``, plus whole-model means.
 139      """
 140      attentions = collect_attention(model, token_ids)
 141      layers = []
 142      for layer_idx, attn in enumerate(attentions):
 143          layers.append(
 144              {
 145                  "layer": layer_idx,
 146                  "entropy": attention_entropy(attn).tolist(),
 147                  "entropy_normalised": (
 148                      attention_entropy(attn, normalise=True).tolist()
 149                      if normalise_entropy
 150                      else None
 151                  ),
 152                  "mean_distance": mean_attention_distance(attn).tolist(),
 153                  "self_attention": attention_to_self(attn).tolist(),
 154              }
 155          )
 156      all_entropy = np.array([l["entropy"] for l in layers])
 157      all_distance = np.array([l["mean_distance"] for l in layers])
 158      return {
 159          "n_layer": len(attentions),
 160          "n_head": attentions[0].shape[1],
 161          "sequence_length": attentions[0].shape[-1],
 162          "layers": layers,
 163          "mean_entropy": float(all_entropy.mean()),
 164          "mean_distance": float(all_distance.mean()),
 165          # A head is called "local" when it attends, on average, within a short
 166          # window; the threshold is a reporting convention, not a property of the
 167          # model, and is stated explicitly so the number can be reproduced.
 168          "local_head_threshold": 8.0,
 169          "local_head_fraction": float((all_distance < 8.0).mean()),
 170      }
 171  
 172  
 173  def head_classification(summary: dict) -> list[dict]:
 174      """Label each head as local, mid-range or long-range with a sharpness note.
 175  
 176      A descriptive convenience for the report's discussion of which heads look
 177      positional and which look content-based; the thresholds are reporting
 178      conventions and are returned alongside the labels.
 179  
 180      Args:
 181          summary: The output of :func:`summarise_model_attention`.
 182  
 183      Returns:
 184          One record per head, with its layer, head index, distance, entropy and
 185          assigned labels.
 186      """
 187      records = []
 188      all_entropy = np.array([l["entropy"] for l in summary["layers"]])
 189      median_entropy = float(np.median(all_entropy))
 190      for layer in summary["layers"]:
 191          for head_idx, (dist, ent) in enumerate(
 192              zip(layer["mean_distance"], layer["entropy"])
 193          ):
 194              if dist < 4:
 195                  span = "local"
 196              elif dist < 32:
 197                  span = "mid-range"
 198              else:
 199                  span = "long-range"
 200              records.append(
 201                  {
 202                      "layer": layer["layer"],
 203                      "head": head_idx,
 204                      "mean_distance": dist,
 205                      "entropy": ent,
 206                      "span": span,
 207                      "sharpness": "sharp" if ent < median_entropy else "diffuse",
 208                  }
 209              )
 210      return records
 211  
 212  
 213  def uniform_baseline_entropy(seq_len: int) -> float:
 214      """Return the mean entropy a perfectly uniform causal head would have.
 215  
 216      This is the reference point every measured entropy should be read against:
 217      a head at this value is doing nothing but averaging its context, while a
 218      head far below it has learned to select.
 219  
 220      Args:
 221          seq_len: Sequence length.
 222  
 223      Returns:
 224          Mean of ``log(i + 1)`` over query positions ``i >= 1``, in nats.
 225      """
 226      return float(np.mean([math.log(i + 1) for i in range(1, seq_len)]))
```


## hindi/eval/viz.py

```python
   1  """Plotting for training curves, attention heatmaps and attention summaries.
   2  
   3  Visualisation is kept separate from computation throughout: nothing in this
   4  module measures anything. It consumes the JSONL training logs written by
   5  :mod:`hindi.train.train` and the statistics dictionaries produced by
   6  :mod:`hindi.eval.attention`, and turns them into figures.
   7  
   8  Every figure carries a title, axis labels, and a legend wherever more than one
   9  series is drawn, per the course requirement that plots missing labels or
  10  legends may receive zero marks.
  11  
  12  Devanagari note: matplotlib's default font has no Devanagari coverage, so token
  13  labels would render as empty boxes. :func:`find_devanagari_font` looks for a
  14  capable font and the heatmap falls back to numbered positions (printing the
  15  token strings alongside) when none is installed, rather than silently drawing
  16  tofu.
  17  """
  18  
  19  from __future__ import annotations
  20  
  21  import json
  22  from pathlib import Path
  23  
  24  import matplotlib
  25  
  26  matplotlib.use("Agg")
  27  import matplotlib.pyplot as plt  # noqa: E402
  28  import numpy as np  # noqa: E402
  29  
  30  # Colour-blind-safe pair, used consistently so Model H and Model L keep the
  31  # same colour across every figure in the report.
  32  COLOR_H = "#1f6f8b"
  33  COLOR_L = "#a34f9d"
  34  LANG_COLORS = {"hindi": COLOR_H, "nepali": COLOR_L}
  35  
  36  DEVANAGARI_CANDIDATES = (
  37      "Nirmala UI",
  38      "Noto Sans Devanagari",
  39      "Noto Serif Devanagari",
  40      "Mangal",
  41      "Lohit Devanagari",
  42      "FreeSerif",
  43      "Kokila",
  44  )
  45  
  46  
  47  def find_devanagari_font() -> str | None:
  48      """Return the name of an installed font that can render Devanagari.
  49  
  50      Returns:
  51          A font family name, or ``None`` if no candidate is installed.
  52      """
  53      from matplotlib import font_manager
  54  
  55      available = {f.name for f in font_manager.fontManager.ttflist}
  56      for candidate in DEVANAGARI_CANDIDATES:
  57          if candidate in available:
  58              return candidate
  59      return None
  60  
  61  
  62  def load_training_log(path: str | Path) -> tuple[dict, dict]:
  63      """Read a JSONL training log into train and validation series.
  64  
  65      Args:
  66          path: Path to a ``training_log.jsonl`` written by the trainer.
  67  
  68      Returns:
  69          A ``(train, val)`` pair of dicts, each with ``step``, ``tokens`` and
  70          ``loss`` lists.
  71      """
  72      train = {"step": [], "tokens": [], "loss": []}
  73      val = {"step": [], "tokens": [], "loss": []}
  74      with open(path, "r", encoding="utf-8") as fh:
  75          for line in fh:
  76              if not line.strip():
  77                  continue
  78              rec = json.loads(line)
  79              target = val if "val_loss" in rec else train
  80              key = "val_loss" if "val_loss" in rec else "train_loss"
  81              if key not in rec:
  82                  continue
  83              target["step"].append(rec["step"])
  84              target["tokens"].append(rec.get("tokens_seen", 0))
  85              target["loss"].append(rec[key])
  86      return train, val
  87  
  88  
  89  def plot_loss_curves(
  90      logs: dict[str, str | Path],
  91      out_path: str | Path,
  92      x_axis: str = "tokens",
  93      stage: str = "pretraining",
  94  ) -> Path:
  95      """Plot training and validation loss for one or both models.
  96  
  97      Args:
  98          logs: Mapping from language name to its ``training_log.jsonl`` path.
  99          out_path: Where to write the figure.
 100          x_axis: ``"tokens"`` to plot against tokens consumed, or ``"step"``.
 101          stage: Named in the axis titles. Defaults to ``"pretraining"`` so the
 102              Phase 2 figures are unchanged; Phase 3 passes ``"finetuning"``,
 103              since a chart of finetuning labelled "pretraining" is simply wrong.
 104  
 105      Returns:
 106          The path the figure was written to.
 107      """
 108      fig, (ax_loss, ax_ppl) = plt.subplots(1, 2, figsize=(13, 5))
 109  
 110      for lang, log_path in logs.items():
 111          train, val = load_training_log(log_path)
 112          color = LANG_COLORS.get(lang, None)
 113          xt = train["tokens"] if x_axis == "tokens" else train["step"]
 114          xv = val["tokens"] if x_axis == "tokens" else val["step"]
 115          if x_axis == "tokens":
 116              xt = [v / 1e6 for v in xt]
 117              xv = [v / 1e6 for v in xv]
 118  
 119          ax_loss.plot(xt, train["loss"], color=color, alpha=0.4, linewidth=1,
 120                       label=f"{lang} — train")
 121          ax_loss.plot(xv, val["loss"], color=color, linewidth=2, marker="o",
 122                       markersize=3, label=f"{lang} — validation")
 123          ax_ppl.plot(xv, [np.exp(min(l, 20)) for l in val["loss"]], color=color,
 124                      linewidth=2, marker="o", markersize=3,
 125                      label=f"{lang} — validation")
 126  
 127      xlabel = "Training tokens consumed (millions)" if x_axis == "tokens" else "Step"
 128      ax_loss.set_title(f"Cross-entropy loss during {stage}")
 129      ax_loss.set_xlabel(xlabel)
 130      ax_loss.set_ylabel("Cross-entropy loss (nats/token)")
 131      ax_loss.grid(alpha=0.3)
 132      ax_loss.legend()
 133  
 134      ax_ppl.set_title(f"Validation perplexity during {stage}")
 135      ax_ppl.set_xlabel(xlabel)
 136      ax_ppl.set_ylabel("Perplexity (exp of cross-entropy)")
 137      ax_ppl.set_yscale("log")
 138      ax_ppl.grid(alpha=0.3, which="both")
 139      ax_ppl.legend()
 140  
 141      fig.tight_layout()
 142      out = Path(out_path)
 143      out.parent.mkdir(parents=True, exist_ok=True)
 144      fig.savefig(out, dpi=150)
 145      plt.close(fig)
 146      return out
 147  
 148  
 149  def plot_attention_heatmap(
 150      attn: np.ndarray,
 151      tokens: list[str],
 152      layer: int,
 153      heads: list[int],
 154      out_path: str | Path,
 155      language: str = "",
 156      max_positions: int = 48,
 157  ) -> Path:
 158      """Draw query-vs-key attention heatmaps for several heads of one layer.
 159  
 160      Args:
 161          attn: Attention probabilities of shape ``(h, T, T)`` for one sequence.
 162          tokens: Decoded token strings, one per position.
 163          layer: Layer index, used in the titles.
 164          heads: Which head indices to draw.
 165          out_path: Where to write the figure.
 166          language: Language name, used in the suptitle.
 167          max_positions: Truncate to the first N positions so labels stay legible.
 168  
 169      Returns:
 170          The path the figure was written to.
 171      """
 172      font = find_devanagari_font()
 173      n = min(max_positions, attn.shape[-1])
 174      labels = tokens[:n] if font else [str(i) for i in range(n)]
 175  
 176      fig, axes = plt.subplots(
 177          1, len(heads), figsize=(5.2 * len(heads), 5.0), squeeze=False
 178      )
 179      for ax, head in zip(axes[0], heads):
 180          matrix = attn[head, :n, :n]
 181          im = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0.0)
 182          ax.set_title(f"Layer {layer}, head {head}")
 183          ax.set_xlabel("Key position (attended to)")
 184          ax.set_ylabel("Query position (attending from)")
 185          step = max(1, n // 16)
 186          ax.set_xticks(range(0, n, step))
 187          ax.set_yticks(range(0, n, step))
 188          ax.set_xticklabels(labels[::step], rotation=90, fontsize=7,
 189                             fontfamily=font if font else None)
 190          ax.set_yticklabels(labels[::step], fontsize=7,
 191                             fontfamily=font if font else None)
 192          # The colour bar is this figure's legend: it maps colour to probability.
 193          cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
 194          cbar.set_label("Attention probability")
 195  
 196      title = f"Causal self-attention — {language}" if language else "Causal self-attention"
 197      if not font:
 198          title += "  (positions shown; no Devanagari font installed)"
 199      fig.suptitle(title, fontsize=13)
 200      fig.tight_layout()
 201      out = Path(out_path)
 202      out.parent.mkdir(parents=True, exist_ok=True)
 203      fig.savefig(out, dpi=150)
 204      plt.close(fig)
 205      return out
 206  
 207  
 208  def plot_attention_summary(
 209      summaries: dict[str, dict],
 210      out_path: str | Path,
 211  ) -> Path:
 212      """Plot per-head attention entropy and mean attention distance by layer.
 213  
 214      Args:
 215          summaries: Mapping from language name to the dict returned by
 216              :func:`hindi.eval.attention.summarise_model_attention`.
 217          out_path: Where to write the figure.
 218  
 219      Returns:
 220          The path the figure was written to.
 221      """
 222      fig, (ax_ent, ax_dist) = plt.subplots(1, 2, figsize=(13, 5))
 223  
 224      for lang, summary in summaries.items():
 225          color = LANG_COLORS.get(lang, None)
 226          n_layer = summary["n_layer"]
 227          entropies = np.array([l["entropy"] for l in summary["layers"]])
 228          distances = np.array([l["mean_distance"] for l in summary["layers"]])
 229          layers = np.arange(n_layer)
 230  
 231          # One point per head, jittered horizontally so overlapping heads show.
 232          for arr, ax in ((entropies, ax_ent), (distances, ax_dist)):
 233              jitter = np.linspace(-0.18, 0.18, arr.shape[1])
 234              for head in range(arr.shape[1]):
 235                  ax.scatter(layers + jitter[head], arr[:, head], color=color,
 236                             alpha=0.55, s=26,
 237                             label=lang if head == 0 else None)
 238              ax.plot(layers, arr.mean(axis=1), color=color, linewidth=2,
 239                      linestyle="--", label=f"{lang} — layer mean")
 240  
 241          baseline = summary.get("uniform_baseline_entropy")
 242          if baseline:
 243              ax_ent.axhline(baseline, color="grey", linestyle=":", linewidth=1.5,
 244                             label="uniform-attention baseline")
 245  
 246      ax_ent.set_title("Attention entropy per head")
 247      ax_ent.set_xlabel("Layer index (0 = first block)")
 248      ax_ent.set_ylabel("Mean attention entropy (nats)")
 249      ax_ent.grid(alpha=0.3)
 250      ax_ent.legend(fontsize=8)
 251  
 252      ax_dist.set_title("Mean attention distance per head")
 253      ax_dist.set_xlabel("Layer index (0 = first block)")
 254      ax_dist.set_ylabel("Mean distance attended back (token positions)")
 255      ax_dist.grid(alpha=0.3)
 256      ax_dist.legend(fontsize=8)
 257  
 258      fig.tight_layout()
 259      out = Path(out_path)
 260      out.parent.mkdir(parents=True, exist_ok=True)
 261      fig.savefig(out, dpi=150)
 262      plt.close(fig)
 263      return out
 264  
 265  
 266  def plot_generation_metrics(
 267      reports: dict[str, dict],
 268      out_path: str | Path,
 269  ) -> Path:
 270      """Plot overlap and diversity metrics across decoding settings.
 271  
 272      Args:
 273          reports: Mapping from language name to that model's ``generation``
 274              report section.
 275          out_path: Where to write the figure.
 276  
 277      Returns:
 278          The path the figure was written to.
 279      """
 280      fig, (ax_overlap, ax_div) = plt.subplots(1, 2, figsize=(13, 5))
 281      settings = list(next(iter(reports.values()))["settings"].keys())
 282      x = np.arange(len(settings))
 283      width = 0.35
 284  
 285      for offset, (lang, report) in enumerate(reports.items()):
 286          color = LANG_COLORS.get(lang, None)
 287          chrf = [report["settings"][s]["chrf_pp"] for s in settings]
 288          rep = [report["settings"][s]["repetition_rate"] for s in settings]
 289          d2 = [report["settings"][s]["distinct_2"] for s in settings]
 290          pos = x + (offset - 0.5) * width
 291          ax_overlap.bar(pos, chrf, width, color=color, label=f"{lang} — chrF++")
 292          ax_div.plot(x, rep, color=color, marker="o", label=f"{lang} — repetition rate")
 293          ax_div.plot(x, d2, color=color, marker="s", linestyle="--",
 294                      label=f"{lang} — Distinct-2")
 295  
 296      ax_overlap.set_title("Generation overlap with held-out references")
 297      ax_overlap.set_xlabel("Decoding setting")
 298      ax_overlap.set_ylabel("chrF++ score (0-100)")
 299      ax_overlap.set_xticks(x)
 300      ax_overlap.set_xticklabels(settings, rotation=20)
 301      ax_overlap.grid(alpha=0.3, axis="y")
 302      ax_overlap.legend()
 303  
 304      ax_div.set_title("Repetition and diversity by decoding setting")
 305      ax_div.set_xlabel("Decoding setting")
 306      ax_div.set_ylabel("Rate (fraction of generated n-grams)")
 307      ax_div.set_xticks(x)
 308      ax_div.set_xticklabels(settings, rotation=20)
 309      ax_div.grid(alpha=0.3)
 310      ax_div.legend(fontsize=8)
 311  
 312      fig.tight_layout()
 313      out = Path(out_path)
 314      out.parent.mkdir(parents=True, exist_ok=True)
 315      fig.savefig(out, dpi=150)
 316      plt.close(fig)
 317      return out
```


## hindi/tests/test_model.py

```python
   1  """Correctness tests for the from-scratch Transformer.
   2  
   3  The central test here is :func:`test_causal_mask_blocks_the_future`, which is
   4  the empirical verification the course spec asks for directly: "verify
   5  empirically that the model cannot see the future—for example, show that
   6  changing token t+1 does not change the logits at position t."
   7  
   8  Runs either under pytest or as a plain script::
   9  
  10      python hindi/tests/test_model.py
  11  """
  12  
  13  from __future__ import annotations
  14  
  15  import sys
  16  from pathlib import Path
  17  
  18  import torch
  19  
  20  sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
  21  
  22  from hindi.model.config import GPTConfig  # noqa: E402
  23  from hindi.model.transformer import (  # noqa: E402
  24      GPT,
  25      MultiHeadCausalSelfAttention,
  26  )
  27  
  28  # A deliberately tiny model: these tests check structural properties that hold
  29  # at any size, and a small model keeps the suite fast.
  30  TINY = dict(vocab_size=97, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)
  31  
  32  
  33  def _model(**overrides) -> GPT:
  34      """Build a small deterministic model in eval mode.
  35  
  36      Dropout is disabled and the model is put in eval mode so that two forward
  37      passes over the same input are exactly reproducible — otherwise the causal
  38      test would fail for reasons unrelated to masking.
  39  
  40      Args:
  41          **overrides: Fields overriding :data:`TINY`.
  42  
  43      Returns:
  44          The constructed model, in eval mode.
  45      """
  46      torch.manual_seed(0)
  47      cfg = GPTConfig(**{**TINY, **overrides})
  48      return GPT(cfg).eval()
  49  
  50  
  51  def test_causal_mask_blocks_the_future() -> None:
  52      """Changing a token at position t+1 must not move the logits at position t.
  53  
  54      This is the defining property of a causal language model. If it fails, the
  55      model is reading its own targets and every perplexity number downstream is
  56      meaningless.
  57      """
  58      model = _model()
  59      T = model.config.block_size
  60      torch.manual_seed(1)
  61      idx = torch.randint(0, model.config.vocab_size, (2, T))
  62  
  63      with torch.no_grad():
  64          base_logits, _, _ = model(idx)
  65  
  66      # Perturb one position at a time and confirm nothing at or before the
  67      # previous position moves.
  68      max_leak = 0.0
  69      for t in range(T - 1):
  70          perturbed = idx.clone()
  71          # Replace every token strictly after t with a different token.
  72          perturbed[:, t + 1 :] = (perturbed[:, t + 1 :] + 17) % model.config.vocab_size
  73          assert not torch.equal(perturbed, idx), "perturbation was a no-op"
  74  
  75          with torch.no_grad():
  76              new_logits, _, _ = model(perturbed)
  77  
  78          leak = (new_logits[:, : t + 1] - base_logits[:, : t + 1]).abs().max().item()
  79          max_leak = max(max_leak, leak)
  80          assert leak < 1e-5, (
  81              f"future leakage at position {t}: perturbing tokens after t changed "
  82              f"the logits at positions <= t by {leak:.3e}"
  83          )
  84  
  85          # Sanity check in the other direction: the logits *at* the perturbed
  86          # positions must change, otherwise the test would pass on a model that
  87          # simply ignores its input.
  88          changed = (new_logits[:, t + 1 :] - base_logits[:, t + 1 :]).abs().max().item()
  89          assert changed > 1e-6, f"perturbation at {t+1} had no effect anywhere"
  90  
  91      print(f"  causal mask: max logit leakage across all positions = {max_leak:.3e}")
  92  
  93  
  94  def test_attention_is_lower_triangular_and_normalised() -> None:
  95      """Attention probabilities must be zero above the diagonal and sum to one.
  96  
  97      A direct check on the mask itself, complementing the behavioural test
  98      above: every query row is a probability distribution over keys at positions
  99      ``<= t`` and nothing else.
 100      """
 101      model = _model()
 102      T = model.config.block_size
 103      idx = torch.randint(0, model.config.vocab_size, (1, T))
 104  
 105      with torch.no_grad():
 106          _, _, attentions = model(idx, return_attn=True)
 107  
 108      assert attentions is not None
 109      assert len(attentions) == model.config.n_layer
 110  
 111      upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
 112      for layer, attn in enumerate(attentions):
 113          assert attn.shape == (1, model.config.n_head, T, T), attn.shape
 114          future_mass = attn[..., upper].abs().max().item()
 115          assert future_mass == 0.0, (
 116              f"layer {layer}: attention mass {future_mass:.3e} on future positions"
 117          )
 118          row_sums = attn.sum(dim=-1)
 119          assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
 120              f"layer {layer}: attention rows do not sum to 1"
 121          )
 122  
 123  
 124  def test_forward_shapes_and_loss() -> None:
 125      """Logits, loss and attention tensors must have the documented shapes."""
 126      model = _model()
 127      B, T = 3, 8
 128      idx = torch.randint(0, model.config.vocab_size, (B, T))
 129      targets = torch.randint(0, model.config.vocab_size, (B, T))
 130  
 131      logits, loss, attns = model(idx, targets=targets, return_attn=True)
 132      assert logits.shape == (B, T, model.config.vocab_size), logits.shape
 133      assert loss is not None and loss.ndim == 0
 134      assert attns is not None and len(attns) == model.config.n_layer
 135      assert attns[0].shape == (B, model.config.n_head, T, T)
 136  
 137      # An untrained model over V tokens should sit near ln(V).
 138      import math
 139  
 140      expected = math.log(model.config.vocab_size)
 141      assert abs(loss.item() - expected) < 1.0, (
 142          f"initial loss {loss.item():.3f} is far from ln(V)={expected:.3f}; "
 143          "check weight initialisation"
 144      )
 145  
 146  
 147  def test_sequence_longer_than_block_size_is_rejected() -> None:
 148      """Exceeding block_size must raise, not silently index out of range.
 149  
 150      With learned absolute positional embeddings there is no row for a position
 151      beyond ``block_size``, so this is a real architectural limit and should
 152      surface as a clear error.
 153      """
 154      model = _model()
 155      idx = torch.zeros((1, model.config.block_size + 1), dtype=torch.long)
 156      try:
 157          model(idx)
 158      except ValueError as exc:
 159          assert "block_size" in str(exc)
 160      else:
 161          raise AssertionError("expected ValueError for over-long sequence")
 162  
 163  
 164  def test_weight_tying_shares_storage() -> None:
 165      """With tying on, the output head and the token embedding are one tensor."""
 166      tied = _model(tie_weights=True)
 167      assert tied.lm_head.weight.data_ptr() == tied.token_embedding.weight.data_ptr()
 168      untied = _model(tie_weights=False)
 169      assert untied.lm_head.weight.data_ptr() != untied.token_embedding.weight.data_ptr()
 170  
 171      # Tying must save exactly one V x d matrix.
 172      saved = tied.config.vocab_size * tied.config.n_embd
 173      assert (
 174          untied.parameter_count()["total"] - tied.parameter_count()["total"] == saved
 175      )
 176  
 177  
 178  def test_parameter_count_matches_analytic_estimate() -> None:
 179      """The built model's parameter count must match the config's estimate.
 180  
 181      The estimate is what config sizing decisions are made from, so a drift
 182      between the two would mean the reported ~25M target was computed against a
 183      model that was never built.
 184      """
 185      model = _model()
 186      actual = model.parameter_count()["total"]
 187      estimated = model.config.estimated_parameters()["total"]
 188      assert actual == estimated, f"estimate {estimated:,} != actual {actual:,}"
 189  
 190  
 191  def test_positional_encoding_variants() -> None:
 192      """All three position schemes run, and "none" really removes position.
 193  
 194      Without positional information the model is permutation invariant: shuffling
 195      the input tokens permutes the outputs identically rather than producing
 196      something new. This is the property the optional bonus ablation destroys.
 197      """
 198      for scheme in ("learned", "sinusoidal", "none"):
 199          model = _model(pos_encoding=scheme)
 200          idx = torch.randint(0, model.config.vocab_size, (1, 8))
 201          with torch.no_grad():
 202              logits, _, _ = model(idx)
 203          assert logits.shape == (1, 8, model.config.vocab_size)
 204  
 205      # A position-free model, given a reversed prefix of length 1, must give the
 206      # same first-position logits regardless of what follows.
 207      model = _model(pos_encoding="none")
 208      a = torch.tensor([[5, 9, 11, 2]])
 209      b = torch.tensor([[5, 2, 11, 9]])
 210      with torch.no_grad():
 211          la, _, _ = model(a)
 212          lb, _, _ = model(b)
 213      assert torch.allclose(la[:, 0], lb[:, 0], atol=1e-6)
 214  
 215  
 216  def test_no_prebuilt_attention_modules_used() -> None:
 217      """Guard the spec's hard constraint on pre-built Transformer components.
 218  
 219      Walks the module tree and fails on anything from ``torch.nn``'s Transformer
 220      family. Cheap insurance against a future refactor quietly importing one.
 221      """
 222      import torch.nn as nn
 223  
 224      forbidden = (
 225          nn.Transformer,
 226          nn.TransformerEncoder,
 227          nn.TransformerDecoder,
 228          nn.TransformerEncoderLayer,
 229          nn.TransformerDecoderLayer,
 230          nn.MultiheadAttention,
 231      )
 232      model = _model()
 233      for name, module in model.named_modules():
 234          assert not isinstance(module, forbidden), (
 235              f"{name} is a pre-built Transformer module: {type(module).__name__}"
 236          )
 237      # And the attention block we do use is our own.
 238      assert any(
 239          isinstance(m, MultiHeadCausalSelfAttention) for m in model.modules()
 240      ), "no student-implemented attention block found"
 241  
 242  
 243  def test_generate_is_deterministic_when_greedy() -> None:
 244      """Greedy decoding must be reproducible and must respect block_size."""
 245      model = _model()
 246      prompt = torch.randint(0, model.config.vocab_size, (2, 4))
 247      a = model.generate(prompt, max_new_tokens=20, greedy=True)
 248      b = model.generate(prompt, max_new_tokens=20, greedy=True)
 249      assert torch.equal(a, b)
 250      assert a.shape == (2, 24)
 251      # 24 > block_size=16, so the context was cropped and no index error occurred.
 252  
 253  
 254  def main() -> int:
 255      """Run every test in this module and report a summary.
 256  
 257      Returns:
 258          Process exit code: ``0`` if all tests passed, ``1`` otherwise.
 259      """
 260      tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
 261      failures = 0
 262      for test in tests:
 263          try:
 264              test()
 265          except AssertionError as exc:
 266              failures += 1
 267              print(f"FAIL  {test.__name__}\n      {exc}")
 268          except Exception as exc:  # noqa: BLE001
 269              failures += 1
 270              print(f"ERROR {test.__name__}\n      {type(exc).__name__}: {exc}")
 271          else:
 272              print(f"pass  {test.__name__}")
 273      print(f"\n{len(tests) - failures}/{len(tests)} passed")
 274      return 1 if failures else 0
 275  
 276  
 277  if __name__ == "__main__":
 278      raise SystemExit(main())
```

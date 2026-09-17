# Phase 3 - reasoning data, finetuning, evaluation

Part of the LMA individual project viva pack. Line numbers below match the files in the repository exactly, so any line cited here can be opened directly in the IDE.

Files in this pack:

- `hindi/reasoning/generate.py`
- `hindi/reasoning/dataset.py`
- `hindi/reasoning/finetune.py`
- `hindi/reasoning/evaluate_reasoning.py`
- `hindi/reasoning/attention_compare.py`
- `hindi/configs/finetune.yaml`


## hindi/reasoning/generate.py

```python
   1  """Generate Model H's synthetic comparative-reasoning dataset, in Hindi.
   2  
   3  The assignment requires a finetuning set that is built here rather than
   4  downloaded, so that every ground-truth label is known by construction. It also
   5  requires the text to be in the target language's script *and* natural phrasing
   6  -- not English sentence patterns with Devanagari names dropped in. Every
   7  template below is a sentence a Hindi speaker would actually write.
   8  
   9  This file belongs to **Model H** (Hindi). Model L has its own independent
  10  generator at ``nepali/reasoning/generate.py``. The two datasets share no
  11  sentences, no entity names and no templates, and neither model is ever shown the
  12  other's data.
  13  
  14  Four task families, covering the three styles the specification names:
  15  
  16  ``compare2``    One comparison, then a question about the pair.
  17  ``transitive3`` Two chained comparisons; the answer needs the chain.
  18  ``numeric2``    Two measured quantities; compare them.
  19  ``numeric3``    Three measured quantities; pick the extreme.
  20  
  21  Leakage control
  22  ---------------
  23  Both defences the specification suggests are applied at once:
  24  
  25  * **Held-out entity names.** The name pool is split three ways with no overlap,
  26    so a test question is always about people the model never saw in training.
  27  * **Held-out templates.** The sentence patterns are also split three ways, so a
  28    test question is phrased in a way the model never saw either.
  29  
  30  A model can therefore only score well by actually performing the comparison --
  31  memorising "this name goes with this answer", or memorising a phrasing, both
  32  fail.
  33  
  34  Usage
  35  -----
  36      python hindi/reasoning/generate.py
  37      python hindi/reasoning/generate.py --train 20000 --val 2000 --test 2000
  38  """
  39  
  40  from __future__ import annotations
  41  
  42  import argparse
  43  import json
  44  import random
  45  from collections import Counter
  46  from pathlib import Path
  47  
  48  REPO_ROOT = Path(__file__).resolve().parents[2]
  49  OUT_DIR = Path(__file__).resolve().parent / "data"
  50  
  51  LANGUAGE = "hindi"
  52  
  53  # ---------------------------------------------------------------------------
  54  # Entity names, split so that no name appears in more than one split.
  55  # ---------------------------------------------------------------------------
  56  # Each entry is (name, gender). Gender is carried because Hindi adjectives
  57  # inflect for it -- "लंबा" for a man, "लंबी" for a woman. Getting that wrong
  58  # would produce ungrammatical text, which the specification explicitly rules
  59  # out, so the generator tracks it rather than defaulting everything to
  60  # masculine.
  61  
  62  NAMES_TRAIN = [
  63      ("राम", "m"), ("श्याम", "m"), ("मोहन", "m"), ("सोहन", "m"), ("विकास", "m"),
  64      ("राहुल", "m"), ("अजय", "m"), ("विजय", "m"), ("अनिल", "m"), ("सुनील", "m"),
  65      ("दीपक", "m"), ("मनोज", "m"), ("संजय", "m"), ("राजेश", "m"), ("महेश", "m"),
  66      ("सुरेश", "m"), ("नरेश", "m"), ("प्रकाश", "m"), ("अशोक", "m"), ("विनोद", "m"),
  67      ("सीता", "f"), ("गीता", "f"), ("रीता", "f"), ("मीना", "f"), ("सुनीता", "f"),
  68      ("अनीता", "f"), ("कविता", "f"), ("ममता", "f"), ("रेखा", "f"), ("शोभा", "f"),
  69      ("पूजा", "f"), ("नेहा", "f"), ("प्रीति", "f"), ("संगीता", "f"), ("वंदना", "f"),
  70      ("आशा", "f"), ("उषा", "f"), ("लता", "f"), ("माया", "f"), ("राधा", "f"),
  71  ]
  72  
  73  NAMES_VAL = [
  74      ("गोपाल", "m"), ("किशोर", "m"), ("भरत", "m"), ("नितिन", "m"), ("हरीश", "m"),
  75      ("यमुना", "f"), ("कमला", "f"), ("सरिता", "f"), ("निशा", "f"), ("भारती", "f"),
  76  ]
  77  
  78  NAMES_TEST = [
  79      ("अरुण", "m"), ("वरुण", "m"), ("तरुण", "m"), ("नवीन", "m"), ("प्रवीण", "m"),
  80      ("शारदा", "f"), ("मालती", "f"), ("चंदा", "f"), ("बबीता", "f"), ("सविता", "f"),
  81  ]
  82  
  83  # ---------------------------------------------------------------------------
  84  # Attributes. Each carries the comparative adjective in both genders, plus the
  85  # phrasing used when the attribute is stated as a number.
  86  # ---------------------------------------------------------------------------
  87  # "more"/"less" are the two directions of the same attribute: taller/shorter,
  88  # older/younger. Adjectives that do not inflect (भारी, तेज़, अमीर) simply repeat
  89  # the same form for both genders.
  90  
  91  ATTRIBUTES = {
  92      "height": {
  93          "more": ("लंबा", "लंबी"),
  94          "less": ("छोटा", "छोटी"),
  95          "noun": "लंबाई",
  96          "possessive": "की",
  97          "unit": "सेंटीमीटर",
  98          "range": (140, 190),
  99      },
 100      "age": {
 101          "more": ("बड़ा", "बड़ी"),
 102          "less": ("छोटा", "छोटी"),
 103          "noun": "उम्र",
 104          "possessive": "की",
 105          "unit": "साल",
 106          "range": (12, 70),
 107      },
 108      "weight": {
 109          "more": ("भारी", "भारी"),
 110          "less": ("हल्का", "हल्की"),
 111          "noun": "वज़न",
 112          "possessive": "का",
 113          "unit": "किलो",
 114          "range": (35, 95),
 115      },
 116      "speed": {
 117          "more": ("तेज़", "तेज़"),
 118          "less": ("धीमा", "धीमी"),
 119          "noun": None,          # no natural numeric phrasing; comparisons only
 120          "possessive": None,
 121          "unit": None,
 122          "range": None,
 123      },
 124      "wealth": {
 125          "more": ("अमीर", "अमीर"),
 126          "less": ("गरीब", "गरीब"),
 127          "noun": None,
 128          "possessive": None,
 129          "unit": None,
 130          "range": None,
 131      },
 132  }
 133  
 134  NUMERIC_ATTRIBUTES = [k for k, v in ATTRIBUTES.items() if v["noun"]]
 135  
 136  # ---------------------------------------------------------------------------
 137  # Templates, split so that a test question is phrased in a way never trained on.
 138  # ---------------------------------------------------------------------------
 139  # {a} {b} {c}   entity names
 140  # {adj}         the comparative adjective, already inflected for the subject
 141  # {q}           the adjective used in the question, inflected for the answer
 142  # Sentences end with the Devanagari danda (।), as Hindi prose does.
 143  
 144  TEMPLATES = {
 145      "compare2": {
 146          "train": [
 147              "{a} {b} से {adj} है। दोनों में से {q} कौन है?",
 148              "{a} {b} से {adj} है। बताइए, {q} कौन है?",
 149              "{a} {b} से {adj} है। तो {q} कौन हुआ?",
 150              "यह ज्ञात है कि {a} {b} से {adj} है। {q} कौन है?",
 151          ],
 152          "val": ["{a} {b} से {adj} है। इनमें {q} कौन है?"],
 153          "test": ["ध्यान दीजिए: {a} {b} से {adj} है। {q} कौन है?"],
 154      },
 155      "transitive3": {
 156          "train": [
 157              "{a} {b} से {adj} है। {b} {c} से {adj} है। सबसे {q} कौन है?",
 158              "{a} {b} से {adj} है, और {b} {c} से {adj} है। सबसे {q} कौन है?",
 159              "अगर {a} {b} से {adj} है और {b} {c} से {adj} है, तो सबसे {q} कौन है?",
 160              "{a} {b} से {adj} है। {b} {c} से {adj} है। तीनों में सबसे {q} कौन है?",
 161          ],
 162          "val": ["मान लीजिए {a} {b} से {adj} है और {b} {c} से {adj} है। सबसे {q} कौन है?"],
 163          "test": ["{a} {b} से {adj} है। इसी तरह {b} {c} से {adj} है। सबसे {q} कौन है?"],
 164      },
 165      "numeric2": {
 166          "train": [
 167              "{a} {p} {noun} {n1} {unit} है। {b} {p} {noun} {n2} {unit} है। {q} कौन है?",
 168              "{a} {p} {noun} {n1} {unit} है, और {b} {p} {noun} {n2} {unit} है। {q} कौन है?",
 169              "{a} {p} {noun} {n1} {unit} और {b} {p} {noun} {n2} {unit} है। दोनों में {q} कौन है?",
 170          ],
 171          "val": ["{a} {p} {noun} {n1} {unit} है। वहीं {b} {p} {noun} {n2} {unit} है। {q} कौन है?"],
 172          "test": ["दिया गया है कि {a} {p} {noun} {n1} {unit} है और {b} {p} {noun} {n2} {unit} है। {q} कौन है?"],
 173      },
 174      "numeric3": {
 175          "train": [
 176              "{a} {p} {noun} {n1} {unit} है। {b} {p} {noun} {n2} {unit} है। {c} {p} {noun} {n3} {unit} है। सबसे {q} कौन है?",
 177              "{a} {p} {noun} {n1} {unit}, {b} {p} {noun} {n2} {unit} और {c} {p} {noun} {n3} {unit} है। सबसे {q} कौन है?",
 178              "तीन लोगों में {a} {p} {noun} {n1} {unit}, {b} {p} {noun} {n2} {unit} और {c} {p} {noun} {n3} {unit} है। सबसे {q} कौन है?",
 179          ],
 180          "val": ["{a} {p} {noun} {n1} {unit} है, {b} {p} {noun} {n2} {unit} है, और {c} {p} {noun} {n3} {unit} है। सबसे {q} कौन है?"],
 181          "test": ["यहाँ {a} {p} {noun} {n1} {unit}, {b} {p} {noun} {n2} {unit}, तथा {c} {p} {noun} {n3} {unit} है। सबसे {q} कौन है?"],
 182      },
 183  }
 184  
 185  # Hindi text normally writes numbers in Devanagari digits; the Phase 1 corpus
 186  # contains both forms, so the tokenizer handles either.
 187  DEVANAGARI_DIGITS = "०१२३४५६७८९"
 188  
 189  
 190  def to_devanagari(number: int) -> str:
 191      """Render an integer using Devanagari digits.
 192  
 193      Args:
 194          number: A non-negative integer.
 195  
 196      Returns:
 197          The same number written with the characters ०-९.
 198      """
 199      return "".join(DEVANAGARI_DIGITS[int(d)] for d in str(number))
 200  
 201  
 202  def inflect(attribute: str, direction: str, gender: str) -> str:
 203      """Return the comparative adjective agreeing with the subject's gender.
 204  
 205      Args:
 206          attribute: Key into :data:`ATTRIBUTES`.
 207          direction: ``"more"`` or ``"less"``.
 208          gender: ``"m"`` or ``"f"``.
 209  
 210      Returns:
 211          The inflected adjective, e.g. ``"लंबा"`` or ``"लंबी"``.
 212      """
 213      masculine, feminine = ATTRIBUTES[attribute][direction]
 214      return masculine if gender == "m" else feminine
 215  
 216  
 217  class ReasoningGenerator:
 218      """Builds one split of the Hindi comparative-reasoning dataset."""
 219  
 220      def __init__(self, split: str, seed: int) -> None:
 221          """Fix the name pool and template pool for one split.
 222  
 223          Args:
 224              split: ``"train"``, ``"val"`` or ``"test"``.
 225              seed: Seed, so a split is byte-for-byte reproducible.
 226  
 227          Raises:
 228              ValueError: If the split name is unknown.
 229          """
 230          pools = {"train": NAMES_TRAIN, "val": NAMES_VAL, "test": NAMES_TEST}
 231          if split not in pools:
 232              raise ValueError(f"unknown split: {split}")
 233          self.split = split
 234          self.names = pools[split]
 235          self.rng = random.Random(seed)
 236  
 237      def _template(self, task: str) -> tuple[str, int]:
 238          """Pick a template for this task from this split's pool.
 239  
 240          Args:
 241              task: Task family name.
 242  
 243          Returns:
 244              The template string and its index within the split's pool.
 245          """
 246          pool = TEMPLATES[task][self.split]
 247          index = self.rng.randrange(len(pool))
 248          return pool[index], index
 249  
 250      def _pick(self, count: int) -> list[tuple[str, str]]:
 251          """Draw distinct entities of the *same* gender from this split's pool.
 252  
 253          Same-gender is not a stylistic choice -- it closes a shortcut. Hindi
 254          comparative adjectives inflect ("छोटा" for a man, "छोटी" for a woman),
 255          so a question ending "सबसे छोटी कौन है?" asked about one woman and two
 256          men has only one grammatically possible answer. The model could score
 257          by matching gender agreement and never perform the comparison at all.
 258          An earlier draft of this generator leaked exactly that way in 45% of
 259          examples.
 260  
 261          Drawing every entity in an example from one gender makes the adjective
 262          carry no information about which entity is meant, so the answer can
 263          only come from the comparison itself.
 264  
 265          Args:
 266              count: How many entities are needed.
 267  
 268          Returns:
 269              A list of ``(name, gender)`` pairs, all different, all same gender.
 270          """
 271          gender = self.rng.choice(["m", "f"])
 272          pool = [p for p in self.names if p[1] == gender]
 273          if len(pool) < count:
 274              pool = self.names
 275          return self.rng.sample(pool, count)
 276  
 277      def compare2(self) -> dict:
 278          """One stated comparison, then a question about the same pair."""
 279          (a, ga), (b, gb) = self._pick(2)
 280          attribute = self.rng.choice(list(ATTRIBUTES))
 281          # Both the stated direction and the asked direction vary. If the
 282          # sentence says "a is taller than b" and the question asks who is
 283          # taller, the answer is a; if it asks who is shorter, it is b. Varying
 284          # the stated direction too means the answer is not simply "whichever
 285          # name was mentioned first".
 286          stated_more = self.rng.random() < 0.5
 287          ask_more = self.rng.random() < 0.5
 288          answer, answer_gender = (a, ga) if stated_more == ask_more else (b, gb)
 289          template, index = self._template("compare2")
 290          text = template.format(
 291              a=a, b=b,
 292              adj=inflect(attribute, "more" if stated_more else "less", ga),
 293              q=inflect(attribute, "more" if ask_more else "less", answer_gender),
 294          )
 295          return {
 296              "task": "compare2", "attribute": attribute, "template_index": index,
 297              "prompt": text, "answer": answer, "entities": [a, b],
 298              "stated_direction": "more" if stated_more else "less",
 299              "asked_direction": "more" if ask_more else "less",
 300              "hops": 1,
 301          }
 302  
 303      def transitive3(self) -> dict:
 304          """Two chained comparisons; the answer requires following the chain."""
 305          (a, ga), (b, gb), (c, gc) = self._pick(3)
 306          attribute = self.rng.choice(list(ATTRIBUTES))
 307          # The chain is stated as "a REL b" then "b REL c". If REL is "more" the
 308          # order is a > b > c; if "less" it is a < b < c. Either way the two
 309          # extremes are a and c, and which one answers the question depends on
 310          # whether the asked direction matches the stated one -- that agreement
 311          # is the two-hop inference being tested.
 312          stated_more = self.rng.random() < 0.5
 313          ask_more = self.rng.random() < 0.5
 314          answer, answer_gender = (a, ga) if stated_more == ask_more else (c, gc)
 315          template, index = self._template("transitive3")
 316          text = template.format(
 317              a=a, b=b, c=c,
 318              adj=inflect(attribute, "more" if stated_more else "less", ga),
 319              q=inflect(attribute, "more" if ask_more else "less", answer_gender),
 320          )
 321          return {
 322              "task": "transitive3", "attribute": attribute, "template_index": index,
 323              "prompt": text, "answer": answer, "entities": [a, b, c],
 324              "stated_direction": "more" if stated_more else "less",
 325              "asked_direction": "more" if ask_more else "less",
 326              "hops": 2,
 327          }
 328  
 329      def numeric2(self) -> dict:
 330          """Two measured quantities, compared."""
 331          (a, ga), (b, gb) = self._pick(2)
 332          attribute = self.rng.choice(NUMERIC_ATTRIBUTES)
 333          spec = ATTRIBUTES[attribute]
 334          low, high = spec["range"]
 335          n1, n2 = self.rng.sample(range(low, high), 2)
 336          ask_more = self.rng.random() < 0.5
 337          bigger = (a, ga) if n1 > n2 else (b, gb)
 338          smaller = (b, gb) if n1 > n2 else (a, ga)
 339          answer, answer_gender = bigger if ask_more else smaller
 340          template, index = self._template("numeric2")
 341          text = template.format(
 342              a=a, b=b, p=spec["possessive"], noun=spec["noun"], unit=spec["unit"],
 343              n1=to_devanagari(n1), n2=to_devanagari(n2),
 344              q=inflect(attribute, "more" if ask_more else "less", answer_gender),
 345          )
 346          return {
 347              "task": "numeric2", "attribute": attribute, "template_index": index,
 348              "prompt": text, "answer": answer, "entities": [a, b],
 349              "values": [n1, n2], "hops": 1,
 350          }
 351  
 352      def numeric3(self) -> dict:
 353          """Three measured quantities; pick the largest or smallest."""
 354          (a, ga), (b, gb), (c, gc) = self._pick(3)
 355          attribute = self.rng.choice(NUMERIC_ATTRIBUTES)
 356          spec = ATTRIBUTES[attribute]
 357          low, high = spec["range"]
 358          n1, n2, n3 = self.rng.sample(range(low, high), 3)
 359          people = [(a, ga, n1), (b, gb, n2), (c, gc, n3)]
 360          ask_more = self.rng.random() < 0.5
 361          chosen = max(people, key=lambda t: t[2]) if ask_more \
 362              else min(people, key=lambda t: t[2])
 363          template, index = self._template("numeric3")
 364          text = template.format(
 365              a=a, b=b, c=c, p=spec["possessive"], noun=spec["noun"],
 366              unit=spec["unit"],
 367              n1=to_devanagari(n1), n2=to_devanagari(n2), n3=to_devanagari(n3),
 368              q=inflect(attribute, "more" if ask_more else "less", chosen[1]),
 369          )
 370          return {
 371              "task": "numeric3", "attribute": attribute, "template_index": index,
 372              "prompt": text, "answer": chosen[0], "entities": [a, b, c],
 373              "values": [n1, n2, n3], "hops": 1,
 374          }
 375  
 376      def build(self, count: int) -> list[dict]:
 377          """Generate one split, with the four task families equally represented.
 378  
 379          Duplicate prompts are dropped: with a small name pool the test split in
 380          particular can otherwise repeat itself, which would make the reported
 381          accuracy depend on how often one question happened to appear.
 382  
 383          Args:
 384              count: Target number of unique examples.
 385  
 386          Returns:
 387              The generated examples, shuffled.
 388          """
 389          makers = [self.compare2, self.transitive3, self.numeric2, self.numeric3]
 390          seen: set[str] = set()
 391          out: list[dict] = []
 392          attempts = 0
 393          while len(out) < count and attempts < count * 200:
 394              attempts += 1
 395              row = makers[len(out) % len(makers)]()
 396              if row["prompt"] in seen:
 397                  continue
 398              seen.add(row["prompt"])
 399              row["split"] = self.split
 400              row["language"] = LANGUAGE
 401              out.append(row)
 402          self.rng.shuffle(out)
 403          return out
 404  
 405  
 406  def describe(rows: list[dict]) -> dict:
 407      """Summarise one split, for the statistics the specification asks for.
 408  
 409      Args:
 410          rows: Generated examples.
 411  
 412      Returns:
 413          Counts by task, attribute and template, plus answer balance.
 414      """
 415      return {
 416          "examples": len(rows),
 417          "by_task": dict(Counter(r["task"] for r in rows)),
 418          "by_attribute": dict(Counter(r["attribute"] for r in rows)),
 419          "distinct_templates": len({(r["task"], r["template_index"]) for r in rows}),
 420          "distinct_answers": len({r["answer"] for r in rows}),
 421          "distinct_prompts": len({r["prompt"] for r in rows}),
 422          "mean_prompt_chars": round(sum(len(r["prompt"]) for r in rows) / max(len(rows), 1), 1),
 423      }
 424  
 425  
 426  def main() -> None:
 427      """Generate all three splits and write them as JSONL."""
 428      parser = argparse.ArgumentParser(description=__doc__)
 429      parser.add_argument("--train", type=int, default=20000)
 430      parser.add_argument("--val", type=int, default=2000)
 431      parser.add_argument("--test", type=int, default=2000)
 432      parser.add_argument("--seed", type=int, default=20260914)
 433      args = parser.parse_args()
 434  
 435      OUT_DIR.mkdir(parents=True, exist_ok=True)
 436      stats = {"language": LANGUAGE, "splits": {}}
 437      all_names = {"train": set(), "val": set(), "test": set()}
 438  
 439      for offset, (split, count) in enumerate(
 440          (("train", args.train), ("val", args.val), ("test", args.test))
 441      ):
 442          gen = ReasoningGenerator(split, args.seed + offset)
 443          rows = gen.build(count)
 444          path = OUT_DIR / f"{split}.jsonl"
 445          with open(path, "w", encoding="utf-8") as fh:
 446              for row in rows:
 447                  fh.write(json.dumps(row, ensure_ascii=False) + "\n")
 448          stats["splits"][split] = describe(rows)
 449          all_names[split] = {n for r in rows for n in r["entities"]}
 450          print(f"  {split:5s} {len(rows):6,} examples -> {path.name}")
 451  
 452      # The leakage guarantee is asserted here rather than assumed: if the name
 453      # pools ever overlap, this fails loudly instead of quietly inflating
 454      # test accuracy.
 455      for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
 456          overlap = all_names[a] & all_names[b]
 457          assert not overlap, f"name leakage between {a} and {b}: {overlap}"
 458      stats["leakage_check"] = {
 459          "entity_names_disjoint_across_splits": True,
 460          "templates_disjoint_across_splits": True,
 461          "train_names": len(all_names["train"]),
 462          "val_names": len(all_names["val"]),
 463          "test_names": len(all_names["test"]),
 464      }
 465      (OUT_DIR / "dataset_stats.json").write_text(
 466          json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
 467      print(f"\n  wrote {OUT_DIR / 'dataset_stats.json'}")
 468      print("  name pools verified disjoint across all three splits")
 469  
 470  
 471  if __name__ == "__main__":
 472      main()
```


## hindi/reasoning/dataset.py

```python
   1  """Turn Model H's Hindi reasoning JSONL into batches the Transformer can train on.
   2  
   3  This is supervised finetuning on top of a causal language model, so the format
   4  is deliberately plain: prompt and answer are concatenated into one token
   5  sequence and the model is trained to continue it. Two details matter.
   6  
   7  **Loss is computed on the answer only.** The prompt's target positions are set
   8  to ``-1``, which the model's ``cross_entropy(..., ignore_index=-1)`` already
   9  skips. Training on the prompt too would spend most of the gradient teaching the
  10  model to reproduce question templates it will always be given for free at test
  11  time, and the thing being measured -- can it name the right entity -- would get
  12  a small share of the signal.
  13  
  14  **Batches are bucketed by prompt length rather than padded to a common one.**
  15  The model uses *learned absolute* positional embeddings, so a token's
  16  representation depends on its index in the sequence. Left-padding would shift
  17  every real token to a position it never occupied during pretraining;
  18  right-padding is fine for training but breaks generation, which appends at the
  19  end. Grouping examples whose prompts tokenize to the same length sidesteps both
  20  problems: every sequence in a batch is genuinely aligned, and no padding is
  21  needed for the prompt at all.
  22  
  23  This file belongs to **Model H** (Hindi). Model L has its own independent copy
  24  under ``nepali/reasoning/``; no data, tokenizer or vocabulary is shared.
  25  """
  26  
  27  from __future__ import annotations
  28  
  29  import json
  30  import random
  31  from collections import defaultdict
  32  from pathlib import Path
  33  
  34  import torch
  35  
  36  REPO_ROOT = Path(__file__).resolve().parents[2]
  37  
  38  LANGUAGE = "hindi"
  39  DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
  40  DEFAULT_TOKENIZER = REPO_ROOT / "hindi/tokenizer/hindi_unigram_8000.model"
  41  
  42  
  43  def load_tokenizer(model_path: str | Path | None = None):
  44      """Open this language's SentencePiece model.
  45  
  46      Args:
  47          model_path: Override for the tokenizer model; defaults to Model H's.
  48  
  49      Returns:
  50          A loaded ``SentencePieceProcessor``.
  51  
  52      Raises:
  53          FileNotFoundError: If the model file is missing.
  54      """
  55      import sentencepiece as spm
  56  
  57      path = Path(model_path) if model_path else DEFAULT_TOKENIZER
  58      if not path.is_file():
  59          raise FileNotFoundError(f"tokenizer model not found: {path}")
  60      return spm.SentencePieceProcessor(model_file=str(path))
  61  
  62  
  63  class ReasoningDataset:
  64      """One split of the reasoning corpus, tokenized and ready to batch."""
  65  
  66      def __init__(
  67          self,
  68          split: str,
  69          tokenizer=None,
  70          data_dir: str | Path | None = None,
  71          tokenizer_model: str | Path | None = None,
  72          device: str = "cpu",
  73      ) -> None:
  74          """Load and tokenize one split.
  75  
  76          Args:
  77              split: ``"train"``, ``"val"`` or ``"test"``.
  78              tokenizer: An already-loaded SentencePiece processor, reused across
  79                  splits so the model file is only read once.
  80              data_dir: Directory holding the JSONL splits.
  81              tokenizer_model: Path to the tokenizer, if ``tokenizer`` is None.
  82              device: Device the produced tensors should live on.
  83  
  84          Raises:
  85              FileNotFoundError: If the split's JSONL file is missing.
  86          """
  87          self.split = split
  88          self.device = device
  89          self.sp = tokenizer or load_tokenizer(tokenizer_model)
  90          self.eos_id = self.sp.eos_id()
  91          # Nepali's BPE and Hindi's Unigram assign different ids to their special
  92          # tokens, so nothing here may be hardcoded. Padding ids never contribute
  93          # to the loss (their targets are -1); when the tokenizer declares no pad
  94          # token, reusing EOS is harmless for the same reason.
  95          self.pad_id = self.sp.pad_id() if self.sp.pad_id() >= 0 else self.eos_id
  96  
  97          directory = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
  98          path = directory / f"{split}.jsonl"
  99          if not path.is_file():
 100              raise FileNotFoundError(
 101                  f"{path} not found - run "
 102                  f"`python {LANGUAGE}/reasoning/generate.py` first"
 103              )
 104          self.rows = [json.loads(line) for line in open(path, encoding="utf-8")]
 105          for row in self.rows:
 106              self._encode(row)
 107  
 108      def _encode(self, row: dict) -> None:
 109          """Tokenize one example in place.
 110  
 111          The answer is encoded with a leading space so SentencePiece emits it as
 112          a word-initial piece (``▁राम``), which is how the name appears in
 113          running text. Encoding it bare would produce a word-*internal* piece
 114          that the model has rarely seen in that position.
 115  
 116          Args:
 117              row: One dataset record; gains ``prompt_ids`` and ``answer_ids``.
 118          """
 119          row["prompt_ids"] = self.sp.encode(row["prompt"])
 120          row["answer_ids"] = self.sp.encode(" " + row["answer"]) + [self.eos_id]
 121  
 122      def __len__(self) -> int:
 123          """Return the number of examples in this split."""
 124          return len(self.rows)
 125  
 126      def sequence_stats(self) -> dict:
 127          """Summarise tokenized lengths, for the report's dataset table.
 128  
 129          Returns:
 130              Mean and maximum prompt / answer / total token counts.
 131          """
 132          prompts = [len(r["prompt_ids"]) for r in self.rows]
 133          answers = [len(r["answer_ids"]) for r in self.rows]
 134          totals = [p + a for p, a in zip(prompts, answers)]
 135          return {
 136              "examples": len(self.rows),
 137              "prompt_tokens_mean": round(sum(prompts) / len(prompts), 2),
 138              "prompt_tokens_max": max(prompts),
 139              "answer_tokens_mean": round(sum(answers) / len(answers), 2),
 140              "answer_tokens_max": max(answers),
 141              "total_tokens_max": max(totals),
 142              "total_tokens": sum(totals),
 143          }
 144  
 145      def _tensor_pair(self, rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
 146          """Build the ``(inputs, targets)`` pair for a group of examples.
 147  
 148          Every sequence is ``prompt + answer``. Position ``i`` of the input
 149          predicts token ``i + 1``, so the target at ``i`` is masked whenever
 150          token ``i + 1`` still belongs to the prompt. The first unmasked target
 151          is therefore the first answer token, predicted from the final prompt
 152          token -- exactly the decision the evaluation measures.
 153  
 154          Args:
 155              rows: Examples to batch. Prompts must already share a length.
 156  
 157          Returns:
 158              Inputs of shape ``(B, T)`` and targets of shape ``(B, T)``, where
 159              masked target positions hold ``-1``.
 160          """
 161          sequences = [r["prompt_ids"] + r["answer_ids"] for r in rows]
 162          width = max(len(s) for s in sequences)
 163          inputs, targets = [], []
 164          for row, seq in zip(rows, sequences):
 165              padded = seq + [self.pad_id] * (width - len(seq))
 166              x = padded[:-1]
 167              y = padded[1:]
 168              # -1 everywhere the next token is not part of the answer: the whole
 169              # prompt, and any padding past the real end of this sequence.
 170              start = len(row["prompt_ids"]) - 1
 171              end = len(seq) - 1
 172              y = [t if start <= i < end else -1 for i, t in enumerate(y)]
 173              inputs.append(x)
 174              targets.append(y)
 175          return (
 176              torch.tensor(inputs, dtype=torch.long, device=self.device),
 177              torch.tensor(targets, dtype=torch.long, device=self.device),
 178          )
 179  
 180      def batches(
 181          self, batch_size: int, shuffle: bool = True, seed: int = 0
 182      ) -> list[tuple[torch.Tensor, torch.Tensor]]:
 183          """Build training batches, grouped so each batch shares a prompt length.
 184  
 185          Args:
 186              batch_size: Examples per batch.
 187              shuffle: Whether to shuffle within buckets and across batches.
 188              seed: Seed for that shuffling, so an epoch is reproducible.
 189  
 190          Returns:
 191              A list of ``(inputs, targets)`` pairs covering the whole split.
 192          """
 193          buckets: dict[int, list[dict]] = defaultdict(list)
 194          for row in self.rows:
 195              buckets[len(row["prompt_ids"])].append(row)
 196  
 197          rng = random.Random(seed)
 198          out = []
 199          for length in sorted(buckets):
 200              group = list(buckets[length])
 201              if shuffle:
 202                  rng.shuffle(group)
 203              for i in range(0, len(group), batch_size):
 204                  out.append(self._tensor_pair(group[i : i + batch_size]))
 205          if shuffle:
 206              rng.shuffle(out)
 207          return out
 208  
 209      def prompt_batches(
 210          self, batch_size: int
 211      ) -> list[tuple[torch.Tensor, list[dict]]]:
 212          """Build prompt-only batches for generation, bucketed by length.
 213  
 214          Generation appends tokens at the end of the sequence, so every prompt in
 215          a batch must end at the same index. Bucketing by token length gives that
 216          exactly, with no padding and therefore no positional distortion.
 217  
 218          Args:
 219              batch_size: Prompts per batch.
 220  
 221          Returns:
 222              A list of ``(prompt_ids, rows)`` pairs, where ``prompt_ids`` has
 223              shape ``(B, T)`` and ``rows`` are the matching dataset records.
 224          """
 225          buckets: dict[int, list[dict]] = defaultdict(list)
 226          for row in self.rows:
 227              buckets[len(row["prompt_ids"])].append(row)
 228  
 229          out = []
 230          for length in sorted(buckets):
 231              group = buckets[length]
 232              for i in range(0, len(group), batch_size):
 233                  chunk = group[i : i + batch_size]
 234                  ids = torch.tensor(
 235                      [r["prompt_ids"] for r in chunk],
 236                      dtype=torch.long,
 237                      device=self.device,
 238                  )
 239                  out.append((ids, chunk))
 240          return out
```


## hindi/reasoning/finetune.py

```python
   1  """Finetune Model H on its own Hindi reasoning corpus.
   2  
   3  The protocol the specification fixes is followed exactly:
   4  
   5  * training starts from **this language's** pretrained checkpoint, never from
   6    the other model and never from scratch;
   7  * the tokenizer and vocabulary are the ones Phase 1 produced and are not
   8    touched, so the embedding and output matrices keep their pretrained rows;
   9  * every hyperparameter is recorded in a config file that is written next to the
  10    checkpoints;
  11  * checkpoints carry model weights, optimizer state, scheduler state, the step
  12    counter and the config -- the same resume-capable format as pretraining, so
  13    the Phase 2 evaluation and attention tooling loads a finetuned checkpoint
  14    without modification. That reuse is what deliverable 4 asks for.
  15  
  16  Loss is computed on answer tokens only; see ``dataset.py`` for why.
  17  
  18  Usage
  19  -----
  20      python -m hindi.reasoning.finetune --config hindi/configs/finetune.yaml
  21      python -m hindi.reasoning.finetune --config ... --no-resume --max-minutes 45
  22  """
  23  
  24  from __future__ import annotations
  25  
  26  import argparse
  27  import json
  28  import math
  29  import time
  30  from contextlib import nullcontext
  31  from dataclasses import asdict, dataclass, field
  32  from pathlib import Path
  33  
  34  import numpy as np
  35  import torch
  36  import yaml
  37  
  38  from ..model.config import GPTConfig
  39  from ..model.transformer import GPT
  40  from ..train.train import CosineWarmupSchedule, resolve_device, resolve_dtype
  41  from .dataset import ReasoningDataset, load_tokenizer
  42  
  43  REPO_ROOT = Path(__file__).resolve().parents[2]
  44  
  45  LANGUAGE = "hindi"
  46  
  47  
  48  @dataclass
  49  class FinetuneConfig:
  50      """Every knob of the reasoning finetuning run, logged alongside the weights.
  51  
  52      Attributes:
  53          language: Language name, used for run naming and output paths.
  54          pretrained_checkpoint: Path to this language's Phase 2 checkpoint. The
  55              run starts from these weights.
  56          data_dir: Directory holding the reasoning JSONL splits.
  57          tokenizer_model: Phase 1 tokenizer, held fixed throughout.
  58          out_dir: Directory for finetuned checkpoints and the config snapshot.
  59          epochs: Passes over the training split.
  60          batch_size: Examples per optimiser step.
  61          learning_rate: Peak learning rate after warmup. Well below the
  62              pretraining peak: the model already has a language model worth
  63              preserving, and a large step would wash it out before it learns the
  64              task.
  65          min_lr: Floor the cosine schedule decays to.
  66          warmup_steps: Linear warmup length in steps.
  67          weight_decay: AdamW decoupled weight decay, matrices only.
  68          beta1: AdamW first-moment decay.
  69          beta2: AdamW second-moment decay.
  70          grad_clip: Global gradient-norm clip; ``0`` disables clipping.
  71          eval_interval: Steps between validation passes.
  72          log_interval: Steps between training-loss log lines.
  73          seed: Base RNG seed.
  74          device: ``"cuda"``, ``"cpu"``, or ``"auto"``.
  75          dtype: Autocast dtype: ``"bfloat16"``, ``"float16"`` or ``"float32"``.
  76          notes: Free-text description carried into the checkpoint.
  77      """
  78  
  79      language: str = LANGUAGE
  80      pretrained_checkpoint: str = ""
  81      data_dir: str = f"{LANGUAGE}/reasoning/data"
  82      tokenizer_model: str = ""
  83      out_dir: str = f"checkpoints/{LANGUAGE}_reasoning"
  84      epochs: int = 3
  85      batch_size: int = 32
  86      learning_rate: float = 1e-4
  87      min_lr: float = 1e-5
  88      warmup_steps: int = 100
  89      weight_decay: float = 0.1
  90      beta1: float = 0.9
  91      beta2: float = 0.95
  92      grad_clip: float = 1.0
  93      eval_interval: int = 100
  94      log_interval: int = 25
  95      seed: int = 20260914
  96      device: str = "auto"
  97      dtype: str = "float16"
  98      notes: str = ""
  99  
 100      @classmethod
 101      def load(cls, path: str | Path) -> "FinetuneConfig":
 102          """Read a finetuning config from YAML.
 103  
 104          Args:
 105              path: Path to the YAML file.
 106  
 107          Returns:
 108              The populated config.
 109          """
 110          data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
 111          known = {f for f in cls.__dataclass_fields__}
 112          return cls(**{k: v for k, v in data.items() if k in known})
 113  
 114      def save(self, path: str | Path) -> None:
 115          """Write this config to YAML next to the checkpoints.
 116  
 117          Args:
 118              path: Destination file.
 119          """
 120          Path(path).write_text(
 121              yaml.safe_dump(asdict(self), sort_keys=False, allow_unicode=True),
 122              encoding="utf-8",
 123          )
 124  
 125  
 126  class FineTuner:
 127      """Runs one language's reasoning finetuning, with checkpointing and resume."""
 128  
 129      def __init__(self, cfg: FinetuneConfig, resume: bool = True) -> None:
 130          """Load the pretrained model and prepare the finetuning run.
 131  
 132          Args:
 133              cfg: The finetuning configuration.
 134              resume: Whether to continue from ``latest.pt`` in ``out_dir``.
 135  
 136          Raises:
 137              FileNotFoundError: If the pretrained checkpoint does not exist.
 138          """
 139          self.cfg = cfg
 140          self.device = resolve_device(cfg.device)
 141          self.dtype = resolve_dtype(cfg.dtype, self.device)
 142          torch.manual_seed(cfg.seed)
 143          np.random.seed(cfg.seed)
 144  
 145          self.out_dir = REPO_ROOT / cfg.out_dir
 146          self.out_dir.mkdir(parents=True, exist_ok=True)
 147          self.log_path = self.out_dir / "finetune_log.jsonl"
 148  
 149          pretrained = Path(cfg.pretrained_checkpoint)
 150          if not pretrained.is_absolute():
 151              pretrained = REPO_ROOT / pretrained
 152          if not pretrained.is_file():
 153              raise FileNotFoundError(
 154                  f"pretrained checkpoint not found: {pretrained}\n"
 155                  "Download this language's Phase 2 checkpoint first (README has "
 156                  "the Drive link)."
 157              )
 158          ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
 159  
 160          # The architecture is not a choice here -- it is whatever Phase 2
 161          # trained, read straight out of the checkpoint. Rebuilding it from the
 162          # YAML would risk a silent mismatch.
 163          self.model_config = GPTConfig.from_dict(ckpt["config"]["model"])
 164          self.pretrain_config = ckpt["config"]["train"]
 165          self.pretrain_step = ckpt.get("step", -1)
 166  
 167          self.model = GPT(self.model_config).to(self.device)
 168          self.model.load_state_dict(ckpt["model"])
 169          print(
 170              f"  loaded pretrained weights from {pretrained.name} "
 171              f"(step {self.pretrain_step:,})"
 172          )
 173  
 174          # A fresh optimizer: the pretraining moments describe a different
 175          # objective on a different corpus and carrying them over would fight the
 176          # new, much smaller learning rate.
 177          self.optimizer = self.model.configure_optimizers(
 178              weight_decay=cfg.weight_decay,
 179              learning_rate=cfg.learning_rate,
 180              betas=(cfg.beta1, cfg.beta2),
 181              device_type="cuda" if self.device.startswith("cuda") else "cpu",
 182          )
 183          self.scaler = torch.amp.GradScaler(
 184              "cuda", enabled=(self.dtype == torch.float16)
 185          )
 186          self.autocast = (
 187              torch.amp.autocast(device_type="cuda", dtype=self.dtype)
 188              if self.device.startswith("cuda")
 189              else nullcontext()
 190          )
 191  
 192          tokenizer_model = cfg.tokenizer_model or None
 193          if tokenizer_model and not Path(tokenizer_model).is_absolute():
 194              tokenizer_model = REPO_ROOT / tokenizer_model
 195          self.tokenizer = load_tokenizer(tokenizer_model)
 196          data_dir = REPO_ROOT / cfg.data_dir
 197          self.train_data = ReasoningDataset(
 198              "train", self.tokenizer, data_dir, device=self.device
 199          )
 200          self.val_data = ReasoningDataset(
 201              "val", self.tokenizer, data_dir, device=self.device
 202          )
 203          if self.tokenizer.get_piece_size() != self.model_config.vocab_size:
 204              raise ValueError(
 205                  f"tokenizer vocabulary ({self.tokenizer.get_piece_size()}) does "
 206                  f"not match the pretrained model ({self.model_config.vocab_size}). "
 207                  "The finetuning tokenizer must be the pretraining one."
 208              )
 209  
 210          steps_per_epoch = math.ceil(len(self.train_data) / cfg.batch_size)
 211          self.max_steps = steps_per_epoch * cfg.epochs
 212          self.schedule = CosineWarmupSchedule(
 213              self.optimizer,
 214              peak_lr=cfg.learning_rate,
 215              min_lr=cfg.min_lr,
 216              warmup_steps=min(cfg.warmup_steps, max(self.max_steps // 10, 1)),
 217              max_steps=self.max_steps,
 218          )
 219          self.val_batches = self.val_data.batches(
 220              cfg.batch_size, shuffle=False, seed=cfg.seed
 221          )
 222  
 223          self.step = 0
 224          self.examples_seen = 0
 225          self.best_val_loss = float("inf")
 226          if resume:
 227              self._maybe_resume()
 228  
 229          cfg.save(self.out_dir / "finetune_config.yaml")
 230          print(
 231              f"  {len(self.train_data):,} train / {len(self.val_data):,} val "
 232              f"examples | {steps_per_epoch:,} steps/epoch | "
 233              f"{self.max_steps:,} steps total"
 234          )
 235  
 236      # -- checkpointing ----------------------------------------------------
 237  
 238      def _checkpoint_payload(self) -> dict:
 239          """Assemble everything needed to resume this run exactly.
 240  
 241          The ``config.model`` and ``config.train`` keys are carried forward from
 242          the pretrained checkpoint verbatim. That is deliberate: the Phase 2
 243          ``Evaluator`` and attention toolkit read those keys to rebuild the model
 244          and find the packed corpus, so preserving them is what lets a finetuned
 245          checkpoint be dropped into the existing tooling unchanged.
 246  
 247          Returns:
 248              The dict that gets serialised to a checkpoint file.
 249          """
 250          return {
 251              "model": self.model.state_dict(),
 252              "optimizer": self.optimizer.state_dict(),
 253              "scheduler": self.schedule.state_dict(),
 254              "scaler": self.scaler.state_dict(),
 255              "step": self.step,
 256              "examples_seen": self.examples_seen,
 257              "best_val_loss": self.best_val_loss,
 258              "config": {
 259                  "model": self.model_config.to_dict(),
 260                  "train": self.pretrain_config,
 261                  "finetune": asdict(self.cfg),
 262                  "notes": self.cfg.notes,
 263              },
 264              "stage": "reasoning_finetune",
 265              "pretrained_checkpoint": self.cfg.pretrained_checkpoint,
 266              "pretrained_step": self.pretrain_step,
 267              "parameter_count": self.model.parameter_count(),
 268              "torch_rng_state": torch.get_rng_state(),
 269          }
 270  
 271      def save_checkpoint(self, tag: str | None = None) -> Path:
 272          """Write a checkpoint.
 273  
 274          Args:
 275              tag: Optional filename tag; defaults to the step number.
 276  
 277          Returns:
 278              Path to the checkpoint that was written.
 279          """
 280          name = f"ckpt_{tag}.pt" if tag else f"ckpt_step{self.step:06d}.pt"
 281          path = self.out_dir / name
 282          payload = self._checkpoint_payload()
 283          torch.save(payload, path)
 284          torch.save(payload, self.out_dir / "latest.pt")
 285          return path
 286  
 287      def _maybe_resume(self) -> None:
 288          """Restore state from ``latest.pt`` in the output directory, if present."""
 289          path = self.out_dir / "latest.pt"
 290          if not path.exists():
 291              print("  no finetuning checkpoint yet; starting from the "
 292                    "pretrained weights")
 293              return
 294          ckpt = torch.load(path, map_location=self.device, weights_only=False)
 295          self.model.load_state_dict(ckpt["model"])
 296          self.optimizer.load_state_dict(ckpt["optimizer"])
 297          self.schedule.load_state_dict(ckpt["scheduler"])
 298          self.scaler.load_state_dict(ckpt["scaler"])
 299          self.step = ckpt["step"]
 300          self.examples_seen = ckpt.get("examples_seen", 0)
 301          self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
 302          if "torch_rng_state" in ckpt:
 303              torch.set_rng_state(ckpt["torch_rng_state"].cpu())
 304          print(f"  resumed finetuning from {path} at step {self.step:,}")
 305  
 306      # -- evaluation -------------------------------------------------------
 307  
 308      @torch.no_grad()
 309      def evaluate(self) -> float:
 310          """Average the answer-token loss over the validation split.
 311  
 312          Returns:
 313              Mean validation cross-entropy in nats over answer tokens only.
 314          """
 315          self.model.eval()
 316          losses = []
 317          for x, y in self.val_batches:
 318              with self.autocast:
 319                  _, loss, _ = self.model(x, targets=y)
 320              losses.append(loss.item())
 321          self.model.train()
 322          return float(np.mean(losses)) if losses else float("nan")
 323  
 324      # -- logging ----------------------------------------------------------
 325  
 326      def _log(self, record: dict) -> None:
 327          """Append one record to the run's JSONL finetuning log.
 328  
 329          Args:
 330              record: The fields to write.
 331          """
 332          with open(self.log_path, "a", encoding="utf-8") as fh:
 333              fh.write(json.dumps(record) + "\n")
 334  
 335      # -- training ---------------------------------------------------------
 336  
 337      def train(self, max_minutes: float | None = None) -> None:
 338          """Run the finetuning loop to completion.
 339  
 340          Args:
 341              max_minutes: Optional wall-clock budget; the run stops cleanly and
 342                  saves once it is exceeded.
 343          """
 344          self.model.train()
 345          started = time.time()
 346          print(f"  finetuning on {self.device} ({self.dtype}) "
 347                f"for {self.max_steps:,} steps")
 348  
 349          # A resumed run must not replay the batches it already trained on. Each
 350          # epoch is regenerated from a seed derived from the epoch number, so the
 351          # batch order is reproducible, and the position within it is simply the
 352          # step counter -- skipping forward to it lands exactly where the
 353          # interrupted run stopped.
 354          consumed = 0
 355          for epoch in range(self.cfg.epochs):
 356              batches = self.train_data.batches(
 357                  self.cfg.batch_size, shuffle=True, seed=self.cfg.seed + epoch
 358              )
 359              if consumed + len(batches) <= self.step:
 360                  consumed += len(batches)
 361                  continue  # this whole epoch was already trained on
 362              for x, y in batches:
 363                  if consumed < self.step:
 364                      consumed += 1
 365                      continue
 366                  consumed += 1
 367                  if self.step >= self.max_steps:
 368                      break
 369                  lr = self.schedule.step(self.step)
 370  
 371                  with self.autocast:
 372                      _, loss, _ = self.model(x, targets=y)
 373                  self.scaler.scale(loss).backward()
 374                  if self.cfg.grad_clip > 0:
 375                      self.scaler.unscale_(self.optimizer)
 376                      torch.nn.utils.clip_grad_norm_(
 377                          self.model.parameters(), self.cfg.grad_clip
 378                      )
 379                  self.scaler.step(self.optimizer)
 380                  self.scaler.update()
 381                  self.optimizer.zero_grad(set_to_none=True)
 382  
 383                  self.step += 1
 384                  self.examples_seen += x.size(0)
 385  
 386                  if self.step % self.cfg.log_interval == 0:
 387                      record = {
 388                          "step": self.step,
 389                          "epoch": epoch,
 390                          "train_loss": round(loss.item(), 4),
 391                          "lr": lr,
 392                          "examples_seen": self.examples_seen,
 393                          "elapsed_s": round(time.time() - started, 1),
 394                      }
 395                      self._log(record)
 396                      print(
 397                          f"    step {self.step:,}/{self.max_steps:,} "
 398                          f"loss {record['train_loss']:.4f} lr {lr:.2e}"
 399                      )
 400  
 401                  if self.step % self.cfg.eval_interval == 0:
 402                      val = self.evaluate()
 403                      self._log({
 404                          "step": self.step,
 405                          "epoch": epoch,
 406                          "val_loss": round(val, 4),
 407                          "val_perplexity": round(math.exp(min(val, 20)), 3),
 408                      })
 409                      print(f"    step {self.step:,} val loss {val:.4f} "
 410                            f"(ppl {math.exp(min(val, 20)):.2f})")
 411                      if val < self.best_val_loss:
 412                          self.best_val_loss = val
 413                          self.save_checkpoint("best")
 414                          print(f"      new best; saved ckpt_best.pt")
 415  
 416                  if max_minutes and (time.time() - started) / 60 >= max_minutes:
 417                      print("  wall-clock budget reached; saving and stopping")
 418                      self.save_checkpoint("final")
 419                      return
 420  
 421          val = self.evaluate()
 422          if val < self.best_val_loss:
 423              self.best_val_loss = val
 424              self.save_checkpoint("best")
 425          self._log({
 426              "step": self.step,
 427              "val_loss": round(val, 4),
 428              "final": True,
 429              "elapsed_s": round(time.time() - started, 1),
 430          })
 431          self.save_checkpoint("final")
 432          print(f"  done: {self.step:,} steps in "
 433                f"{(time.time() - started) / 60:.1f} min, "
 434                f"best val loss {self.best_val_loss:.4f}")
 435  
 436  
 437  def main() -> None:
 438      """Command-line entry point for reasoning finetuning."""
 439      parser = argparse.ArgumentParser(description=__doc__)
 440      parser.add_argument("--config", default=f"{LANGUAGE}/configs/finetune.yaml")
 441      parser.add_argument("--pretrained-checkpoint", default=None,
 442                          help="override the checkpoint named in the config")
 443      parser.add_argument("--out-dir", default=None)
 444      parser.add_argument("--device", default=None)
 445      parser.add_argument("--dtype", default=None)
 446      parser.add_argument("--epochs", type=int, default=None)
 447      parser.add_argument("--max-minutes", type=float, default=None)
 448      parser.add_argument("--no-resume", action="store_true")
 449      args = parser.parse_args()
 450  
 451      cfg = FinetuneConfig.load(args.config)
 452      for name in ("pretrained_checkpoint", "out_dir", "device", "dtype", "epochs"):
 453          value = getattr(args, name, None)
 454          if value is not None:
 455              setattr(cfg, name, value)
 456  
 457      tuner = FineTuner(cfg, resume=not args.no_resume)
 458      tuner.train(max_minutes=args.max_minutes)
 459  
 460  
 461  if __name__ == "__main__":
 462      main()
```


## hindi/reasoning/evaluate_reasoning.py

```python
   1  """Score a checkpoint on Model H's Hindi reasoning test set.
   2  
   3  The specification asks for *pretrained vs. finetuned* accuracy, so this script
   4  must work on both, and the comparison has to be fair. That turns out to need
   5  two different measurements, because a pretrained language model has never seen
   6  this prompt format and will not spontaneously answer with a bare name:
   7  
   8  **Exact match after greedy decoding.** The model is given the question and
   9  generates freely; the continuation is compared to the gold answer. This is the
  10  honest end-to-end number and the headline metric, but on a pretrained
  11  checkpoint it mostly measures "does it know the output format", which is not
  12  the thing under study, and it bottoms out near zero.
  13  
  14  **Forced choice among the entities named in the prompt.** The candidates are
  15  exactly the people the question mentions. For each, the model scores the
  16  continuation, and the highest-scoring candidate is taken as its answer. A model
  17  that cannot format an answer but *has* learned something about comparatives can
  18  still show it here, and chance is a known 1/2 or 1/3, so the pretrained
  19  baseline is interpretable rather than a floor artefact.
  20  
  21  Scores are length-normalised (mean log-probability per answer token). Without
  22  that, short names win on total log-probability for no reason connected to the
  23  reasoning, which would silently bias every comparison toward whichever entity
  24  happened to tokenize shortest.
  25  
  26  Both numbers are broken down by task family and by hop count, because the
  27  interesting question is not the average -- it is whether two-hop transitive
  28  chains behave differently from one-hop comparisons.
  29  
  30  Usage
  31  -----
  32      python -m hindi.reasoning.evaluate_reasoning \\
  33          --checkpoint checkpoints/hindi/ckpt_best.pt --tag pretrained
  34      python -m hindi.reasoning.evaluate_reasoning \\
  35          --checkpoint checkpoints/hindi_reasoning/ckpt_best.pt --tag finetuned
  36  """
  37  
  38  from __future__ import annotations
  39  
  40  import argparse
  41  import json
  42  import math
  43  import re
  44  import unicodedata
  45  from collections import defaultdict
  46  from pathlib import Path
  47  
  48  import torch
  49  import torch.nn.functional as F
  50  
  51  from ..model.config import GPTConfig
  52  from ..model.transformer import GPT
  53  from .dataset import ReasoningDataset, load_tokenizer
  54  
  55  REPO_ROOT = Path(__file__).resolve().parents[2]
  56  
  57  LANGUAGE = "hindi"
  58  
  59  # Devanagari danda and double danda, plus ASCII sentence punctuation. A model
  60  # that answers "राम।" has got the answer right and added terminal punctuation;
  61  # counting that as wrong would understate every model equally but pointlessly.
  62  _TRIM = re.compile(r"^[\s।॥.,!?;:\"'()\[\]]+|[\s।॥.,!?;:\"'()\[\]]+$")
  63  
  64  
  65  def normalise(text: str) -> str:
  66      """Strip punctuation and normalise Unicode for answer comparison.
  67  
  68      NFC normalisation matters for Devanagari: the same visible string can be
  69      encoded with precomposed or decomposed forms, and a byte comparison would
  70      call two identical-looking names different.
  71  
  72      Args:
  73          text: Raw generated or gold text.
  74  
  75      Returns:
  76          The trimmed, NFC-normalised string.
  77      """
  78      return _TRIM.sub("", unicodedata.normalize("NFC", text)).strip()
  79  
  80  
  81  class ReasoningEvaluator:
  82      """Loads a checkpoint and scores it on the reasoning splits."""
  83  
  84      def __init__(
  85          self,
  86          checkpoint: str | Path,
  87          data_dir: str | Path | None = None,
  88          tokenizer_model: str | Path | None = None,
  89          device: str = "auto",
  90      ) -> None:
  91          """Restore a model and open the reasoning corpus.
  92  
  93          Args:
  94              checkpoint: Path to a ``.pt`` file from pretraining or finetuning.
  95              data_dir: Override for the reasoning data directory.
  96              tokenizer_model: Override for the tokenizer model path.
  97              device: ``"auto"``, ``"cuda"`` or ``"cpu"``.
  98          """
  99          self.device = (
 100              ("cuda" if torch.cuda.is_available() else "cpu")
 101              if device == "auto"
 102              else device
 103          )
 104          path = Path(checkpoint)
 105          if not path.is_absolute():
 106              path = REPO_ROOT / path
 107          ckpt = torch.load(path, map_location=self.device, weights_only=False)
 108  
 109          self.model_config = GPTConfig.from_dict(ckpt["config"]["model"])
 110          self.model = GPT(self.model_config).to(self.device)
 111          self.model.load_state_dict(ckpt["model"])
 112          self.model.eval()
 113  
 114          # "stage" is written only by the finetuner, so its presence is what
 115          # distinguishes a finetuned checkpoint from a pretrained one.
 116          self.stage = ckpt.get("stage", "pretrained")
 117          self.step = ckpt.get("step", -1)
 118          self.checkpoint_path = str(path)
 119  
 120          tok = Path(tokenizer_model) if tokenizer_model else None
 121          if tok and not tok.is_absolute():
 122              tok = REPO_ROOT / tok
 123          self.tokenizer = load_tokenizer(tok)
 124          self.data_dir = Path(data_dir) if data_dir else None
 125          self.eos_id = self.tokenizer.eos_id()
 126  
 127      def load_split(self, split: str) -> ReasoningDataset:
 128          """Open one split of the reasoning corpus.
 129  
 130          Args:
 131              split: ``"train"``, ``"val"`` or ``"test"``.
 132  
 133          Returns:
 134              The tokenized split.
 135          """
 136          return ReasoningDataset(
 137              split, self.tokenizer, self.data_dir, device=self.device
 138          )
 139  
 140      # -- metric 1: free generation ---------------------------------------
 141  
 142      @torch.no_grad()
 143      def greedy_answers(
 144          self, data: ReasoningDataset, batch_size: int = 64
 145      ) -> list[dict]:
 146          """Greedily decode an answer for every example.
 147  
 148          Args:
 149              data: The split to score.
 150              batch_size: Prompts per generation batch.
 151  
 152          Returns:
 153              One record per example with the generated text and whether it
 154              exactly matches the gold answer.
 155          """
 156          budget = max(len(r["answer_ids"]) for r in data.rows) + 2
 157          results = []
 158          for prompt_ids, rows in data.prompt_batches(batch_size):
 159              out = self.model.generate(
 160                  prompt_ids, max_new_tokens=budget, greedy=True,
 161                  eos_id=self.eos_id,
 162              )
 163              generated = out[:, prompt_ids.size(1):].tolist()
 164              for row, ids in zip(rows, generated):
 165                  # Everything after the first EOS is the model padding out the
 166                  # batch for other sequences, not part of this answer.
 167                  if self.eos_id in ids:
 168                      ids = ids[: ids.index(self.eos_id)]
 169                  text = normalise(self.tokenizer.decode(ids))
 170                  entities = [normalise(e) for e in row["entities"]]
 171                  # Which of the entities the question named did the model
 172                  # produce, by position in the prompt? -1 means it produced
 173                  # something that is not in the prompt at all -- typically a
 174                  # training name, which is the signature of having memorised the
 175                  # answer prior instead of reading the question.
 176                  model_index = entities.index(text) if text in entities else -1
 177                  results.append({
 178                      "task": row["task"], "hops": row["hops"],
 179                      "attribute": row["attribute"], "prompt": row["prompt"],
 180                      "gold": row["answer"], "generated": text,
 181                      "correct": text == normalise(row["answer"]),
 182                      "entities": row["entities"],
 183                      "n_candidates": len(entities),
 184                      "gold_index": entities.index(normalise(row["answer"])),
 185                      "model_index": model_index,
 186                  })
 187          return results
 188  
 189      # -- metric 2: forced choice -----------------------------------------
 190  
 191      @torch.no_grad()
 192      def forced_choice(self, data: ReasoningDataset) -> list[dict]:
 193          """Pick, among the entities the prompt names, the best-scoring answer.
 194  
 195          Each candidate is appended to the prompt and scored by its mean
 196          log-probability per answer token. All candidates for one example share a
 197          prompt, so they are evaluated in a single batched forward pass.
 198  
 199          Args:
 200              data: The split to score.
 201  
 202          Returns:
 203              One record per example with the chosen entity and correctness.
 204          """
 205          results = []
 206          for row in data.rows:
 207              prompt_ids = row["prompt_ids"]
 208              candidates = row["entities"]
 209              encoded = [
 210                  self.tokenizer.encode(" " + name) + [self.eos_id]
 211                  for name in candidates
 212              ]
 213              width = len(prompt_ids) + max(len(e) for e in encoded)
 214              batch = torch.full(
 215                  (len(candidates), width), self.eos_id,
 216                  dtype=torch.long, device=self.device,
 217              )
 218              for i, answer_ids in enumerate(encoded):
 219                  seq = prompt_ids + answer_ids
 220                  batch[i, : len(seq)] = torch.tensor(
 221                      seq, dtype=torch.long, device=self.device
 222                  )
 223  
 224              logits, _, _ = self.model(batch[:, :-1])
 225              log_probs = F.log_softmax(logits.float(), dim=-1)
 226              targets = batch[:, 1:]
 227              gathered = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
 228  
 229              scores = []
 230              for i, answer_ids in enumerate(encoded):
 231                  start = len(prompt_ids) - 1
 232                  end = start + len(answer_ids)
 233                  span = gathered[i, start:end]
 234                  # Length-normalised: see the module docstring. Without this a
 235                  # one-token name beats a three-token one on total log-prob
 236                  # regardless of what the question asked.
 237                  scores.append(float(span.mean()))
 238  
 239              best = int(max(range(len(scores)), key=lambda i: scores[i]))
 240              chosen = candidates[best]
 241              results.append({
 242                  "task": row["task"], "hops": row["hops"],
 243                  "attribute": row["attribute"], "prompt": row["prompt"],
 244                  "gold": row["answer"], "chosen": chosen,
 245                  "n_candidates": len(candidates),
 246                  "correct": normalise(chosen) == normalise(row["answer"]),
 247                  "scores": [round(s, 4) for s in scores],
 248              })
 249          return results
 250  
 251      # -- metric 3: answer-token loss --------------------------------------
 252  
 253      @torch.no_grad()
 254      def answer_loss(self, data: ReasoningDataset, batch_size: int = 32) -> dict:
 255          """Average cross-entropy over gold answer tokens.
 256  
 257          This is the quantity finetuning actually minimises, so it is the
 258          cleanest before/after number -- unlike accuracy, it moves smoothly and
 259          is not floored by output-format problems.
 260  
 261          Args:
 262              data: The split to score.
 263              batch_size: Examples per batch.
 264  
 265          Returns:
 266              Mean loss in nats and the corresponding perplexity.
 267          """
 268          totals, counts = 0.0, 0
 269          for x, y in data.batches(batch_size, shuffle=False):
 270              logits, _, _ = self.model(x)
 271              loss = F.cross_entropy(
 272                  logits.float().reshape(-1, logits.size(-1)),
 273                  y.reshape(-1), ignore_index=-1, reduction="sum",
 274              )
 275              totals += float(loss)
 276              counts += int((y != -1).sum())
 277          mean = totals / max(counts, 1)
 278          return {
 279              "answer_token_loss": round(mean, 4),
 280              "answer_token_perplexity": round(math.exp(min(mean, 20)), 3),
 281              "answer_tokens_scored": counts,
 282          }
 283  
 284  
 285  def breakdown(records: list[dict], key: str) -> dict:
 286      """Accuracy grouped by one field.
 287  
 288      Args:
 289          records: Scored records carrying ``correct`` and the grouping field.
 290          key: Field to group by, e.g. ``"task"`` or ``"hops"``.
 291  
 292      Returns:
 293          Mapping from group value to accuracy, count and number correct.
 294      """
 295      groups: dict = defaultdict(lambda: [0, 0])
 296      for record in records:
 297          bucket = groups[record[key]]
 298          bucket[0] += int(record["correct"])
 299          bucket[1] += 1
 300      return {
 301          str(name): {
 302              "accuracy": round(hits / total, 4),
 303              "correct": hits,
 304              "total": total,
 305          }
 306          for name, (hits, total) in sorted(groups.items(), key=lambda kv: str(kv[0]))
 307      }
 308  
 309  
 310  def position_analysis(records: list[dict]) -> dict:
 311      """Explain *where* a model's answers come from, not just whether they match.
 312  
 313      Accuracy alone cannot distinguish a model guessing uniformly among the
 314      entities in the question from one that has latched onto a positional habit
 315      -- "always answer with the last name mentioned" -- or from one ignoring the
 316      question and emitting a memorised training name. Those have very different
 317      implications and this separates them.
 318  
 319      For each task family it reports how often the model produced the entity at
 320      each prompt position, how often the gold answer sits at each position, and
 321      how often the model produced a name the question never mentioned.
 322  
 323      Args:
 324          records: Greedy-generation records carrying ``model_index``,
 325              ``gold_index`` and ``n_candidates``.
 326  
 327      Returns:
 328          Per-task distributions over answer position, plus the off-prompt rate.
 329      """
 330      groups: dict[str, list[dict]] = defaultdict(list)
 331      for record in records:
 332          groups[record["task"]].append(record)
 333  
 334      out = {}
 335      for task, rows in sorted(groups.items()):
 336          width = max(r["n_candidates"] for r in rows)
 337          model = [0] * width
 338          gold = [0] * width
 339          off_prompt = 0
 340          for row in rows:
 341              gold[row["gold_index"]] += 1
 342              if row["model_index"] < 0:
 343                  off_prompt += 1
 344              else:
 345                  model[row["model_index"]] += 1
 346          total = len(rows)
 347          out[task] = {
 348              "examples": total,
 349              "model_answer_position": [round(c / total, 4) for c in model],
 350              "gold_answer_position": [round(c / total, 4) for c in gold],
 351              "answered_off_prompt": round(off_prompt / total, 4),
 352              "candidates": width,
 353          }
 354      return out
 355  
 356  
 357  def qualitative(records: list[dict], limit: int = 8) -> dict:
 358      """Collect example successes and failures for the report.
 359  
 360      Args:
 361          records: Scored records.
 362          limit: How many of each to keep.
 363  
 364      Returns:
 365          Two lists of examples, the first correct and the second not.
 366      """
 367      field = "generated" if "generated" in records[0] else "chosen"
 368      hits = [r for r in records if r["correct"]][:limit]
 369      misses = [r for r in records if not r["correct"]][:limit]
 370      keep = lambda r: {
 371          "task": r["task"], "prompt": r["prompt"],
 372          "gold": r["gold"], "model": r[field],
 373      }
 374      return {
 375          "correct": [keep(r) for r in hits],
 376          "incorrect": [keep(r) for r in misses],
 377      }
 378  
 379  
 380  def build_report(
 381      ev: ReasoningEvaluator,
 382      data: ReasoningDataset,
 383      tag: str,
 384      split: str,
 385      batch_size: int = 64,
 386  ) -> dict:
 387      """Score a checkpoint on one split and assemble the full JSON report.
 388  
 389      Factored out of :func:`main` so the Kaggle notebook and the command line
 390      produce byte-identical reports. Two code paths computing "the same" metrics
 391      is how a side-by-side table stops being comparable.
 392  
 393      Args:
 394          ev: An evaluator with the checkpoint already loaded.
 395          data: The tokenized split to score.
 396          tag: Label for this run, e.g. ``"pretrained"`` or ``"finetuned"``.
 397          split: Name of the split, recorded in the report.
 398          batch_size: Batch size for generation and loss.
 399  
 400      Returns:
 401          The report dict.
 402      """
 403      generated = ev.greedy_answers(data, batch_size=batch_size)
 404      chosen = ev.forced_choice(data)
 405      losses = ev.answer_loss(data, batch_size=batch_size)
 406  
 407      gen_accuracy = sum(r["correct"] for r in generated) / len(generated)
 408      fc_accuracy = sum(r["correct"] for r in chosen) / len(chosen)
 409      # Chance depends on how many entities each question names, which varies by
 410      # task, so it is computed rather than assumed to be one half.
 411      chance = sum(1 / r["n_candidates"] for r in chosen) / len(chosen)
 412  
 413      return {
 414          "language": LANGUAGE,
 415          "tag": tag,
 416          "stage": ev.stage,
 417          "checkpoint": ev.checkpoint_path,
 418          "checkpoint_step": ev.step,
 419          "split": split,
 420          "examples": len(data),
 421          "sequence_stats": data.sequence_stats(),
 422          "exact_match": {
 423              "accuracy": round(gen_accuracy, 4),
 424              "correct": sum(r["correct"] for r in generated),
 425              "by_task": breakdown(generated, "task"),
 426              "by_hops": breakdown(generated, "hops"),
 427              "by_attribute": breakdown(generated, "attribute"),
 428          },
 429          "forced_choice": {
 430              "accuracy": round(fc_accuracy, 4),
 431              "correct": sum(r["correct"] for r in chosen),
 432              "chance_accuracy": round(chance, 4),
 433              "by_task": breakdown(chosen, "task"),
 434              "by_hops": breakdown(chosen, "hops"),
 435              "by_attribute": breakdown(chosen, "attribute"),
 436          },
 437          "answer_token_loss": losses,
 438          "error_analysis": position_analysis(generated),
 439          "qualitative_generation": qualitative(generated),
 440          "qualitative_forced_choice": qualitative(chosen),
 441      }
 442  
 443  
 444  def main() -> None:
 445      """Command-line entry point: score one checkpoint and write a JSON report."""
 446      parser = argparse.ArgumentParser(description=__doc__)
 447      parser.add_argument("--checkpoint", required=True)
 448      parser.add_argument("--split", default="test")
 449      parser.add_argument("--tag", default=None,
 450                          help="label for the output file, e.g. pretrained")
 451      parser.add_argument("--data-dir", default=None)
 452      parser.add_argument("--tokenizer-model", default=None)
 453      parser.add_argument("--device", default="auto")
 454      parser.add_argument("--batch-size", type=int, default=64)
 455      parser.add_argument("--out", default=None)
 456      args = parser.parse_args()
 457  
 458      ev = ReasoningEvaluator(
 459          args.checkpoint, data_dir=args.data_dir,
 460          tokenizer_model=args.tokenizer_model, device=args.device,
 461      )
 462      data = ev.load_split(args.split)
 463      tag = args.tag or ev.stage
 464      print(f"  {LANGUAGE} | {tag} | {len(data):,} {args.split} examples "
 465            f"on {ev.device}")
 466  
 467      report = build_report(ev, data, tag, args.split, args.batch_size)
 468  
 469      out = Path(args.out) if args.out else (
 470          REPO_ROOT / f"report/phase3/{LANGUAGE}_reasoning_{tag}.json"
 471      )
 472      out.parent.mkdir(parents=True, exist_ok=True)
 473      out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
 474                     encoding="utf-8")
 475  
 476      print(f"    exact match   {report['exact_match']['accuracy']:6.2%}")
 477      print(f"    forced choice {report['forced_choice']['accuracy']:6.2%}  "
 478            f"(chance {report['forced_choice']['chance_accuracy']:.2%})")
 479      print(f"    answer loss   "
 480            f"{report['answer_token_loss']['answer_token_loss']:.4f}")
 481      print(f"  wrote {out}")
 482  
 483  
 484  if __name__ == "__main__":
 485      main()
```


## hindi/reasoning/attention_compare.py

```python
   1  """Compare Model H's attention before and after reasoning finetuning.
   2  
   3  Deliverable 4 asks for a pretrained-vs-finetuned attention comparison on at
   4  least one early and one late layer, specifically on comparative-reasoning
   5  prompts. Three choices make that comparison mean something.
   6  
   7  **The same prompts go through both models.** Attention entropy depends heavily
   8  on what is being attended to, so feeding the two checkpoints different text
   9  would confound the finetuning effect with a text effect. The prompts here come
  10  from the reasoning *test* split -- held-out names, held-out templates -- so
  11  neither checkpoint has memorised them and the finetuned model is not being
  12  flattered by its training data.
  13  
  14  **Prompts are bucketed to a single length.** Mean attention distance is measured
  15  in token positions and entropy's uniform baseline is ``log(t)`` at position
  16  ``t``, so both grow with sequence length. Comparing a 20-token prompt against a
  17  40-token one would show a difference that is purely arithmetic. Every prompt in
  18  a comparison batch has the same token count.
  19  
  20  **Per-head deltas are reported, not just per-layer means.** Finetuning a small
  21  model for a narrow task tends to change a few heads sharply rather than shift
  22  every head a little; a layer mean hides exactly that.
  23  
  24  This file belongs to **Model H** (Hindi). Model L has its own independent copy
  25  under ``nepali/reasoning/``.
  26  
  27  Usage
  28  -----
  29      python -m hindi.reasoning.attention_compare \\
  30          --pretrained checkpoints/hindi/ckpt_best.pt \\
  31          --finetuned  checkpoints/hindi_reasoning/ckpt_best.pt
  32  """
  33  
  34  from __future__ import annotations
  35  
  36  import argparse
  37  import json
  38  from collections import defaultdict
  39  from pathlib import Path
  40  
  41  import torch
  42  
  43  from ..eval import attention as attn_lib
  44  from ..eval import viz
  45  from ..model.config import GPTConfig
  46  from ..model.transformer import GPT
  47  from .dataset import ReasoningDataset, load_tokenizer
  48  
  49  REPO_ROOT = Path(__file__).resolve().parents[2]
  50  
  51  LANGUAGE = "hindi"
  52  
  53  
  54  def load_model(checkpoint: str | Path, device: str) -> tuple[GPT, dict]:
  55      """Restore a model from either a pretrained or a finetuned checkpoint.
  56  
  57      Args:
  58          checkpoint: Path to the ``.pt`` file.
  59          device: Device to place the model on.
  60  
  61      Returns:
  62          The model in eval mode, and a dict of provenance fields for the report.
  63      """
  64      path = Path(checkpoint)
  65      if not path.is_absolute():
  66          path = REPO_ROOT / path
  67      ckpt = torch.load(path, map_location=device, weights_only=False)
  68      config = GPTConfig.from_dict(ckpt["config"]["model"])
  69      model = GPT(config).to(device)
  70      model.load_state_dict(ckpt["model"])
  71      model.eval()
  72      meta = {
  73          "checkpoint": str(path),
  74          "stage": ckpt.get("stage", "pretrained"),
  75          "step": ckpt.get("step", -1),
  76      }
  77      return model, meta
  78  
  79  
  80  def reasoning_prompt_batch(
  81      data: ReasoningDataset, batch_size: int, device: str
  82  ) -> tuple[torch.Tensor, list[dict]]:
  83      """Take one batch of equal-length comparative-reasoning prompts.
  84  
  85      The largest same-length bucket is used, so the batch is as representative as
  86      a single length allows.
  87  
  88      Args:
  89          data: The reasoning split to draw from.
  90          batch_size: How many prompts to include.
  91          device: Device for the returned tensor.
  92  
  93      Returns:
  94          Token ids of shape ``(B, T)`` and the matching dataset records.
  95      """
  96      buckets: dict[int, list[dict]] = defaultdict(list)
  97      for row in data.rows:
  98          buckets[len(row["prompt_ids"])].append(row)
  99      # Prefer a long prompt bucket: two-hop transitive questions are the ones
 100      # where long-range attention would have to show up if it shows up at all.
 101      length = max(buckets, key=lambda k: (len(buckets[k]) >= batch_size, k))
 102      rows = buckets[length][:batch_size]
 103      ids = torch.tensor([r["prompt_ids"] for r in rows],
 104                         dtype=torch.long, device=device)
 105      return ids, rows
 106  
 107  
 108  def per_head_table(summary: dict) -> dict[str, dict]:
 109      """Flatten a model attention summary to ``"L{layer}H{head}"`` keys.
 110  
 111      Args:
 112          summary: Output of :func:`attention.summarise_model_attention`.
 113  
 114      Returns:
 115          Mapping from head label to its entropy and mean distance.
 116      """
 117      out = {}
 118      for layer in summary["layers"]:
 119          index = layer["layer"]
 120          for head, (entropy, distance) in enumerate(
 121              zip(layer["entropy"], layer["mean_distance"])
 122          ):
 123              out[f"L{index}H{head}"] = {
 124                  "entropy": round(float(entropy), 4),
 125                  "mean_distance": round(float(distance), 4),
 126              }
 127      return out
 128  
 129  
 130  def compare(
 131      pretrained: str | Path,
 132      finetuned: str | Path,
 133      data_dir: str | Path | None = None,
 134      tokenizer_model: str | Path | None = None,
 135      split: str = "test",
 136      batch_size: int = 16,
 137      device: str = "auto",
 138      fig_dir: str | Path | None = None,
 139  ) -> dict:
 140      """Run both checkpoints on identical prompts and summarise the difference.
 141  
 142      Args:
 143          pretrained: Path to the Phase 2 checkpoint.
 144          finetuned: Path to the Phase 3 checkpoint.
 145          data_dir: Override for the reasoning data directory.
 146          tokenizer_model: Override for the tokenizer.
 147          split: Which reasoning split to draw prompts from.
 148          batch_size: Prompts per comparison batch.
 149          device: ``"auto"``, ``"cuda"`` or ``"cpu"``.
 150          fig_dir: Where heatmaps are written.
 151  
 152      Returns:
 153          The comparison report, including per-head deltas and the layers plotted.
 154      """
 155      resolved = (
 156          ("cuda" if torch.cuda.is_available() else "cpu")
 157          if device == "auto" else device
 158      )
 159      tok = Path(tokenizer_model) if tokenizer_model else None
 160      if tok and not tok.is_absolute():
 161          tok = REPO_ROOT / tok
 162      tokenizer = load_tokenizer(tok)
 163      data = ReasoningDataset(split, tokenizer, data_dir, device=resolved)
 164      ids, rows = reasoning_prompt_batch(data, batch_size, resolved)
 165      tokens = [tokenizer.decode([int(t)]) for t in ids[0].cpu().numpy()]
 166  
 167      figures = Path(fig_dir) if fig_dir else REPO_ROOT / "report/phase3/figures"
 168      figures.mkdir(parents=True, exist_ok=True)
 169  
 170      report: dict = {
 171          "language": LANGUAGE,
 172          "split": split,
 173          "prompts_compared": len(rows),
 174          "sequence_length": int(ids.size(1)),
 175          "uniform_baseline_entropy": round(
 176              attn_lib.uniform_baseline_entropy(int(ids.size(1))), 4
 177          ),
 178          "example_prompts": [r["prompt"] for r in rows[:3]],
 179          "example_tokens": tokens,
 180          "models": {},
 181          "figures": [],
 182      }
 183  
 184      summaries = {}
 185      for name, checkpoint in (("pretrained", pretrained), ("finetuned", finetuned)):
 186          model, meta = load_model(checkpoint, resolved)
 187          attentions = attn_lib.collect_attention(model, ids)
 188          early, late = 0, len(attentions) - 1
 189          for layer in (early, late):
 190              path = viz.plot_attention_heatmap(
 191                  attentions[layer][0].cpu().numpy(), tokens, layer, [0, 1, 2, 3],
 192                  figures / f"{LANGUAGE}_{name}_reasoning_layer{layer}.png",
 193                  language=f"{LANGUAGE} ({name})",
 194              )
 195              report["figures"].append(str(path.name))
 196          summary = attn_lib.summarise_model_attention(model, ids)
 197          summary.update(meta)
 198          summaries[name] = summary
 199          report["models"][name] = {
 200              **meta,
 201              "mean_entropy": summary.get("mean_entropy"),
 202              "mean_distance": summary.get("mean_distance"),
 203              "per_head": per_head_table(summary),
 204              "layers_plotted": [early, late],
 205          }
 206          del model
 207  
 208      before = report["models"]["pretrained"]["per_head"]
 209      after = report["models"]["finetuned"]["per_head"]
 210      deltas = {
 211          head: {
 212              "entropy_delta": round(
 213                  after[head]["entropy"] - before[head]["entropy"], 4),
 214              "distance_delta": round(
 215                  after[head]["mean_distance"] - before[head]["mean_distance"], 4),
 216          }
 217          for head in before if head in after
 218      }
 219      report["per_head_delta"] = deltas
 220      # The heads that moved most are what the written analysis should discuss;
 221      # ranking them here means the report quotes a measurement, not an impression.
 222      report["largest_entropy_shifts"] = sorted(
 223          deltas.items(), key=lambda kv: abs(kv[1]["entropy_delta"]), reverse=True
 224      )[:8]
 225      report["largest_distance_shifts"] = sorted(
 226          deltas.items(), key=lambda kv: abs(kv[1]["distance_delta"]), reverse=True
 227      )[:8]
 228  
 229      path = viz.plot_attention_summary(
 230          summaries, figures / f"{LANGUAGE}_attention_pretrained_vs_finetuned.png"
 231      )
 232      report["figures"].append(str(path.name))
 233      return report
 234  
 235  
 236  def main() -> None:
 237      """Command-line entry point for the attention comparison."""
 238      parser = argparse.ArgumentParser(description=__doc__)
 239      parser.add_argument("--pretrained", required=True)
 240      parser.add_argument("--finetuned", required=True)
 241      parser.add_argument("--split", default="test")
 242      parser.add_argument("--data-dir", default=None)
 243      parser.add_argument("--tokenizer-model", default=None)
 244      parser.add_argument("--batch-size", type=int, default=16)
 245      parser.add_argument("--device", default="auto")
 246      parser.add_argument("--fig-dir", default=None)
 247      parser.add_argument("--out", default=None)
 248      args = parser.parse_args()
 249  
 250      report = compare(
 251          args.pretrained, args.finetuned, data_dir=args.data_dir,
 252          tokenizer_model=args.tokenizer_model, split=args.split,
 253          batch_size=args.batch_size, device=args.device, fig_dir=args.fig_dir,
 254      )
 255      out = Path(args.out) if args.out else (
 256          REPO_ROOT / f"report/phase3/{LANGUAGE}_attention_comparison.json"
 257      )
 258      out.parent.mkdir(parents=True, exist_ok=True)
 259      out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
 260                     encoding="utf-8")
 261  
 262      print(f"  {LANGUAGE}: {report['prompts_compared']} prompts, "
 263            f"T={report['sequence_length']}")
 264      for name, block in report["models"].items():
 265          print(f"    {name:11s} mean entropy {block['mean_entropy']}  "
 266                f"mean distance {block['mean_distance']}")
 267      print("    largest entropy shifts:")
 268      for head, delta in report["largest_entropy_shifts"][:4]:
 269          print(f"      {head}: {delta['entropy_delta']:+.4f} nats, "
 270                f"{delta['distance_delta']:+.3f} positions")
 271      print(f"  wrote {out}")
 272  
 273  
 274  if __name__ == "__main__":
 275      main()
```


## hindi/configs/finetune.yaml

```yaml
   1  # Model H — Hindi. Phase 3 reasoning-finetuning configuration.
   2  #
   3  # Independent of Model L in every graded respect: its own reasoning corpus
   4  # (hindi/reasoning/data/, no sentence or entity name shared with Nepali), its
   5  # own Phase 1 tokenizer and 8,000-token vocabulary, and its own pretrained
   6  # weights. Only source code is shared.
   7  #
   8  # The architecture is NOT specified here. It is read out of the pretrained
   9  # checkpoint, so a mismatch between this file and the trained model is
  10  # impossible by construction.
  11  
  12  language: hindi
  13  
  14  # Starting point: Model H's own Phase 2 checkpoint. The spec forbids
  15  # initialising one language's model from the other's, and forbids starting from
  16  # scratch — this must be the Hindi pretrained checkpoint and nothing else.
  17  pretrained_checkpoint: checkpoints/hindi/ckpt_best.pt
  18  
  19  data_dir: hindi/reasoning/data
  20  
  21  # Held fixed from Phase 1, as the protocol requires. Keeping the tokenizer means
  22  # the embedding and output matrices keep their pretrained rows; swapping it
  23  # would strand every one of them.
  24  tokenizer_model: hindi/tokenizer/hindi_unigram_8000.model
  25  
  26  out_dir: checkpoints/hindi_reasoning
  27  
  28  # Run B. Run A evaluated every 100 steps and found its best validation loss at
  29  # the very first eval point, step 100, with loss rising monotonically after --
  30  # so the checkpoint that got reported was selected from a grid too coarse to
  31  # locate the actual optimum. With 40 training names and 14 templates the model
  32  # memorises the answer prior within a few hundred examples, so the useful part
  33  # of this run is its first few dozen steps.
  34  #
  35  # Only the checkpoint-selection granularity and the epoch count change here.
  36  # The learning rate, batch size, dataset and test split are identical to run A,
  37  # so the two runs differ in one variable and remain directly comparable.
  38  epochs: 2
  39  batch_size: 32
  40  
  41  # Peak LR is 6e-5, one tenth of the 6e-4 pretraining peak. The model already
  42  # holds a language model worth keeping, and a pretraining-sized step would
  43  # overwrite it before the task is learned — the usual failure mode is a model
  44  # that answers in the right format but has forgotten how to read the question.
  45  learning_rate: 6.0e-5
  46  min_lr: 6.0e-6
  47  # Run A's best validation loss landed at step 100, which is exactly where
  48  # warmup ends and the learning rate first reaches its peak. Before that the
  49  # model is barely moving; after it, memorisation of the 40 training names
  50  # accelerates. Evaluating every 20 steps now samples that window properly.
  51  warmup_steps: 100
  52  
  53  weight_decay: 0.1
  54  beta1: 0.9
  55  beta2: 0.95
  56  grad_clip: 1.0
  57  
  58  eval_interval: 20
  59  log_interval: 25
  60  
  61  seed: 20260914
  62  device: auto
  63  
  64  # float16 with a GradScaler. The Kaggle T4 is compute-capability 7.5 and has no
  65  # bfloat16 support, so the pretraining config's bfloat16 cannot be reused here.
  66  dtype: float16
  67  
  68  notes: >
  69    Model H (higher-resource) reasoning finetuning. 20,000 synthetic Hindi
  70    comparative-reasoning examples across four task families, with entity names
  71    and sentence templates both held out across splits. Answer-token loss only.
```

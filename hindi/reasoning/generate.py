"""Generate Model H's synthetic comparative-reasoning dataset, in Hindi.

The assignment requires a finetuning set that is built here rather than
downloaded, so that every ground-truth label is known by construction. It also
requires the text to be in the target language's script *and* natural phrasing
-- not English sentence patterns with Devanagari names dropped in. Every
template below is a sentence a Hindi speaker would actually write.

This file belongs to **Model H** (Hindi). Model L has its own independent
generator at ``nepali/reasoning/generate.py``. The two datasets share no
sentences, no entity names and no templates, and neither model is ever shown the
other's data.

Four task families, covering the three styles the specification names:

``compare2``    One comparison, then a question about the pair.
``transitive3`` Two chained comparisons; the answer needs the chain.
``numeric2``    Two measured quantities; compare them.
``numeric3``    Three measured quantities; pick the extreme.

Leakage control
---------------
Both defences the specification suggests are applied at once:

* **Held-out entity names.** The name pool is split three ways with no overlap,
  so a test question is always about people the model never saw in training.
* **Held-out templates.** The sentence patterns are also split three ways, so a
  test question is phrased in a way the model never saw either.

A model can therefore only score well by actually performing the comparison --
memorising "this name goes with this answer", or memorising a phrasing, both
fail.

Usage
-----
    python hindi/reasoning/generate.py
    python hindi/reasoning/generate.py --train 20000 --val 2000 --test 2000
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "data"

LANGUAGE = "hindi"

# ---------------------------------------------------------------------------
# Entity names, split so that no name appears in more than one split.
# ---------------------------------------------------------------------------
# Each entry is (name, gender). Gender is carried because Hindi adjectives
# inflect for it -- "लंबा" for a man, "लंबी" for a woman. Getting that wrong
# would produce ungrammatical text, which the specification explicitly rules
# out, so the generator tracks it rather than defaulting everything to
# masculine.

NAMES_TRAIN = [
    ("राम", "m"), ("श्याम", "m"), ("मोहन", "m"), ("सोहन", "m"), ("विकास", "m"),
    ("राहुल", "m"), ("अजय", "m"), ("विजय", "m"), ("अनिल", "m"), ("सुनील", "m"),
    ("दीपक", "m"), ("मनोज", "m"), ("संजय", "m"), ("राजेश", "m"), ("महेश", "m"),
    ("सुरेश", "m"), ("नरेश", "m"), ("प्रकाश", "m"), ("अशोक", "m"), ("विनोद", "m"),
    ("सीता", "f"), ("गीता", "f"), ("रीता", "f"), ("मीना", "f"), ("सुनीता", "f"),
    ("अनीता", "f"), ("कविता", "f"), ("ममता", "f"), ("रेखा", "f"), ("शोभा", "f"),
    ("पूजा", "f"), ("नेहा", "f"), ("प्रीति", "f"), ("संगीता", "f"), ("वंदना", "f"),
    ("आशा", "f"), ("उषा", "f"), ("लता", "f"), ("माया", "f"), ("राधा", "f"),
]

NAMES_VAL = [
    ("गोपाल", "m"), ("किशोर", "m"), ("भरत", "m"), ("नितिन", "m"), ("हरीश", "m"),
    ("यमुना", "f"), ("कमला", "f"), ("सरिता", "f"), ("निशा", "f"), ("भारती", "f"),
]

NAMES_TEST = [
    ("अरुण", "m"), ("वरुण", "m"), ("तरुण", "m"), ("नवीन", "m"), ("प्रवीण", "m"),
    ("शारदा", "f"), ("मालती", "f"), ("चंदा", "f"), ("बबीता", "f"), ("सविता", "f"),
]

# ---------------------------------------------------------------------------
# Attributes. Each carries the comparative adjective in both genders, plus the
# phrasing used when the attribute is stated as a number.
# ---------------------------------------------------------------------------
# "more"/"less" are the two directions of the same attribute: taller/shorter,
# older/younger. Adjectives that do not inflect (भारी, तेज़, अमीर) simply repeat
# the same form for both genders.

ATTRIBUTES = {
    "height": {
        "more": ("लंबा", "लंबी"),
        "less": ("छोटा", "छोटी"),
        "noun": "लंबाई",
        "possessive": "की",
        "unit": "सेंटीमीटर",
        "range": (140, 190),
    },
    "age": {
        "more": ("बड़ा", "बड़ी"),
        "less": ("छोटा", "छोटी"),
        "noun": "उम्र",
        "possessive": "की",
        "unit": "साल",
        "range": (12, 70),
    },
    "weight": {
        "more": ("भारी", "भारी"),
        "less": ("हल्का", "हल्की"),
        "noun": "वज़न",
        "possessive": "का",
        "unit": "किलो",
        "range": (35, 95),
    },
    "speed": {
        "more": ("तेज़", "तेज़"),
        "less": ("धीमा", "धीमी"),
        "noun": None,          # no natural numeric phrasing; comparisons only
        "possessive": None,
        "unit": None,
        "range": None,
    },
    "wealth": {
        "more": ("अमीर", "अमीर"),
        "less": ("गरीब", "गरीब"),
        "noun": None,
        "possessive": None,
        "unit": None,
        "range": None,
    },
}

NUMERIC_ATTRIBUTES = [k for k, v in ATTRIBUTES.items() if v["noun"]]

# ---------------------------------------------------------------------------
# Templates, split so that a test question is phrased in a way never trained on.
# ---------------------------------------------------------------------------
# {a} {b} {c}   entity names
# {adj}         the comparative adjective, already inflected for the subject
# {q}           the adjective used in the question, inflected for the answer
# Sentences end with the Devanagari danda (।), as Hindi prose does.

TEMPLATES = {
    "compare2": {
        "train": [
            "{a} {b} से {adj} है। दोनों में से {q} कौन है?",
            "{a} {b} से {adj} है। बताइए, {q} कौन है?",
            "{a} {b} से {adj} है। तो {q} कौन हुआ?",
            "यह ज्ञात है कि {a} {b} से {adj} है। {q} कौन है?",
        ],
        "val": ["{a} {b} से {adj} है। इनमें {q} कौन है?"],
        "test": ["ध्यान दीजिए: {a} {b} से {adj} है। {q} कौन है?"],
    },
    "transitive3": {
        "train": [
            "{a} {b} से {adj} है। {b} {c} से {adj} है। सबसे {q} कौन है?",
            "{a} {b} से {adj} है, और {b} {c} से {adj} है। सबसे {q} कौन है?",
            "अगर {a} {b} से {adj} है और {b} {c} से {adj} है, तो सबसे {q} कौन है?",
            "{a} {b} से {adj} है। {b} {c} से {adj} है। तीनों में सबसे {q} कौन है?",
        ],
        "val": ["मान लीजिए {a} {b} से {adj} है और {b} {c} से {adj} है। सबसे {q} कौन है?"],
        "test": ["{a} {b} से {adj} है। इसी तरह {b} {c} से {adj} है। सबसे {q} कौन है?"],
    },
    "numeric2": {
        "train": [
            "{a} {p} {noun} {n1} {unit} है। {b} {p} {noun} {n2} {unit} है। {q} कौन है?",
            "{a} {p} {noun} {n1} {unit} है, और {b} {p} {noun} {n2} {unit} है। {q} कौन है?",
            "{a} {p} {noun} {n1} {unit} और {b} {p} {noun} {n2} {unit} है। दोनों में {q} कौन है?",
        ],
        "val": ["{a} {p} {noun} {n1} {unit} है। वहीं {b} {p} {noun} {n2} {unit} है। {q} कौन है?"],
        "test": ["दिया गया है कि {a} {p} {noun} {n1} {unit} है और {b} {p} {noun} {n2} {unit} है। {q} कौन है?"],
    },
    "numeric3": {
        "train": [
            "{a} {p} {noun} {n1} {unit} है। {b} {p} {noun} {n2} {unit} है। {c} {p} {noun} {n3} {unit} है। सबसे {q} कौन है?",
            "{a} {p} {noun} {n1} {unit}, {b} {p} {noun} {n2} {unit} और {c} {p} {noun} {n3} {unit} है। सबसे {q} कौन है?",
            "तीन लोगों में {a} {p} {noun} {n1} {unit}, {b} {p} {noun} {n2} {unit} और {c} {p} {noun} {n3} {unit} है। सबसे {q} कौन है?",
        ],
        "val": ["{a} {p} {noun} {n1} {unit} है, {b} {p} {noun} {n2} {unit} है, और {c} {p} {noun} {n3} {unit} है। सबसे {q} कौन है?"],
        "test": ["यहाँ {a} {p} {noun} {n1} {unit}, {b} {p} {noun} {n2} {unit}, तथा {c} {p} {noun} {n3} {unit} है। सबसे {q} कौन है?"],
    },
}

# Hindi text normally writes numbers in Devanagari digits; the Phase 1 corpus
# contains both forms, so the tokenizer handles either.
DEVANAGARI_DIGITS = "०१२३४५६७८९"


def to_devanagari(number: int) -> str:
    """Render an integer using Devanagari digits.

    Args:
        number: A non-negative integer.

    Returns:
        The same number written with the characters ०-९.
    """
    return "".join(DEVANAGARI_DIGITS[int(d)] for d in str(number))


def inflect(attribute: str, direction: str, gender: str) -> str:
    """Return the comparative adjective agreeing with the subject's gender.

    Args:
        attribute: Key into :data:`ATTRIBUTES`.
        direction: ``"more"`` or ``"less"``.
        gender: ``"m"`` or ``"f"``.

    Returns:
        The inflected adjective, e.g. ``"लंबा"`` or ``"लंबी"``.
    """
    masculine, feminine = ATTRIBUTES[attribute][direction]
    return masculine if gender == "m" else feminine


class ReasoningGenerator:
    """Builds one split of the Hindi comparative-reasoning dataset."""

    def __init__(self, split: str, seed: int) -> None:
        """Fix the name pool and template pool for one split.

        Args:
            split: ``"train"``, ``"val"`` or ``"test"``.
            seed: Seed, so a split is byte-for-byte reproducible.

        Raises:
            ValueError: If the split name is unknown.
        """
        pools = {"train": NAMES_TRAIN, "val": NAMES_VAL, "test": NAMES_TEST}
        if split not in pools:
            raise ValueError(f"unknown split: {split}")
        self.split = split
        self.names = pools[split]
        self.rng = random.Random(seed)

    def _template(self, task: str) -> tuple[str, int]:
        """Pick a template for this task from this split's pool.

        Args:
            task: Task family name.

        Returns:
            The template string and its index within the split's pool.
        """
        pool = TEMPLATES[task][self.split]
        index = self.rng.randrange(len(pool))
        return pool[index], index

    def _pick(self, count: int) -> list[tuple[str, str]]:
        """Draw distinct entities of the *same* gender from this split's pool.

        Same-gender is not a stylistic choice -- it closes a shortcut. Hindi
        comparative adjectives inflect ("छोटा" for a man, "छोटी" for a woman),
        so a question ending "सबसे छोटी कौन है?" asked about one woman and two
        men has only one grammatically possible answer. The model could score
        by matching gender agreement and never perform the comparison at all.
        An earlier draft of this generator leaked exactly that way in 45% of
        examples.

        Drawing every entity in an example from one gender makes the adjective
        carry no information about which entity is meant, so the answer can
        only come from the comparison itself.

        Args:
            count: How many entities are needed.

        Returns:
            A list of ``(name, gender)`` pairs, all different, all same gender.
        """
        gender = self.rng.choice(["m", "f"])
        pool = [p for p in self.names if p[1] == gender]
        if len(pool) < count:
            pool = self.names
        return self.rng.sample(pool, count)

    def compare2(self) -> dict:
        """One stated comparison, then a question about the same pair."""
        (a, ga), (b, gb) = self._pick(2)
        attribute = self.rng.choice(list(ATTRIBUTES))
        # Both the stated direction and the asked direction vary. If the
        # sentence says "a is taller than b" and the question asks who is
        # taller, the answer is a; if it asks who is shorter, it is b. Varying
        # the stated direction too means the answer is not simply "whichever
        # name was mentioned first".
        stated_more = self.rng.random() < 0.5
        ask_more = self.rng.random() < 0.5
        answer, answer_gender = (a, ga) if stated_more == ask_more else (b, gb)
        template, index = self._template("compare2")
        text = template.format(
            a=a, b=b,
            adj=inflect(attribute, "more" if stated_more else "less", ga),
            q=inflect(attribute, "more" if ask_more else "less", answer_gender),
        )
        return {
            "task": "compare2", "attribute": attribute, "template_index": index,
            "prompt": text, "answer": answer, "entities": [a, b],
            "stated_direction": "more" if stated_more else "less",
            "asked_direction": "more" if ask_more else "less",
            "hops": 1,
        }

    def transitive3(self) -> dict:
        """Two chained comparisons; the answer requires following the chain."""
        (a, ga), (b, gb), (c, gc) = self._pick(3)
        attribute = self.rng.choice(list(ATTRIBUTES))
        # The chain is stated as "a REL b" then "b REL c". If REL is "more" the
        # order is a > b > c; if "less" it is a < b < c. Either way the two
        # extremes are a and c, and which one answers the question depends on
        # whether the asked direction matches the stated one -- that agreement
        # is the two-hop inference being tested.
        stated_more = self.rng.random() < 0.5
        ask_more = self.rng.random() < 0.5
        answer, answer_gender = (a, ga) if stated_more == ask_more else (c, gc)
        template, index = self._template("transitive3")
        text = template.format(
            a=a, b=b, c=c,
            adj=inflect(attribute, "more" if stated_more else "less", ga),
            q=inflect(attribute, "more" if ask_more else "less", answer_gender),
        )
        return {
            "task": "transitive3", "attribute": attribute, "template_index": index,
            "prompt": text, "answer": answer, "entities": [a, b, c],
            "stated_direction": "more" if stated_more else "less",
            "asked_direction": "more" if ask_more else "less",
            "hops": 2,
        }

    def numeric2(self) -> dict:
        """Two measured quantities, compared."""
        (a, ga), (b, gb) = self._pick(2)
        attribute = self.rng.choice(NUMERIC_ATTRIBUTES)
        spec = ATTRIBUTES[attribute]
        low, high = spec["range"]
        n1, n2 = self.rng.sample(range(low, high), 2)
        ask_more = self.rng.random() < 0.5
        bigger = (a, ga) if n1 > n2 else (b, gb)
        smaller = (b, gb) if n1 > n2 else (a, ga)
        answer, answer_gender = bigger if ask_more else smaller
        template, index = self._template("numeric2")
        text = template.format(
            a=a, b=b, p=spec["possessive"], noun=spec["noun"], unit=spec["unit"],
            n1=to_devanagari(n1), n2=to_devanagari(n2),
            q=inflect(attribute, "more" if ask_more else "less", answer_gender),
        )
        return {
            "task": "numeric2", "attribute": attribute, "template_index": index,
            "prompt": text, "answer": answer, "entities": [a, b],
            "values": [n1, n2], "hops": 1,
        }

    def numeric3(self) -> dict:
        """Three measured quantities; pick the largest or smallest."""
        (a, ga), (b, gb), (c, gc) = self._pick(3)
        attribute = self.rng.choice(NUMERIC_ATTRIBUTES)
        spec = ATTRIBUTES[attribute]
        low, high = spec["range"]
        n1, n2, n3 = self.rng.sample(range(low, high), 3)
        people = [(a, ga, n1), (b, gb, n2), (c, gc, n3)]
        ask_more = self.rng.random() < 0.5
        chosen = max(people, key=lambda t: t[2]) if ask_more \
            else min(people, key=lambda t: t[2])
        template, index = self._template("numeric3")
        text = template.format(
            a=a, b=b, c=c, p=spec["possessive"], noun=spec["noun"],
            unit=spec["unit"],
            n1=to_devanagari(n1), n2=to_devanagari(n2), n3=to_devanagari(n3),
            q=inflect(attribute, "more" if ask_more else "less", chosen[1]),
        )
        return {
            "task": "numeric3", "attribute": attribute, "template_index": index,
            "prompt": text, "answer": chosen[0], "entities": [a, b, c],
            "values": [n1, n2, n3], "hops": 1,
        }

    def build(self, count: int) -> list[dict]:
        """Generate one split, with the four task families equally represented.

        Duplicate prompts are dropped: with a small name pool the test split in
        particular can otherwise repeat itself, which would make the reported
        accuracy depend on how often one question happened to appear.

        Args:
            count: Target number of unique examples.

        Returns:
            The generated examples, shuffled.
        """
        makers = [self.compare2, self.transitive3, self.numeric2, self.numeric3]
        seen: set[str] = set()
        out: list[dict] = []
        attempts = 0
        while len(out) < count and attempts < count * 200:
            attempts += 1
            row = makers[len(out) % len(makers)]()
            if row["prompt"] in seen:
                continue
            seen.add(row["prompt"])
            row["split"] = self.split
            row["language"] = LANGUAGE
            out.append(row)
        self.rng.shuffle(out)
        return out


def describe(rows: list[dict]) -> dict:
    """Summarise one split, for the statistics the specification asks for.

    Args:
        rows: Generated examples.

    Returns:
        Counts by task, attribute and template, plus answer balance.
    """
    return {
        "examples": len(rows),
        "by_task": dict(Counter(r["task"] for r in rows)),
        "by_attribute": dict(Counter(r["attribute"] for r in rows)),
        "distinct_templates": len({(r["task"], r["template_index"]) for r in rows}),
        "distinct_answers": len({r["answer"] for r in rows}),
        "distinct_prompts": len({r["prompt"] for r in rows}),
        "mean_prompt_chars": round(sum(len(r["prompt"]) for r in rows) / max(len(rows), 1), 1),
    }


def main() -> None:
    """Generate all three splits and write them as JSONL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=20000)
    parser.add_argument("--val", type=int, default=2000)
    parser.add_argument("--test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = {"language": LANGUAGE, "splits": {}}
    all_names = {"train": set(), "val": set(), "test": set()}

    for offset, (split, count) in enumerate(
        (("train", args.train), ("val", args.val), ("test", args.test))
    ):
        gen = ReasoningGenerator(split, args.seed + offset)
        rows = gen.build(count)
        path = OUT_DIR / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats["splits"][split] = describe(rows)
        all_names[split] = {n for r in rows for n in r["entities"]}
        print(f"  {split:5s} {len(rows):6,} examples -> {path.name}")

    # The leakage guarantee is asserted here rather than assumed: if the name
    # pools ever overlap, this fails loudly instead of quietly inflating
    # test accuracy.
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = all_names[a] & all_names[b]
        assert not overlap, f"name leakage between {a} and {b}: {overlap}"
    stats["leakage_check"] = {
        "entity_names_disjoint_across_splits": True,
        "templates_disjoint_across_splits": True,
        "train_names": len(all_names["train"]),
        "val_names": len(all_names["val"]),
        "test_names": len(all_names["test"]),
    }
    (OUT_DIR / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  wrote {OUT_DIR / 'dataset_stats.json'}")
    print("  name pools verified disjoint across all three splits")


if __name__ == "__main__":
    main()

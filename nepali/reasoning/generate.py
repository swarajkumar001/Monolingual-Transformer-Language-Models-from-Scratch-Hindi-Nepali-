"""Generate Model L's synthetic comparative-reasoning dataset, in Nepali.

The assignment requires a finetuning set that is built here rather than
downloaded, so that every ground-truth label is known by construction. It also
requires the text to be in the target language's script *and* natural phrasing
-- not English sentence patterns with Devanagari names dropped in. Every
template below is a sentence a Nepali speaker would actually write.

This file belongs to **Model L** (Nepali). Model H has its own independent
generator at ``hindi/reasoning/generate.py``. The two datasets share no
sentences, no entity names and no templates, and neither model is ever shown
the other's data -- the specification forbids sharing finetuning data or mixing
languages, so the overlap here is deliberately zero rather than merely small.

Four task families, covering the three styles the specification names:

``compare2``    One comparison, then a question about the pair.
``transitive3`` Two chained comparisons; the answer needs the chain.
``numeric2``    Two measured quantities; compare them.
``numeric3``    Three measured quantities; pick the extreme.

Nepali orthography
------------------
Two conventions matter for naturalness and are followed throughout:

* the comparative postposition ``भन्दा`` is written joined to the noun it
  governs -- ``श्यामभन्दा``, not ``श्याम भन्दा``;
* the genitive ``को`` is likewise joined -- ``रामको उचाइ``.

Writing them detached is the single most common way generated Nepali reads as
translated-from-Hindi, so the templates attach them.

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
    python nepali/reasoning/generate.py
    python nepali/reasoning/generate.py --train 20000 --val 2000 --test 2000
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "data"

LANGUAGE = "nepali"

# ---------------------------------------------------------------------------
# Entity names, split so that no name appears in more than one split.
# ---------------------------------------------------------------------------
# Each entry is (name, gender). Nepali inflects a subset of its adjectives for
# gender -- अग्लो/अग्ली, जेठो/जेठी -- while others (गह्रौं, धनी) are invariant.
# The generator tracks gender so the inflecting ones agree with their subject.
#
# These names are all distinct from Model H's Hindi pools. Nothing is shared
# between the two finetuning corpora.

NAMES_TRAIN = [
    ("बिक्रम", "m"), ("प्रदीप", "m"), ("सन्दीप", "m"), ("नरेन्द्र", "m"),
    ("राजन", "m"), ("सागर", "m"), ("नीरज", "m"), ("बिनय", "m"),
    ("सुमन", "m"), ("कृष्ण", "m"), ("हरि", "m"), ("गोविन्द", "m"),
    ("केशव", "m"), ("प्रेम", "m"), ("रमेश", "m"), ("बसन्त", "m"),
    ("शंकर", "m"), ("मिलन", "m"), ("दिपेन्द्र", "m"), ("भीम", "m"),
    ("सरस्वती", "f"), ("लक्ष्मी", "f"), ("पार्वती", "f"), ("गंगा", "f"),
    ("अञ्जु", "f"), ("बिन्दु", "f"), ("सपना", "f"), ("मञ्जु", "f"),
    ("सिर्जना", "f"), ("प्रतिमा", "f"), ("अनुपा", "f"), ("दुर्गा", "f"),
    ("राधिका", "f"), ("निर्मला", "f"), ("कोपिला", "f"), ("फूलमाया", "f"),
    ("मनमाया", "f"), ("समिक्षा", "f"), ("सुस्मिता", "f"), ("बिमला", "f"),
]

NAMES_VAL = [
    ("धनराज", "m"), ("तिलक", "m"), ("युवराज", "m"), ("नवराज", "m"),
    ("इन्द्र", "m"),
    ("सञ्जु", "f"), ("रोशनी", "f"), ("अमृता", "f"), ("यशोदा", "f"),
    ("तारा", "f"),
]

NAMES_TEST = [
    ("भोजराज", "m"), ("कुमार", "m"), ("दीपेश", "m"), ("सुबास", "m"),
    ("रबिन", "m"),
    ("निरु", "f"), ("कल्पना", "f"), ("सञ्जिता", "f"), ("उर्मिला", "f"),
    ("पुष्पा", "f"),
]

# ---------------------------------------------------------------------------
# Attributes. Each carries the comparative adjective in both genders, plus the
# phrasing used when the attribute is stated as a number.
# ---------------------------------------------------------------------------
# "more"/"less" are the two directions of the same attribute: taller/shorter,
# older/younger. Adjectives that do not inflect (गह्रौं, हलुका, धनी, गरिब)
# simply repeat the same form for both genders.

ATTRIBUTES = {
    "height": {
        "more": ("अग्लो", "अग्ली"),
        "less": ("होचो", "होची"),
        "noun": "उचाइ",
        "possessive": "को",
        "unit": "सेन्टिमिटर",
        "range": (140, 190),
    },
    "age": {
        "more": ("जेठो", "जेठी"),
        "less": ("कान्छो", "कान्छी"),
        "noun": "उमेर",
        "possessive": "को",
        "unit": "वर्ष",
        "range": (12, 70),
    },
    "weight": {
        "more": ("गह्रौं", "गह्रौं"),
        "less": ("हलुका", "हलुका"),
        "noun": "तौल",
        "possessive": "को",
        "unit": "किलो",
        "range": (35, 95),
    },
    "speed": {
        "more": ("छिटो", "छिटो"),
        "less": ("ढिलो", "ढिलो"),
        "noun": None,          # no natural numeric phrasing; comparisons only
        "possessive": None,
        "unit": None,
        "range": None,
    },
    "wealth": {
        "more": ("धनी", "धनी"),
        "less": ("गरिब", "गरिब"),
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
# Note {b}भन्दा and {a}{p}: the postposition and the genitive attach to the
# preceding word, which is how Nepali is written.
# Sentences end with the danda (।), as Nepali prose does.

TEMPLATES = {
    "compare2": {
        "train": [
            "{a} {b}भन्दा {adj} छ। दुईमध्ये {q} को हो?",
            "{a} {b}भन्दा {adj} छ। भन्नुहोस्, {q} को हो?",
            "{a} {b}भन्दा {adj} छ। त {q} को भयो?",
            "यो थाहा छ कि {a} {b}भन्दा {adj} छ। {q} को हो?",
        ],
        "val": ["{a} {b}भन्दा {adj} छ। यीमध्ये {q} को हो?"],
        "test": ["ध्यान दिनुहोस्: {a} {b}भन्दा {adj} छ। {q} को हो?"],
    },
    "transitive3": {
        "train": [
            "{a} {b}भन्दा {adj} छ। {b} {c}भन्दा {adj} छ। सबैभन्दा {q} को हो?",
            "{a} {b}भन्दा {adj} छ, र {b} {c}भन्दा {adj} छ। सबैभन्दा {q} को हो?",
            "यदि {a} {b}भन्दा {adj} छ र {b} {c}भन्दा {adj} छ भने, सबैभन्दा {q} को हो?",
            "{a} {b}भन्दा {adj} छ। {b} {c}भन्दा {adj} छ। तीनैजनामध्ये सबैभन्दा {q} को हो?",
        ],
        "val": ["मानौं {a} {b}भन्दा {adj} छ र {b} {c}भन्दा {adj} छ। सबैभन्दा {q} को हो?"],
        "test": ["{a} {b}भन्दा {adj} छ। त्यसै गरी {b} {c}भन्दा {adj} छ। सबैभन्दा {q} को हो?"],
    },
    "numeric2": {
        "train": [
            "{a}{p} {noun} {n1} {unit} छ। {b}{p} {noun} {n2} {unit} छ। {q} को हो?",
            "{a}{p} {noun} {n1} {unit} छ, र {b}{p} {noun} {n2} {unit} छ। {q} को हो?",
            "{a}{p} {noun} {n1} {unit} र {b}{p} {noun} {n2} {unit} छ। दुईमध्ये {q} को हो?",
        ],
        "val": ["{a}{p} {noun} {n1} {unit} छ। त्यस्तै {b}{p} {noun} {n2} {unit} छ। {q} को हो?"],
        "test": ["दिइएको छ कि {a}{p} {noun} {n1} {unit} र {b}{p} {noun} {n2} {unit} छ। {q} को हो?"],
    },
    "numeric3": {
        "train": [
            "{a}{p} {noun} {n1} {unit} छ। {b}{p} {noun} {n2} {unit} छ। {c}{p} {noun} {n3} {unit} छ। सबैभन्दा {q} को हो?",
            "{a}{p} {noun} {n1} {unit}, {b}{p} {noun} {n2} {unit} र {c}{p} {noun} {n3} {unit} छ। सबैभन्दा {q} को हो?",
            "तीन जनामध्ये {a}{p} {noun} {n1} {unit}, {b}{p} {noun} {n2} {unit} र {c}{p} {noun} {n3} {unit} छ। सबैभन्दा {q} को हो?",
        ],
        "val": ["{a}{p} {noun} {n1} {unit} छ, {b}{p} {noun} {n2} {unit} छ, र {c}{p} {noun} {n3} {unit} छ। सबैभन्दा {q} को हो?"],
        "test": ["यहाँ {a}{p} {noun} {n1} {unit}, {b}{p} {noun} {n2} {unit}, तथा {c}{p} {noun} {n3} {unit} छ। सबैभन्दा {q} को हो?"],
    },
}

# Numerals: Arabic digits, and the reason is measured rather than stylistic.
# ---------------------------------------------------------------------------
# Model L's tokenizer (BPE, 10k) contains *no Devanagari digit at all* -- not
# one of ०-९ is a vocabulary piece -- and it was trained without byte fallback.
# Every Devanagari numeral therefore encodes to <unk> and is unrecoverable: an
# earlier draft of this file used them and pushed the unknown-token rate of the
# finetuning corpus to 4.8%, which would have trained the model on holes where
# the quantities belong. Arabic digits are fully covered (0 unk over 12-190).
#
# Model H's tokenizer (Unigram, 10k) covers *both* systems at 0 unk, so Model H
# keeps Devanagari numerals. The asymmetry is not an inconsistency between the
# two generators -- it is the low-resource tokenizer's coverage gap showing up
# in practice, and it is reported as a finding rather than papered over.
#
# Arabic numerals are in any case what modern Nepali web prose overwhelmingly
# uses, which is why the Phase 1 corpus taught the tokenizer those and not the
# Devanagari forms. The phrasing stays natural.


def to_numeral(number: int) -> str:
    """Render an integer the way Model L's vocabulary can represent it.

    Uses Arabic digits. See the note above: Nepali Devanagari digits are absent
    from this model's vocabulary and would encode to ``<unk>``.

    Args:
        number: A non-negative integer.

    Returns:
        The number as a decimal string.
    """
    return str(number)


def inflect(attribute: str, direction: str, gender: str) -> str:
    """Return the comparative adjective agreeing with the subject's gender.

    Args:
        attribute: Key into :data:`ATTRIBUTES`.
        direction: ``"more"`` or ``"less"``.
        gender: ``"m"`` or ``"f"``.

    Returns:
        The inflected adjective, e.g. ``"अग्लो"`` or ``"अग्ली"``.
    """
    masculine, feminine = ATTRIBUTES[attribute][direction]
    return masculine if gender == "m" else feminine


class ReasoningGenerator:
    """Builds one split of the Nepali comparative-reasoning dataset."""

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

        Same-gender is not a stylistic choice -- it closes a shortcut. Several
        Nepali comparative adjectives inflect ("अग्लो" for a man, "अग्ली" for a
        woman), so a question ending "सबैभन्दा अग्ली को हो?" asked about one
        woman and two men has only one grammatically possible answer. The model
        could score by matching gender agreement and never perform the
        comparison at all. The Hindi generator leaked exactly that way in 45%
        of examples before the same fix was applied there.

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
            n1=to_numeral(n1), n2=to_numeral(n2),
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
            n1=to_numeral(n1), n2=to_numeral(n2), n3=to_numeral(n3),
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
    parser.add_argument("--seed", type=int, default=20260915)
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

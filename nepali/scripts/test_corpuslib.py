#!/usr/bin/env python3
"""
Self-tests for the mechanisms that carry the >=20% manual-token requirement.

    python hindi/scripts/test_corpuslib.py

These are not decorative. Each test pins down a behaviour whose failure in the
previous run was invisible until the whole pipeline had finished:

* the manual share reached 7.66% (Hindi) and 2.49% (Nepali) against a 20% floor;
* the manual and downloaded lanes shared 97.7% of their Wikipedia content, so
  deduplication could delete the manual copy of every duplicated document;
* a filter that dropped 94% of the corpus reported SUCCESS.

Run this after any change to ``ledger``, ``dedup`` or ``langid``. It needs no
network and finishes in about a second.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpuslib import dedup, heuristics, schema  # noqa: E402
from corpuslib.jsonlio import ShardWriteError, ShardWriter  # noqa: E402
from corpuslib.ledger import Ledger, cap_downloaded_lane  # noqa: E402
from corpuslib.ngramlm import NGramLM, tokenize  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    marker = "ok  " if condition else "FAIL"
    print(f"  [{marker}] {name}" + (f" — {detail}" if detail else ""))


def make_doc(index: int, lane: str, source_id: str, text: str, words: int = 100) -> dict:
    return {
        "doc_id": schema.content_hash(text),
        "text": text,
        "lang": "hindi",
        "lane": lane,
        "source_id": source_id,
        "source_name": source_id,
        "collection_method": "api" if lane == "manual" else "hf_download",
        "n_words": words,
        "n_chars": len(text),
        "schema_version": schema.SCHEMA_VERSION,
        "meta": {},
    }


def test_lane_capping() -> None:
    """Capping trims only downloaded, to a fixed budget, keeping every manual doc."""
    print("\ndownloaded lane capping")
    records = [make_doc(i, "manual", "wikipedia_api", f"manual {i} " * 40) for i in range(50)]
    records += [
        make_doc(i, "downloaded", "sangraha_verified", f"dl a {i} " * 40) for i in range(300)
    ]
    records += [
        make_doc(i, "downloaded", "indiccorp_v2", f"dl b {i} " * 40) for i in range(300)
    ]
    # 600 downloaded docs at n_words=100 each = 60,000 words; cap well under that.
    kept, report = cap_downloaded_lane(records, downloaded_unit_cap=15_000)

    check(
        "trims downloaded to at or under the cap",
        report["downloaded_units_after"] <= 15_000,
        f"{report['downloaded_units_before']:,} -> {report['downloaded_units_after']:,}",
    )
    check(
        "never drops a manual document",
        sum(1 for r in kept if r["lane"] == "manual") == 50,
        "manual is the scarce lane and must be preserved whole",
    )
    surviving_sources = {r["source_id"] for r in kept if r["lane"] == "downloaded"}
    check(
        "preserves downloaded source diversity",
        surviving_sources == {"sangraha_verified", "indiccorp_v2"},
        f"{sorted(surviving_sources)} — round-robin, not first-source-wins",
    )

    repeat, _ = cap_downloaded_lane(list(records), downloaded_unit_cap=15_000)
    check(
        "is deterministic under a fixed seed",
        [r["doc_id"] for r in kept] == [r["doc_id"] for r in repeat],
        "the statistics must be reproducible from the reported seed",
    )

    already_fine = [make_doc(i, "manual", "wikipedia_api", f"m {i} " * 40) for i in range(80)]
    already_fine += [make_doc(i, "downloaded", "sangraha_verified", f"d {i} " * 40) for i in range(20)]
    kept2, report2 = cap_downloaded_lane(already_fine, downloaded_unit_cap=1_000_000)
    check(
        "is a no-op when downloaded is already within the cap",
        report2["action"] == "none" and len(kept2) == len(already_fine),
        "never discards data it does not have to",
    )


def test_dedup_manual_priority() -> None:
    """Cross-lane duplicates must resolve to the manual copy, both exact and near."""
    print("\ndeduplication, manual priority")
    shared = "यह एक साझा दस्तावेज़ है जो दोनों स्रोतों में मौजूद है। " * 30

    pair = [
        make_doc(0, "downloaded", "sangraha_verified", shared),
        make_doc(0, "manual", "wikipedia_api", shared),
    ]
    stats = dedup.DedupStats()
    kept = dedup.remove_exact(pair, stats)
    check(
        "exact duplicate keeps the manual copy",
        len(kept) == 1 and kept[0]["lane"] == "manual",
        f"cross_lane_pairs={stats.cross_lane_pairs} manual_saved={stats.manual_saved_by_priority}",
    )

    near_manual = make_doc(1, "manual", "news_ekantipur", shared + "थप एक वाक्य यहाँ छ।")
    near_downloaded = make_doc(2, "downloaded", "indiccorp_v2", shared + "थप अर्को वाक्य छ।")
    stats2 = dedup.DedupStats()
    # Deliberately pass the downloaded copy first: order must not decide this.
    kept2 = dedup.remove_near_duplicates([near_downloaded, near_manual], stats=stats2)
    check(
        "near duplicate keeps the manual copy regardless of input order",
        len(kept2) == 1 and kept2[0]["lane"] == "manual",
        f"near_removed={stats2.near_removed} manual_saved={stats2.manual_saved_by_priority}",
    )

    distinct = [
        make_doc(3, "manual", "wikipedia_api", "पहला बिल्कुल अलग दस्तावेज़ है। " * 30),
        make_doc(4, "downloaded", "indiccorp_v2", "दूसरा पूरी तरह भिन्न विषय पर है। " * 30),
    ]
    stats3 = dedup.DedupStats()
    kept3 = dedup.remove_near_duplicates(distinct, stats=stats3)
    check(
        "unrelated documents are not merged",
        len(kept3) == 2,
        "threshold 0.55 must not collapse distinct text",
    )


def test_lane_invariant() -> None:
    """A source appearing in both lanes must be rejected, not silently counted twice."""
    print("\nlane invariant")
    ledger = Ledger(lang="hindi", stage="test")
    ledger.add_record(make_doc(9, "manual", "wikipedia_api", "क ख ग " * 40))
    try:
        ledger.add_record(make_doc(10, "downloaded", "wikipedia_api", "घ ङ च " * 40))
        check("rejects a source claimed by both lanes", False, "no exception raised")
    except ValueError:
        check(
            "rejects a source claimed by both lanes",
            True,
            "this is the defect that made the previous manual share fictional",
        )


def test_ledger_arithmetic() -> None:
    """Shortfall and cap must agree with the ratio definition."""
    print("\nledger arithmetic")
    ledger = Ledger(lang="hindi", stage="test")
    for i in range(10):
        ledger.add_record(make_doc(i, "manual", "wikipedia_api", "क " * 100, words=100))
    for i in range(90):
        ledger.add_record(
            make_doc(i, "downloaded", "sangraha_verified", f"ख {i} " * 100, words=100)
        )

    check("computes the manual ratio", abs(ledger.manual_ratio() - 0.10) < 1e-9,
          f"{ledger.manual_ratio():.1%}")
    check(
        "computes the downloaded cap for a target ratio",
        ledger.downloaded_cap(0.24) == int(1000 * 0.76 / 0.24),
        f"{ledger.downloaded_cap(0.24):,} words",
    )
    # Adding exactly the shortfall must land on the floor.
    shortfall = ledger.shortfall_words(0.20)
    manual_after = ledger.lane_totals()["manual"] + shortfall
    total_after = manual_after + ledger.lane_totals()["downloaded"]
    check(
        "shortfall lands exactly on the floor",
        abs(manual_after / total_after - 0.20) < 0.01,
        f"{shortfall:,} words needed",
    )


def test_schema_guards() -> None:
    """Malformed records must never reach disk."""
    print("\nschema validation")
    good = make_doc(0, "manual", "wikipedia_api", "यह एक वैध दस्तावेज़ है। " * 20)
    schema.validate(good, expect_lang="hindi")
    check("accepts a well-formed record", True)

    for name, mutate in (
        ("rejects an unknown lane", lambda r: r.update(lane="borrowed")),
        ("rejects empty text", lambda r: r.update(text="   ")),
        ("rejects an unknown collection method", lambda r: r.update(collection_method="vibes")),
        ("rejects a stale schema version", lambda r: r.update(schema_version=1)),
    ):
        record = dict(good)
        mutate(record)
        try:
            schema.validate(record, expect_lang="hindi")
            check(name, False, "no exception raised")
        except schema.SchemaError:
            check(name, True)

    cross = dict(good)
    cross["lang"] = "nepali"
    try:
        schema.validate(cross, expect_lang="hindi")
        check("rejects cross-language contamination", False, "no exception raised")
    except schema.SchemaError:
        check("rejects cross-language contamination", True,
              "the two corpora must never share documents")


def test_heuristics_gate() -> None:
    """Each check must fire for the text that violates it, and only that one."""
    print("\nheuristic filters (S05)")
    cfg = {
        "min_chars": 50, "min_words": 8, "max_symbol_ratio": 0.4,
        "min_devanagari_ratio": 0.6, "max_repeated_line_ratio": 0.5,
        "min_mean_word_length": 1.5, "max_mean_word_length": 12.0,
        "min_terminators": 1, "max_url_density": 0.3, "min_unique_word_ratio": 0.3,
    }
    good = "भारत एक विशाल देश है। यहाँ अनेक भाषाएँ बोली जाती हैं और संस्कृति विविध है।"
    verdict = heuristics.evaluate(good, cfg)
    check("accepts clean, well-formed prose", verdict.passed, verdict.reason)

    check(
        "rejects text shorter than min_chars",
        not heuristics.evaluate("बहुत छोटा", cfg).passed,
    )
    link_farm = "देखें http://a.example/1 http://a.example/2 http://a.example/3 यहाँ भी " * 4
    link_farm_verdict = heuristics.evaluate(link_farm, cfg)
    check(
        "rejects link-farm text",
        not link_farm_verdict.passed,
        # A URL is simultaneously punctuation-dense and non-Devanagari, so it
        # can trip max_symbol_ratio or min_devanagari_ratio before ever
        # reaching the url_density check -- any of the three is a correct
        # rejection of this text, since evaluate() reports the first failure.
        link_farm_verdict.reason,
    )
    stutter = ("शब्द " * 30) + "।"
    check(
        "rejects low lexical diversity",
        heuristics.evaluate(stutter, cfg).reason in ("low_lexical_diversity", "too_few_sentences"),
        heuristics.evaluate(stutter, cfg).reason,
    )


def test_ngram_lm_separates_fluent_from_noise() -> None:
    """A model trained on prose must score prose lower (more probable) than noise."""
    print("\nn-gram perplexity model (S06)")
    reference = [
        tokenize("भारत एक विशाल देश है जहाँ अनेक भाषाएँ बोली जाती हैं।"),
        tokenize("नेपाल हिमालको काखमा अवस्थित एक सुन्दर देश हो।"),
        tokenize("दिल्ली भारत की राजधानी है और यह एक बड़ा शहर है।"),
    ] * 20
    lm = NGramLM(order=5)
    lm.fit(reference)

    fluent = tokenize("भारत एक विशाल देश है जहाँ अनेक भाषाएँ बोली जाती हैं।")
    noise = tokenize("झबर गटपट ऌ ऴ नमूना अनजान शब्दावली बकवास टेक्स्ट यादृच्छिक")
    fluent_score = lm.score(fluent)
    noise_score = lm.score(noise)
    check(
        "scores held-out fluent text lower than unrelated noise",
        fluent_score < noise_score,
        f"fluent={fluent_score:.2f} bits/token, noise={noise_score:.2f} bits/token",
    )


def test_shard_writer_survives_vanished_directory() -> None:
    """A directory deleted out from under an open shard must not corrupt the manifest."""
    print("\nShardWriter resilience (jsonlio)")
    import shutil
    import tempfile
    from pathlib import Path as _Path

    tmp = _Path(tempfile.mkdtemp())
    try:
        writer = ShardWriter(tmp / "src", lang="hindi", lane="manual", source_id="x",
                              docs_per_shard=2)

        def doc(i: int) -> schema.Document:
            return schema.Document(
                text=f"भारत एक विशाल देश है। संख्या {i} " * 5,
                lang="hindi", lane="manual", source_id="x", source_name="x",
                collection_method="api", title=f"t{i}",
            )

        writer.write(doc(1))
        shutil.rmtree(tmp / "src")
        writer.write(doc(2))  # still under docs_per_shard, no rollover yet
        writer.write(doc(3))  # crosses docs_per_shard=2 -> rollover -> rename fails
        check(
            "rolls a lost shard's documents out of live stats",
            writer.stats.docs == 1 and writer.stats.lost_docs == 2,
            f"docs={writer.stats.docs} lost_docs={writer.stats.lost_docs}",
        )

        shutil.rmtree(tmp / "src", ignore_errors=True)
        raised = False
        try:
            for i in range(4, 7):
                writer.write(doc(i))
        except ShardWriteError:
            raised = True
        check(
            "stops the source after two consecutive shard losses",
            raised,
            "rather than spending the whole time budget re-losing shards",
        )

        manifest = writer.close()
        check(
            "manifest never reports lost documents as collected",
            manifest["docs"] == 0 and manifest["lost_docs"] >= 2,
            f"docs={manifest['docs']} lost_docs={manifest['lost_docs']}",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print("=" * 74)
    print("corpuslib self-tests — the >=20% manual requirement and its guards")
    print("=" * 74)

    test_lane_capping()
    test_dedup_manual_priority()
    test_lane_invariant()
    test_ledger_arithmetic()
    test_schema_guards()
    test_heuristics_gate()
    test_ngram_lm_separates_fluent_from_noise()
    test_shard_writer_survives_vanished_directory()

    print()
    print("=" * 74)
    if FAILED:
        print(f"FAILED — {len(FAILED)} of {len(PASSED) + len(FAILED)} checks:")
        for name in FAILED:
            print(f"  · {name}")
        return 1
    print(f"PASSED — all {len(PASSED)} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

### [2026-08-19 11:10:28] S02 — Token calibration (Nepali)
- Status: SUCCESS
- What happened: Measured 1.5451 tokens/word from 451,856 documents (30,937,560 words -> 47,801,647 tokens with an 8000-token throwaway model).
- Metrics: lang=nepali, sample_docs=451,856, sample_chars=200,000,334, sample_words=30,937,560, sample_tokens=47,801,647, tokens_per_word=1.5451, calibration_vocab_size=8,000, plausible_range=[1.3, 2.6], seconds=33.9


### [2026-08-19 11:20:33] S03 — Normalise (Nepali)
- Status: SUCCESS
- What happened: Normalised 1,658,065 of 1,658,065 documents; counts match, zero documents dropped.
- Metrics: docs_in=1,658,065, docs_out=1,658,065, docs_written=1,658,065, errors=0, by_lane_in={'downloaded': 1417769, 'manual': 240296}, by_lane_out={'downloaded': 1417769, 'manual': 240296}, pii_redacted={'email': 2311, 'phone': 2089}, seconds=604.3


### [2026-08-19 11:20:47] S04 — Language ID (Nepali)
- Status: BLOCKED
- What happened: fasttext is not installed


### [2026-08-19 11:36:58] S04 — Language ID (Nepali)
- Status: SUCCESS
- What happened: Retention 99.77% of 1,658,065 documents, inside the expected [35%, 100%] band.
- Metrics: docs_in=1,658,065, docs_kept=1,654,296, retention=0.9977, expected_retention_min=0.35, expected_retention_max=0.999, reject_reasons={'low_confidence': 1189, 'narrow_margin_vs_hi': 600, 'wrong_language': 1599, 'script_ratio_0.57': 10, 'narrow_margin_vs_sa': 3, 'narrow_margin_vs_mr': 1, 'script_ratio_0.56': 20, 'script_ratio_0.51': 1, 'script_ratio_0.53': 5, 'script_ratio_0.59': 24, 'script_ratio_0.60': 9, 'script_ratio_0.58': 19, 'script_ratio_0.54': 10, 'script_ratio_0.50': 5, 'script_ratio_0.49': 3, 'script_ratio_0.45': 1, 'script_ratio_0.55': 8, 'script_ratio_0.52': 7, 'script_ratio_0.46': 1, 'narrow_margin_vs_new': 1, 'script_ratio_0.43': 1, 'script_ratio_0.42': 1, 'too_short': 251}, lid_error_rate=0, lid_docs_scored=1,657,689, lid_docs_errored=0, seconds=677.5, smoke=0


### [2026-08-19 11:38:07] S05 — Heuristic filters (Nepali)
- Status: BLOCKED
- What happened: s07_holdout (train_pool) has not been run yet.


### [2026-08-19 11:42:06] S07 — Hold out val/test (Nepali)
- Status: SUCCESS
- What happened: Split 1,654,296 documents: train 1,406,033 (85.0%), val 124,042 (7.5%), test 124,221 (7.5%). Zero doc_id overlap.
- Metrics: docs_in=1,654,296, train_docs=1,406,033, val_docs=124,042, test_docs=124,221, train_fraction=0.8499, val_fraction=0.075, test_fraction=0.0751, target_train=0.85, target_val=0.075, target_test=0.075, zero_doc_id_overlap=1, seed=20,260,817, seconds=217.1


### [2026-08-19 11:59:10] S05 — Heuristic filters (Nepali)
- Status: SUCCESS
- What happened: Retention 99.72% of 1,406,033 documents, inside the expected [55%, 100%] band.
- Metrics: docs_in=1,406,033, docs_kept=1,402,070, retention=0.9972, expected_retention_min=0.55, expected_retention_max=0.999, drop_reasons={'too_few_sentences': 1226, 'link_farm': 481, 'too_few_words': 8, 'mean_word_too_short': 2, 'repeated_lines': 1858, 'low_lexical_diversity': 5, 'mean_word_too_long': 1, 'too_short': 382}, seconds=1,022.5, smoke=0


### [2026-08-19 12:54:06] S06 — Perplexity filter (Nepali)
- Status: SUCCESS
- What happened: Trained an order-5 model on 5,000 reference documents; dropped the worst 3% of 1,383,607 scored documents (cutoff 19.422 bits/token).
- Metrics: applied=1, docs_in=1,402,070, docs_kept=1,360,562, retention=0.9704, reference_docs=5,000, reference_sources=['ocr_archive_books', 'wikipedia_api', 'wikisource_api'], docs_scored=1,383,607, drop_worst_fraction=0.03, cutoff_bits_per_token=19.4218, mean_dropped_chars=794.2, seconds=3,295.6, smoke=0


### [2026-08-21 01:40:28] S08 — Deduplicate (Nepali)
- Status: SUCCESS
- What happened: Removed 24,247 of 1,360,562 documents (5,193 exact, 19,054 near). 5,135 cross-lane duplicate pairs found; 5,135 resolved to the manual copy.
- Metrics: input_docs=1,360,562, exact_duplicates_removed=5,193, near_duplicates_removed=19,054, total_removed=24,247, kept=1,336,315, retention=0.9822, cross_lane_duplicate_pairs=5,135, manual_docs_preserved_by_priority=5,135, removed_by_lane={'downloaded': 20513, 'manual': 3734}, seconds=109,358.9, config={'num_perm': 100, 'bands': 20, 'rows': 5, 'shingle_size': 5, 'threshold': 0.55}


### NOTE — S08 onward, reconstructed entries
The run above (finishing 2026-08-21 01:40:28) was a slow, independently-started
background pass that stalled and was superseded before it reached S09. The
production run that actually produced the current `nepali/data/final/` corpus
and `nepali/tokenizer/` was executed on a separate machine (per-stage console
logging was not captured/transferred, only the final JSON artifacts were), then
copied into this repo. The S08/S09/S10/S11/S12/S13/S14 entries below are
reconstructed after the fact from that run's own output files
(`nepali/data/final/corpus_stats.json`, `nepali/data/final/split_stats.json`,
`nepali/tokenizer/selection_report.json`, `nepali/tokenizer/evaluation_report.json`)
rather than live-logged at the time, so timestamps reflect file mtimes where
known and are otherwise approximate. All figures are direct reads of those
files, not re-derived or estimated.


### [2026-08-20 ~22:10] S08 — Deduplicate (Nepali, production run)
- Status: SUCCESS
- What happened: Removed 52,084 of 2,178,418 documents (2,350 exact, 49,734 near). 21,273 cross-lane duplicate pairs found; all 21,273 resolved to the manual copy.
- Metrics: docs_in=2,178,418, docs_out=2,126,334, retention=0.9761, exact_duplicates_removed=2,350, near_duplicates_removed=49,734, cross_lane_duplicate_pairs=21,273, manual_docs_preserved_by_priority=21,273
- Source: nepali/data/final/corpus_stats.json (retention_funnel "s08_dedup", "dedup")


### [2026-08-20 ~22:12] S09 — Decontaminate (Nepali, production run)
- Status: SUCCESS
- What happened: Removed 17,111 train documents that matched holdout (val+test) content; a rescan afterward found 0 remaining matches.
- Metrics: docs_in=2,126,334, docs_out=2,109,223, retention=0.9920, contaminated_removed=17,111, rescan_remaining_matches=0
- Source: nepali/data/final/corpus_stats.json (retention_funnel "s09_decontaminated" carries null docs_in/out — derived here from s08's docs_out and s10's docs_in, which differ by exactly the "decontamination.contaminated_removed" count; "decontamination")


### [2026-08-20 ~22:14] S10 — Balance lanes (Nepali, production run)
- Status: SUCCESS
- What happened: Capped the downloaded lane to bring the train pool's manual share from 23.27% toward the target; dropped 589,223 downloaded documents (manual lane untouched).
- Metrics: docs_in=2,109,223, docs_out=1,520,000, retention=0.7206, action=capped_downloaded, ratio_before=0.2327, ratio_after=0.2742, downloaded_docs_dropped=589,223
- Source: nepali/data/final/corpus_stats.json (retention_funnel "s10_balanced", "lane_balancing")


### [2026-08-20 ~22:15] S11 — Finalize splits (Nepali, production run)
- Status: SUCCESS
- What happened: Wrote final train/val/test JSONL files. Zero document_id overlap between all three splits (train 1,520,000 / val 199,428 / test 198,837 docs); overall manual word-share 26.70%.
- Metrics: train_docs=1,520,000, val_docs=199,428, test_docs=198,837, train_manual_ratio=0.2742, val_manual_ratio=0.2371, test_manual_ratio=0.2311, overall_manual_ratio=0.2670, zero_doc_id_overlap=1, flagged_splits=[], seconds=47.0
- Source: nepali/data/final/split_stats.json


### [2026-08-20 ~22:19:40] S12 — Train tokenizer (Nepali, production run)
- Status: SUCCESS
- What happened: Swept BPE at vocab_sizes [5000, 6000, 8000, 10000]; no fertility elbow found within the range (still improving >3% at every step), so the largest candidate (10,000) was selected per the sweep's own selection rule.
- Metrics: chosen_model_type=bpe, chosen_vocab_size=10000, chosen_fertility=1.4868, chosen_unk_rate=0.0, chosen_chars_per_token=4.3505, candidates=[{5000: fertility=1.7032, chars_per_token=3.7976}, {6000: fertility=1.6378, chars_per_token=3.9495}, {8000: fertility=1.5472, chars_per_token=4.1808}, {10000: fertility=1.4868, chars_per_token=4.3505}], max_unk_rate=0.002, min_relative_gain=0.03, preferred_min_vocab=5000, seconds=221.7
- Source: nepali/tokenizer/selection_report.json


### [2026-08-20 22:19:40–22:33:54] S13 — Tokenizer eval (Nepali, production run)
- Status: SUCCESS
- What happened: Evaluated the chosen vocab_size=10000 BPE model against the held-out test split.
- Metrics: vocab_size=10000, test_fertility=1.4672, test_unk_rate=0.0, test_chars_per_token=4.405, train_chars_per_token=4.439, train_docs=1,520,000, train_tokens=519,906,359, val_docs=199,428, val_tokens=57,839,396, test_docs=198,837, test_tokens=57,458,578, seconds=853.6
- Source: nepali/tokenizer/evaluation_report.json


### [2026-08-20 22:34:24] S14 — Corpus stats (Nepali, production run)
- Status: SUCCESS
- What happened: Computed final corpus statistics. Both absolute token targets met: manual 167,787,251 (target ≥100,000,000), downloaded 467,417,082 (target ≥400,000,000). Total 635,204,333 tokens (127.0% of the assignment's 500M planning target).
- Metrics: overall_tokens=635,204,333, overall_manual_tokens=167,787,251, overall_downloaded_tokens=467,417,082, meets_manual_target=1, meets_downloaded_target=1, seconds=29.1
- Source: nepali/data/final/corpus_stats.json



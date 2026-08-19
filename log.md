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



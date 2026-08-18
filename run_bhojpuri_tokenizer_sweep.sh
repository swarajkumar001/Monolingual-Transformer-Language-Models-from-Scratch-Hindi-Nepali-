#!/bin/bash
# Vocab range constrained to 5,000-10,000 per TA ruling: at ~25M total
# params, a larger vocab (e.g. 24k) makes the embedding table too large a
# share of the model, starving the actual Transformer layers. See
# report/phase1/tokenizer_analysis.md for the parameter-budget arithmetic.
set -e
for VS in 5000 6000 8000 10000; do
  if [ -f "bhojpuri/tokenizer/bhojpuri_unigram_${VS}.model" ]; then
    echo "=== vocab_size=$VS already trained, skipping (resume) ==="
  else
    echo "=== training vocab_size=$VS ==="
    python bhojpuri/tokenizer/train_tokenizer.py --vocab-size $VS --threads 4
  fi
  echo "=== analyzing vocab_size=$VS (validation sample) ==="
  python bhojpuri/tokenizer/analyze_tokenizer.py --vocab-size $VS
  echo "=== full-corpus token count vocab_size=$VS ==="
  python scripts/count_tokens.py --lang bhojpuri --tokenizer-model bhojpuri/tokenizer/bhojpuri_unigram_${VS}.model
done
echo "=== SWEEP COMPLETE ==="

#!/bin/bash
set -e
for VS in 8000 12000 16000 24000 32000; do
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

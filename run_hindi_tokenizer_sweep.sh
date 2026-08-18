#!/bin/bash
# Do not launch until split_hindi.py has finished (needs data/splits/{train,validation}/
# populated from the full non-resume clean_hindi.py run, not the stale Aug-17 split).
set -e
for VS in 8000 12000 16000 24000 32000; do
  if [ -f "hindi/tokenizer/hindi_unigram_${VS}.model" ]; then
    echo "=== vocab_size=$VS already trained, skipping (resume) ==="
  else
    echo "=== training vocab_size=$VS ==="
    python hindi/tokenizer/train_tokenizer.py --vocab-size $VS --threads 4
  fi
  echo "=== analyzing vocab_size=$VS (validation sample) ==="
  python hindi/tokenizer/analyze_tokenizer.py --vocab-size $VS
  echo "=== full-corpus token count vocab_size=$VS ==="
  python scripts/count_tokens.py --lang hindi --tokenizer-model hindi/tokenizer/hindi_unigram_${VS}.model
done
echo "=== SWEEP COMPLETE ==="

echo Starting Hindi 5000 at %TIME%
.venv\Scripts\python hindi\tokenizer\train_tokenizer.py --vocab-size 5000 --threads 6 > hindi_5000.log 2>&1
echo Starting Hindi 10000 at %TIME%
.venv\Scripts\python hindi\tokenizer\train_tokenizer.py --vocab-size 10000 --threads 6 > hindi_10000.log 2>&1
echo Done at %TIME%

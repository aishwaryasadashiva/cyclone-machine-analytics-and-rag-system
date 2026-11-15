#!/usr/bin/env bash
ORIG=$PWD; cd "$(dirname "$0")" || exit 1   # scripts live in checks/; rag_prototype.py is one folder up
# Runs the open-model path (MiniLM + FAISS + flan-t5-small) and saves the raw console output.
# Use inside the virtual environment where sentence-transformers, faiss-cpu, transformers and torch are installed.
#   cd Task2/prototype && source venv/bin/activate && bash run_open_models.sh
# It OVERWRITES run_log_open_models.txt and evaluation_open_models.csv. evaluation.csv (offline mode) is not touched.
set -u
LOG=run_log_open_models.txt
EMB=sentence-transformers/all-MiniLM-L6-v2
LLM=google/flan-t5-small
{
  echo "# raw log, $(date)"
  echo "# python: $(python --version 2>&1)"
  echo "# packages: $(pip freeze | grep -iE 'sentence|faiss|transformers|torch|huggingface' | tr '\n' ' ')"
  echo "# machine: $(uname -sm)"
  echo
  echo '=== --ask'
  python ../rag_prototype.py --embed "$EMB" --llm "$LLM" --ask "What does a sudden draft drop indicate?" 2>&1
  echo
  echo '=== --eval'
  python ../rag_prototype.py --embed "$EMB" --llm "$LLM" --eval --out evaluation_open_models.csv 2>&1
  echo
  echo '=== status counts in evaluation_open_models.csv'
  python - <<'PY'
import csv, collections
rows = list(csv.DictReader(open("evaluation_open_models.csv")))
print(collections.Counter(r["status"] for r in rows))
for r in rows:
    if r["status"] == "ungrounded_rejected": print("rejected:", r["question"])
PY
} 2>&1 | tee "$LOG"
echo
echo "Saved $LOG. Check that the [index] line reads: llm: $LLM (no 'failed to load' text)."

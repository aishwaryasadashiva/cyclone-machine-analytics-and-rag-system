#!/bin/bash
ORIG=$PWD; cd "$(dirname "$0")" || exit 1   # scripts live in checks/; rag_prototype.py is one folder up
# Runs the prototype on the assignment's real PDFs with the open models. Put ONLY the PDFs in a folder (no .txt/.md copies, they would be indexed as duplicates), e.g.
#   mkdir real_docs && unzip ~/Downloads/drive-download-*.zip -d real_docs
# Needs pip install cryptography fonttools (one PDF is AES-encrypted; without cryptography it is skipped with a warning).
# then, with the virtual environment active:  bash run_real_corpus.sh real_docs
D=${1:-$ORIG/real_docs}; case "$D" in /*) ;; *) D="$ORIG/$D";; esac; E=sentence-transformers/all-MiniLM-L6-v2; L=google/flan-t5-small
python ../rag_prototype.py --docs $D --embed $E --llm $L --eval --questions questions_real.csv --out evaluation_real_open_models.csv 2>&1 | tee run_log_real_open_models.txt
python ../rag_prototype.py --docs $D --embed $E --rerank --llm $L --eval --questions questions_real.csv --out evaluation_real_rerank_open_models.csv 2>&1 | tee run_log_real_rerank_open_models.txt
python ../rag_prototype.py --docs $D --embed $E --llm $L --gate-test gate_real.csv 2>&1 | tee gate_test_real_open_models.txt
python score_gate.py --docs $D --sets gate_real.csv --embed $E --rerank 2>&1 | tee score_gate_real_open_models.txt
python ../rag_prototype.py --docs $D --embed $E --llm $L --bench 2>&1 | tee benchmark_real_open_models.txt
echo "done: paste everything printed above back"

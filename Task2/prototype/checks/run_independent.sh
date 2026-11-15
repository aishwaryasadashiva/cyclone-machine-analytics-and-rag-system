#!/bin/bash
ORIG=$PWD; cd "$(dirname "$0")" || exit 1   # scripts live in checks/; rag_prototype.py is one folder up
# The independent question set (questions_independent.csv: 13 answerable + 4 unanswerable, written separately from the project's own question sets) on the real PDFs with the open models.
# Usage, with the virtual environment active:  bash run_independent.sh real_docs     (about 3 to 5 minutes)
D=${1:-$ORIG/real_docs}; case "$D" in /*) ;; *) D="$ORIG/$D";; esac; E=sentence-transformers/all-MiniLM-L6-v2; L=google/flan-t5-small
echo "=== 1. retrieval, MiniLM + BM25 hybrid"
python ../rag_prototype.py --docs $D --embed $E --llm $L --eval --questions questions_independent_eval.csv --out evaluation_independent_open_models.csv 2>&1 | tail -4
echo "=== 2. retrieval, with the cross-encoder reranker"
python ../rag_prototype.py --docs $D --embed $E --rerank --llm $L --eval --questions questions_independent_eval.csv --out evaluation_independent_rerank_open_models.csv 2>&1 | tail -4
echo "=== 3. gate (answerable answered / unanswerable refused), default"
python ../rag_prototype.py --docs $D --embed $E --llm $L --gate-test gate_independent.csv 2>&1 | tee gate_independent_open_models.txt | tail -10
echo "=== 4. answer sheet, default"
python ../rag_prototype.py --docs $D --embed $E --llm $L --questions questions_independent.csv --grading-sheet answer_grading_independent_open_models.csv 2>&1 | tail -1
echo "=== 5. gate with reranker + extract fallback + cross-encoder sentences"
python ../rag_prototype.py --docs $D --embed $E --rerank --llm $L --extract-fallback --ce-sentences --gate-test gate_independent.csv 2>&1 | tee gate_independent_fallback_open_models.txt | tail -10
echo "=== 6. answer sheet, with those options"
python ../rag_prototype.py --docs $D --embed $E --rerank --llm $L --extract-fallback --ce-sentences --questions questions_independent.csv --grading-sheet answer_grading_independent_open_models_fallback.csv 2>&1 | tail -1
zip -q answers_to_send.zip answer_grading_*open_models*.csv && echo "done: paste the output above and attach answers_to_send.zip (it holds every answer sheet)"

#!/bin/bash
ORIG=$PWD; cd "$(dirname "$0")" || exit 1   # scripts live in checks/; rag_prototype.py is one folder up
# Answer-quality run on the real PDFs with the open models (about 3 to 5 minutes). Writes answer sheets with the answer text, so they can be graded against the PDFs.
# Usage, with the virtual environment active:  bash run_answer_quality.sh real_docs
# Then send back the printed output AND the two answer_grading_real_open_models*.csv files.
D=${1:-$ORIG/real_docs}; case "$D" in /*) ;; *) D="$ORIG/$D";; esac; E=sentence-transformers/all-MiniLM-L6-v2; L=google/flan-t5-small
# A. as before: MiniLM + flan-t5-small, a rejected model answer becomes "I could not find this"
python ../rag_prototype.py --docs $D --embed $E --llm $L --questions questions_real.csv --grading-sheet answer_grading_real_open_models.csv 2>&1 | tail -3
python ../rag_prototype.py --docs $D --embed $E --llm $L --gate-test gate_real.csv 2>&1 | tee gate_test_real_base_open_models.txt | tail -12
# B. with the reranker, the extract fallback (needs a cross-encoder score of at least 5) and cross-encoder sentence choice
python ../rag_prototype.py --docs $D --embed $E --rerank --llm $L --extract-fallback --ce-sentences --questions questions_real.csv --grading-sheet answer_grading_real_open_models_fallback.csv 2>&1 | tail -3
python ../rag_prototype.py --docs $D --embed $E --rerank --llm $L --extract-fallback --ce-sentences --gate-test gate_real.csv 2>&1 | tee gate_test_real_fallback_open_models.txt | tail -12
# C. the same two variants on the synthetic documents, to check that the fallback does not let off-topic questions through
python ../rag_prototype.py --embed $E --rerank --llm $L --extract-fallback --ce-sentences --gate-test gate_heldout.csv 2>&1 | tee gate_test_synthetic_fallback_open_models.txt | tail -10
python ../rag_prototype.py --embed $E --rerank --llm $L --extract-fallback --ce-sentences --gate-test gate_heldout_v2.csv 2>&1 | tee -a gate_test_synthetic_fallback_open_models.txt | tail -10
echo "done: paste everything above and attach the two answer_grading_real_open_models*.csv files"

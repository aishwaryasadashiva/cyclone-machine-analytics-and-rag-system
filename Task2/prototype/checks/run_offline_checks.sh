#!/bin/bash
ORIG=$PWD; cd "$(dirname "$0")" || exit 1   # scripts live in checks/; rag_prototype.py is one folder up
# Offline checks (TF-IDF/LSA + BM25, extractive answers). No model download. Run from anywhere: bash Task2/prototype/checks/run_offline_checks.sh
# Writes offline_checks_log.txt. The graded answer sheet is regenerated with: python ../rag_prototype.py --grading-sheet answer_grading.csv (it overwrites the grades).
{
  echo "# offline checks, $(date), $(python3 --version 2>&1)"
  for a in "--eval" "--eval --questions ../questions_extra.csv --out evaluation_extra.csv" "--gate-test gate_heldout.csv" "--gate-test gate_heldout_v2.csv" \
           "--probes guardrail_probes_v2.csv" "--probes guardrail_probes_v3.csv" "--probes guardrail_probes_v4.csv" "--probes guardrail_probes_v5.csv" "--bench"; do
    echo; echo "=== python ../rag_prototype.py $a"; python3 ../rag_prototype.py $a 2>&1
  done
} | tee offline_checks_log.txt

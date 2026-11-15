#!/bin/bash
ORIG=$PWD; cd "$(dirname "$0")" || exit 1   # scripts live in checks/; rag_prototype.py is one folder up
# Runs every check on your machine and keeps the raw logs. Run it from this folder with your virtual environment active:
#   source .venv/bin/activate && bash run_all_checks.sh
# Needs: pip install numpy pypdf cryptography fonttools scikit-learn sentence-transformers faiss-cpu transformers torch
# The first run downloads MiniLM, flan-t5-small and the cross-encoder (a few minutes). It OVERWRITES the logs named below.
E=sentence-transformers/all-MiniLM-L6-v2; L=google/flan-t5-small
python ../rag_prototype.py --embed $E --llm $L --eval --questions ../questions_extra.csv --out evaluation_extra_open_models.csv 2>&1 | tee run_log_extra_open_models.txt
python ../rag_prototype.py --embed $E --llm $L --gate-test gate_heldout.csv 2>&1 | tee gate_test_open_models.txt
python ../rag_prototype.py --embed $E --llm $L --probes guardrail_probes_v3.csv 2>&1 | tee probes_v3_open_models.txt
python ../rag_prototype.py --embed $E --llm $L --probes guardrail_probes_v4.csv 2>&1 | tee probes_v4_open_models.txt
python ../rag_prototype.py --embed $E --llm $L --probes guardrail_probes_v5.csv 2>&1 | tee probes_v5_open_models.txt
python ../rag_prototype.py --embed $E --llm $L --gate-test gate_heldout_v2.csv 2>&1 | tee gate_test_v2_open_models.txt
python ../rag_prototype.py --embed $E --rerank --cutoff -6.3 --llm $L --gate-test gate_heldout.csv 2>&1 | tee gate_test_cutoff_open_models.txt
python ../rag_prototype.py --embed $E --rerank --cutoff -6.3 --llm $L --gate-test gate_heldout_v2.csv 2>&1 | tee -a gate_test_cutoff_open_models.txt
python ../rag_prototype.py --embed $E --rerank --cutoff -6.3 --llm $L --eval --questions ../questions_extra.csv --out evaluation_extra_cutoff_open_models.csv 2>&1 | tee run_log_extra_cutoff_open_models.txt
python score_gate.py --embed $E --rerank 2>&1 | tee score_gate_open_models.txt
python ../rag_prototype.py --embed $E --ablation 2>&1 | tee ablation_open_models.txt
python ../rag_prototype.py --embed $E --ablation --questions ../questions_extra.csv 2>&1 | tee -a ablation_open_models.txt
python ../rag_prototype.py --embed $E --rerank --ablation 2>&1 | tee ablation_rerank_open_models.txt
python ../rag_prototype.py --embed $E --rerank --ablation --questions ../questions_extra.csv 2>&1 | tee -a ablation_rerank_open_models.txt
python ../rag_prototype.py --embed $E --rerank --llm $L --eval --questions ../questions_extra.csv --out evaluation_extra_rerank_open_models.csv 2>&1 | tee run_log_extra_rerank_open_models.txt
python ../rag_prototype.py --embed $E --llm $L --bench 2>&1 | tee benchmark_open_models.txt
echo "done: paste the contents of the *_open_models.txt files and run_log_*.txt back to get the README and slides updated"

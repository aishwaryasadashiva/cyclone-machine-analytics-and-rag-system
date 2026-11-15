#!/bin/bash
ORIG=$PWD; cd "$(dirname "$0")" || exit 1   # scripts live in checks/; rag_prototype.py is one folder up
# One command for everything that is still to be measured on the Mac (about 8 to 10 minutes). With the virtual environment active:   bash run_final.sh real_docs
# It runs the independent question set and the answer-quality checks with the open models, then packs every answer sheet into answers_to_send.zip.
D=${1:-$ORIG/real_docs}; case "$D" in /*) ;; *) D="$ORIG/$D";; esac
bash run_independent.sh $D
bash run_answer_quality.sh $D
rm -f answers_to_send.zip; zip -q answers_to_send.zip answer_grading_*open_models*.csv gate_*open_models*.txt evaluation_independent*open_models.csv 2>/dev/null
echo; echo "ALL DONE. Paste the text printed above (from the top if you can) and attach answers_to_send.zip from this folder."

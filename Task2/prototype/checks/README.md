# checks/ — experiments behind the numbers

Everything here supports the results table in `../README.md`. None of it is needed to run the prototype.

| Item | What it is |
|---|---|
| `EXPERIMENT_LOG.md` | Full chronological write-up (sections 1-18): every evaluation, fix and caveat. File names inside it refer to this folder, or to `../` for `rag_prototype.py`, `docs/`, `evaluation.csv`, `questions_extra.csv` |
| `questions_*.csv`, `gate_*.csv`, `guardrail_probes_v*.csv` | Question, refusal-test and guardrail-probe sets (self-written; `questions_independent*` were supplied separately) |
| `evaluation_*.csv`, `answer_grading*.csv` | Per-question outputs and answer sheets (grades are by me) |
| `*_log.txt`, `benchmark_*.txt`, `run_log_*.txt`, `open_model_logs/` | Raw console output of the runs (Mac runs of 13-15 Nov are in `open_model_logs/`) |
| `run_*.sh` | Run scripts. They change into this folder first and call `../rag_prototype.py`; PDF folders default to `real_docs` in the folder you run from |
| `score_gate.py` | Prints cross-encoder scores for on/off-topic questions so a refusal cut-off can be chosen |

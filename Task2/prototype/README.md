# RAG prototype (free / open-source only)

Question answering over technical PDFs: ingest → chunk → hybrid retrieval → relevance gate → answer with `[file p.N]` citations, or a fixed refusal message. Default mode needs no model download; open models are optional.

## Setup
```bash
cd Task2/prototype
pip install -r requirements.txt        # pypdf, numpy, scikit-learn, reportlab, cryptography, fonttools
```
Offline results were produced with Python 3.12; the open-model runs were done on an Apple-silicon MacBook Air (Python 3.14.4).

## Run
**1. Offline mode (default, no downloads).** BM25 + TF-IDF/LSA vectors, merged by reciprocal-rank fusion; answers are quoted sentences.
```bash
python rag_prototype.py --ask "What does a sudden draft drop indicate?"
python rag_prototype.py --eval          # precision@4, recall@4, hit@1, MRR on 12 questions -> evaluation.csv
python rag_prototype.py                 # interactive prompt (blank line quits)
```

**2. Open-model mode (optional).** BM25 + MiniLM embeddings (FAISS) + flan-t5-small generation.
```bash
pip install sentence-transformers faiss-cpu transformers torch
python rag_prototype.py --embed sentence-transformers/all-MiniLM-L6-v2 --llm google/flan-t5-small --ask "What does a sudden draft drop indicate?"
```
Models download automatically on first use (MiniLM about 90 MB; the full open-model path is about 1 GB in total). The first printed line should read `dense: ...+FAISS | llm: google/flan-t5-small`; if it says `tfidf-lsa` or `extractive fallback`, a model did not load.

**3. Your own PDFs** (e.g. the assignment's Drive folder; put only PDFs in the folder):
```bash
mkdir real_docs                                  # copy the PDFs here
cp questions_template.csv my_questions.csv       # one row per question; gold_doc = PDF file name (| for several)
python rag_prototype.py --docs real_docs --eval --questions my_questions.csv --out evaluation_real_docs.csv
```

## Optional flags
| Flag | Effect |
|---|---|
| `--rerank` | Rescore the top 8 chunks with `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `--cutoff X` | With `--rerank`: refuse if the best cross-encoder score is below X (opt-in; no stable value was found) |
| `--extract-fallback`, `--ce-sentences` | With `--rerank` and an LLM: quote the manual when the small LLM's answer is rejected as ungrounded; pick sentences by cross-encoder |
| `--ocr` | Read scanned pages (needs `pip install pymupdf pytesseract` and the Tesseract program) |
| `--probes F`, `--gate-test F`, `--bench`, `--ablation`, `--grading-sheet F` | Guardrail probes, refusal tests, latency benchmark, retrieval ablation, answer sheet for manual grading |

## Files
| Path | Content |
|---|---|
| `rag_prototype.py` | The whole pipeline (single file) |
| `docs/` | 5 synthetic sample PDFs (cyclone manual, SOPs, troubleshooting, alarms, safety); re-create with `make_sample_docs.py` |
| `evaluation.csv` | Per-question retrieval metrics, offline mode |
| `questions_extra.csv`, `questions_template.csv` | 16 indirect questions; template for your own |
| `checks/` | All further experiments: question and probe sets, logs, answer sheets, run scripts, and `EXPERIMENT_LOG.md` (the full history) |

## Results and limits
The 11 real PDFs (857 chunks) are **not** in this folder.

| Check | Result |
|---|---|
| 12 synthetic questions, offline | hit@1 0.75, MRR 0.875, recall@4 1.00 |
| 12 synthetic questions, MiniLM + FAISS | hit@1 0.83, MRR 0.92 (1.00 / 1.00 with reranker) |
| 20 real-PDF questions, open models | hit@1 0.95, MRR 0.975 (1.00 / 1.00 with reranker) |
| 13 independent questions, real PDFs, open models | hit@1 0.846, MRR 0.885, recall@4 1.00; reranker lowered hit@1 to 0.769 |
| Answer correctness (graded by me against the PDFs, offline) | 12 of 20 real-PDF questions; 5 of 13 independent |
| Off-topic refusals, real PDFs | 8 of 8 by default, but **tuned**: rules were written after those questions were seen |

- Questions, labels and grades were written by me, not by an operator or an independent reader, so these figures are a smoke test, not a benchmark.
- "Answered" does not mean "correct"; open-model answer correctness was not graded.
- Not implemented (design only): NLI support check, document version handling, a trained safety classifier. The keyword guardrail is blunt: unknown phrasings pass.
- To rerun every check: `bash checks/run_offline_checks.sh` (offline), `bash checks/run_open_models.sh` (open models). Scripts that take PDFs use `real_docs` unless you pass a folder: `bash checks/run_final.sh real_docs`.

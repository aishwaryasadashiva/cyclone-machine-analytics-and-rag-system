# Task 2 — RAG system design notes

Diagram: `architecture_diagram.pptx`. Runnable prototype and exact run steps: `prototype/README.md`.

## 1. Architecture
Ingestion (PDF → text, OCR fallback for scans, metadata: doc, version, page, section) → chunking → embeddings → vector index + BM25 index → hybrid retrieval → cross-encoder rerank → LLM → guardrails → cited answer. A feedback and monitoring loop surrounds the whole pipeline.

## 2. Retrieval strategy
- **Chunking:** split on headings and numbered steps first, then about 200–300 tokens with about 15 % overlap; keep tables whole. Small enough for precise citations, large enough to keep a procedure together. The prototype uses about 90-word sentence windows with about 25 words of overlap because its sample documents are tiny.
- **Embeddings:** `BAAI/bge-small-en-v1.5` or `all-MiniLM-L6-v2` (free, CPU-friendly, 384-d); move to `bge-base` or `bge-large` if recall is poor.
- **Search:** BM25 plus dense vectors, merged by reciprocal-rank fusion (no score calibration needed). BM25 stays because the vocabulary is technical ("draft", "dip-leg"). Top 30 go to a cross-encoder reranker, and the top 4 go to the LLM.
- **Faithfulness:** the prompt restricts the model to the retrieved context; every sentence carries a citation; a post-check verifies each sentence against the retrieved text (NLI model or lexical overlap) and drops or refuses it otherwise; source snippets are shown to the user.

## 3. Guardrails and failure modes
| Failure | Mitigation |
|---|---|
| No relevant document | Relevance gate (top-hit score and query-term coverage) → fixed fallback message, no LLM call |
| Hallucination | Context-only prompt, enforced citations, groundedness check, low temperature |
| Sensitive or unsafe query (bypassing interlocks, credentials, prompt injection) | Input pattern filter plus a classifier in the target design; documents are treated as data, never as instructions; safety-critical answers quote the SOP verbatim and point to the safety officer |
| Stale or conflicting documents | Version metadata, prefer the latest, show the document date in the citation |
| PII or secrets in documents | Redact at ingestion; per-role access filter on retrieval |

**Monitoring.** Offline: precision@k, recall@k and MRR on a growing labelled question set built from operator logs. Online: fallback rate, thumbs up/down, citation-click rate, faithfulness score (NLI or LLM-judged on a sample), p95 latency, and audits of "refused although documents existed".

## 4. Scalability
- **10× documents (50 → 500+ PDFs, about 10⁵ chunks):** still small for one node. Move from FAISS-flat to HNSW or IVF-PQ, or to Qdrant/pgvector; use incremental ingestion keyed by document hash; add metadata filters (equipment, document type) to narrow the search.
- **100+ concurrent users:** stateless API behind a load balancer; cache frequent queries and embeddings. The LLM is the bottleneck: serve a quantised 7B model with vLLM or llama.cpp using continuous batching, queue and autoscale on queue depth, stream tokens.
- **Cost:** CPU-only retrieval tier; GPU only for generation, with scale-to-zero, serverless GPU or spot instances; try the small model first and escalate to a larger one only for low-confidence answers; cache answers to repeated questions.
- *These figures are estimates; only latency and index size on the 5 synthetic PDFs were measured (median search 0.7 ms, answer 2 ms offline).*

## 5. Trade-offs
Extractive answers are safest but less fluent; generative answers read better but need the groundedness check. Bigger chunks help procedures and hurt precision. A cross-encoder adds roughly 100–300 ms and is the cheapest way to raise precision, but on the independent question set it lowered hit@1 (0.846 → 0.769), so there is no solid evidence it helps on unseen questions.

## 6. Prototype vs target design
| Component | In the prototype | Target design only |
|---|---|---|
| Parsing / chunking | pypdf, sentence-aware chunks with page metadata; optional `--ocr` | Table- and heading-aware chunking |
| Embeddings / index | TF-IDF + LSA (offline); MiniLM + FAISS (open-model mode) | bge-small/base, HNSW or a managed vector store |
| Retrieval | BM25 + dense + RRF; optional cross-encoder rerank | Reranker on by default |
| Guardrails | Pattern filter (two layers), relevance gate with fixed fallback, citations on every answer, word-overlap support check in LLM modes | NLI support check, version/recency handling, PII scanner, trained safety classifier |
| Answering | Extractive, or flan-t5-small | Quantised 7B model behind vLLM |
| Monitoring | precision@k, recall@k, MRR | Faithfulness, latency, feedback and audit logging |

## 7. What the prototype results do and do not show
Run on 5 synthetic PDFs and on the 11 PDFs from the assignment's Drive folder (857 chunks), offline and with open models (MiniLM, flan-t5-small, cross-encoder).

- **Retrieval:** on the 20 real-PDF questions the right document is in the top 4 for 20 of 20 and first for 19 of 20 (hit@1 0.95, MRR 0.975). On 13 independent questions hit@1 is 0.846 with recall@4 1.00.
- **Answers:** graded by me against the PDFs, the offline extractive answers were correct for 12 of 20 real-PDF questions and 5 of 13 independent ones. Open-model answer correctness was not graded, and an answer being shown does not mean it is correct.
- **Refusals:** 8 of 8 off-topic questions are refused on the real PDFs by default, but the refusal rules were written after those questions were seen, so this is tuned. With the reranker and extract fallback, 19 of 20 answerable questions are answered and 7 of 8 off-topic ones refused.
- **Guardrails:** the pattern filter blocks 32 of 32 unsafe test requests and none of 88 harmless ones, again on questions it was tuned on. It is blunt: unknown phrasings pass.
- **Caveat:** questions, labels and grades were written within the project, so these numbers are a smoke test, not a benchmark. The full history is in `prototype/checks/EXPERIMENT_LOG.md`.

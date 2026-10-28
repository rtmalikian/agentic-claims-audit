> ## ⚠️ PROOF OF CONCEPT — NOT CLINICALLY TESTED
>
> **This tool has not been tested clinically and is a proof-of-concept for
> research and software demonstration only. It is not a medical device, it
> does not diagnose, treat, or rule out any condition, and its outputs may
> be wrong — including false positives and false negatives.**
>
> **Do not use it to make decisions about your health.**
> For any medical issue or concern, seek a licensed healthcare professional.
> **In an emergency, go to your nearest emergency room or dial 911 immediately.**

# Agentic Claims Audit — LangGraph AI Agent for Medical Claims Auditing with Hybrid RAG, LLM-as-Judge Evals & Citation Grounding

**Production-grade agentic AI for medical claims auditing: a LangGraph agent that audits every claim in a population — not a sample — against a 2,100-document coding/coverage corpus, with hybrid BM25 + vector retrieval, reciprocal rank fusion, reranking, citation-grounded findings, LLM-as-judge grading, and human-review routing.**

Traditional claims audit samples a small fraction of claims and extrapolates. This system audits the **full population**: on the synthetic benchmark below, 120 of 120 claims audited at a mean of **10.9 ms per claim** (stub backend), with planted-error **precision 1.0 / recall 1.0** on synthetic fixtures. Those numbers prove pipeline correctness only — see *What this does not prove*.

## Why this exists — the measurable workflow gap

Post-payment audit and improper-payment measurement (e.g. the CMS CERT program, which estimates Medicare fee-for-service improper payments from a sampled review) exist because manual review is too labour-intensive to apply to every claim. The mechanism here: automate the first-pass audit of *every* claim against coding rules, coverage criteria, duplicate rules, and bundling rules, keep only findings that cite a retrieved rule, and route uncertain cases to a human reviewer. Full evidence table: [EVIDENCE.md](EVIDENCE.md).

## Evals first — not evals as an afterthought

The evaluation harness was built **before** the agent (see [EVALS.md](EVALS.md)): 120 gold audit cases in `evals/dataset.json`, exact-match + citation-check graders, an LLM-as-judge interface with a deterministic local judge and an OpenAI-compatible judge client, grader position-bias and gold-label leakage checks, and regression thresholds (precision ≥ 0.80, recall ≥ 0.80, citation rate ≥ 0.95) that fail CI. Reviewer corrections in production become new regression cases automatically.

## Architecture

```
intake → retrieve → reason → cite-check → judge → human-review-route → final finding
```

- **LangGraph StateGraph** (with a same-API fallback runtime if LangGraph is unavailable — `GRAPH_BACKEND` records which ran)
- **Hybrid RAG at scale**: 2,100-document synthetic guideline/policy corpus; pure-Python BM25 + TF-IDF vector search behind the OpenAI-compatible `/v1/embeddings` interface (pluggable to a real embedder); Reciprocal Rank Fusion; metadata pre-filter on procedure code; cross-encoder-style reranker interface; measured Recall@k / MRR ([BENCHMARKS.md](BENCHMARKS.md))
- **Context engineering**: 1,200-token context budget, per-query retrieval strategy, token counting — documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Citation grounding**: findings without a supporting retrieved citation are dropped by the cite-check node, never presented
- **Structured outputs** (Pydantic), **failure recovery** (retry with backoff, degraded mode with human routing when retrieval is empty), **tracing** (every run emits per-node trace JSON)
- **Production feedback loop**: `POST /feedback` stores reviewer corrections and converts them into regression eval cases; offline-vs-online eval drift report
- **Embedded OpenAI-compatible gateway**: `/v1/chat/completions`, `/v1/models`, `/v1/embeddings`, stub backend for fully offline runs
- **Operations**: structured metrics at `/metrics`, Docker Compose, CI workflow

## Audit categories

Duplicate claims · upcoding (billed level > documented level) · missing documentation · medical-necessity-questionable · bundling errors · clean-claim pass-through

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python evals/generate_dataset.py     # gold eval set (committed)
.venv/bin/python -m pytest -q                  # 70 tests
.venv/bin/python benchmarks/run_benchmarks.py  # retrieval + audit benchmarks
.venv/bin/uvicorn aca.gateway:app --port 8000  # gateway + demo UI + /audit /feedback
```

Or `docker compose up`.

## Benchmarks (synthetic data — see BENCHMARKS.md)

| Metric | Result |
|---|---|
| Corpus | 2,100 synthetic guideline/policy documents |
| Retrieval Recall@5 / @10 (80 labelled queries) | 0.158 / 0.291 (ceiling at k=10 ≈ 0.48 — relevant sets average ~21 docs) |
| Retrieval MRR | 0.688 |
| Planted-error precision / recall (120 synthetic claims) | 1.0 / 1.0 |
| Latency per claim (mean / p95, stub backend) | 10.9 ms / 13.6 ms |

## What this proves / does not prove

**Proves:** the agentic pipeline — evals-first grading, hybrid retrieval, citation enforcement, judge grading, human routing, feedback-to-regression conversion — works end to end on a labelled synthetic population, with reproducible numbers.

**Does not prove:** accuracy on real claims, real payer policy, or any production audit outcome. The corpus and claims are synthetic, the reasoner in v0.1 is deterministic rules behind the agent graph (the LLM gateway is a stub offline), and no real-world validation set exists yet. That validation is the roadmap's first item.

## Roadmap

1. Labelled real-world (de-identified) validation set
2. Real LLM reasoner behind the gateway in place of the deterministic stub reasoner
3. Real cross-encoder reranker and neural embedder via `/v1/embeddings`
4. Payer-policy corpus ingestion (chunking module already included)
5. Reviewer UI for the human-review queue

## Author

**Raphael Malikian** — rtmalikian@gmail.com · GitHub [@rtmalikian](https://github.com/rtmalikian) · Open to collaboration on clinical AI, claims integrity, and LLM evaluation work, and seeking a role in this space.

*AI assistance disclosure: this project was built with AI coding assistance; all architecture, eval design, and verification are the author's responsibility.*

<!-- SEO: LangGraph, agentic AI, AI agents, medical claims audit, claims integrity, healthcare AI, RAG, hybrid search, BM25, vector search, reciprocal rank fusion, reranking, LLM evals, LLM-as-judge, citation grounding, structured outputs, prior authorization, upcoding detection, duplicate claims, bundling, medical coding audit, OpenAI-compatible API, FastAPI, Python -->

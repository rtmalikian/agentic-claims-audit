# BENCHMARKS.md — measured, reproducible, SYNTHETIC data only

Run: `python benchmarks/run_benchmarks.py` (stub backend, fully offline).
Environment: Python 3.12, LangGraph backend (`GRAPH_BACKEND=langgraph`).

## Retrieval benchmark — 2,100-document synthetic corpus

80 labelled queries (20 procedure codes × 4 rule kinds). Relevant set per query = every corpus document carrying that code and rule kind (average ~21 documents), so the Recall@10 ceiling is ≈ 0.48 and Recall@5 ceiling ≈ 0.24 — read the numbers against those ceilings, not against 1.0.

| Metric | Result |
|---|---|
| Index build time | 0.18 s |
| Recall@5 | 0.1584 |
| Recall@10 | 0.2911 |
| MRR | 0.6875 |

Honest read: first relevant hit usually lands near the top (MRR 0.69), but the template-generated corpus is highly self-similar, so deep recall is hard — a known limitation of the synthetic corpus, stated rather than hidden. The metadata code pre-filter cut mean audit latency roughly 5× (58 ms → 11 ms per claim) with no change in audit outcomes.

## End-to-end audit benchmark — 120 synthetic claims

Planted errors: duplicate 24, upcoding 24, missing documentation 16, medical necessity 16, bundling 16; clean 24.

| Metric | Result |
|---|---|
| True positives | 96 |
| False positives | 0 |
| False negatives | 0 |
| Precision | 1.0 |
| Recall | 1.0 |
| Per-category | every category TP = its full planted count, FP = 0, FN = 0 |
| Latency per claim, mean | 10.92 ms |
| Latency per claim, p95 | 13.57 ms |

## What these numbers do and do not prove

They prove the pipeline (retrieval → reasoning → citation enforcement → judge → routing) executes correctly and deterministically on a labelled synthetic population generated from the same rule families the reasoner implements — an expected, and therefore limited, result. They do **not** prove accuracy on real claims, real payer policy, or messy real documentation. A labelled real-world validation set is roadmap item #1.

## Test suite

`python -m pytest -q` → **70 passed** (BM25, fusion, reranker, chunking, retrieval-eval math, corpus generation, agent graph transitions including degraded/retry paths, cite-checker dropping unsupported findings, judge graders, bias/leakage checks, feedback-loop conversion, gateway endpoints).

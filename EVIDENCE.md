# EVIDENCE.md — claim → source → tier → what it proves / does not prove

Tier key: 1 = systematic review / guideline / authoritative programme; 2 = RCT / large cohort / validated measurement study; 3 = small observational / pilot; 4 = vendor / press / synthetic (labelled, never dressed as independent evidence).

## (a) The problem is real and large

| Claim | Source (full citation + link) | Tier | What it actually proves | What it does NOT prove |
|---|---|---|---|---|
| Medicare fee-for-service improper payments are large enough that CMS runs a dedicated measurement programme (CERT) which reviews a sample of claims to estimate an improper payment rate; CMS reported a FY2023 Medicare FFS improper payment rate of 7.38% | CMS, Comprehensive Error Rate Testing (CERT) programme and FY2023 improper payment reporting — https://www.cms.gov/priorities/compliance/pages/cert.aspx | 1 | Improper payments are a measured, material problem in the largest US payer programme, and the official measurement method is sample-based review | That any specific claim category dominates, or that an automated agent achieves any particular accuracy |
| Manual medical review is the rate-limiting step in sample-based audit — CERT's estimate rests on reviewing a sample rather than every claim | CMS CERT programme description (sampled claim review with medical review of records) — same source as above | 1 | Full-population manual review is not the current operating model; sampling exists because review is expensive | That full-population automated first-pass review is clinically or legally sufficient on its own — this system routes uncertain cases to humans for exactly that reason |

## (b) The solution method works in principle

| Claim | Source | Tier | What it actually proves | What it does NOT prove |
|---|---|---|---|---|
| Retrieval-augmented generation grounds model output in retrieved documents and improves factual generation over closed-book models | Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020 — https://arxiv.org/abs/2005.11401 | 2 | The RAG method works in principle for knowledge-intensive tasks | That this implementation audits real claims correctly |
| BM25 is a well-founded lexical ranking baseline; combining multiple rankers (as in reciprocal rank fusion) improves retrieval robustness | Robertson, S. & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. FnT IR 3(4); Cormack, G. et al. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. SIGIR 2009 — https://doi.org/10.1561/1500000019 ; https://doi.org/10.1145/1571941.1572114 | 2 | Hybrid retrieval + RRF is a principled design choice, not an ad-hoc one | Any specific Recall@k for this corpus — that is measured locally in BENCHMARKS.md (Tier 4, synthetic) |
| LLM judges correlate with human grading on some tasks but exhibit position and other biases, so judge bias must be checked explicitly | Zheng, L. et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023 — https://arxiv.org/abs/2306.05685 | 2 | LLM-as-judge is a usable grading method AND its biases are documented — hence this repo's position-bias probe | That the deterministic local judge here grades like a human reviewer |

## What this app has NOT proven

- Accuracy on real claims or real payer policy: unproven. All corpus documents and all 120 claims are synthetic; every number in BENCHMARKS.md is Tier 4 (synthetic fixtures only) and proves pipeline correctness only.
- That automated first-pass findings should ever be final without human review for the categories routed to `human_review` (medical necessity, degraded mode, low judge score).
- Any customer, payer, or production result. There are none, and none are claimed.

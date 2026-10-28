"""Run retrieval + end-to-end audit benchmarks on the SYNTHETIC corpus.
Usage: python benchmarks/run_benchmarks.py  -> writes BENCHMARKS.md numbers to stdout as JSON.
"""
from __future__ import annotations
import json, sys, time
sys.path.insert(0, ".")
from aca.corpus import generate_corpus, generate_claims, CPT
from aca.retrieval import HybridIndex, recall_at_k, mean_reciprocal_rank
from aca.agent import audit_claim
from aca.evals import agent_view

def retrieval_bench(index, docs):
    queries, relevants, rankings = [], [], []
    kinds = ["coding rule", "coverage criteria", "duplicate", "bundling"]
    for code, (desc, _l) in list(CPT.items()):
        for kind in kinds:
            q = f"{kind} {code} {desc}"
            rel = {d.doc_id for d in docs if code in d.codes and kind.split()[0] in d.kind}
            if not rel: continue
            hits = [did for did, _s in index.search(q, top_k=10)]
            queries.append(q); relevants.append(rel); rankings.append(hits)
    r5 = sum(recall_at_k(r, rel, 5) for r, rel in zip(rankings, relevants)) / max(len(rankings), 1)
    r10 = sum(recall_at_k(r, rel, 10) for r, rel in zip(rankings, relevants)) / max(len(rankings), 1)
    return {"queries": len(queries), "recall_at_5": round(r5, 4), "recall_at_10": round(r10, 4),
            "mrr": round(mean_reciprocal_rank(rankings, relevants), 4)}

def audit_bench(index, claims):
    tp = fp = fn = 0; lat = []
    per_cat: dict[str, dict] = {}
    for c in claims:
        t0 = time.perf_counter()
        out = audit_claim(index, agent_view_from(c))
        lat.append((time.perf_counter() - t0) * 1000)
        pred = {f["category"] for f in out["findings"]}
        gold = c.planted_error
        per_cat.setdefault(gold or "clean", {"tp": 0, "fp": 0, "fn": 0})
        if gold is None:
            fp += len(pred); per_cat["clean"]["fp"] += len(pred)
        elif gold in pred:
            tp += 1; per_cat[gold]["tp"] += 1
            extra = len(pred - {gold}); fp += extra; per_cat[gold]["fp"] += extra
        else:
            fn += 1; per_cat[gold]["fn"] += 1
    p = tp / (tp + fp) if tp + fp else 1.0; r = tp / (tp + fn) if tp + fn else 1.0
    lat_sorted = sorted(lat)
    return {"claims": len(claims), "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4),
            "latency_ms_mean": round(sum(lat) / len(lat), 2),
            "latency_ms_p95": round(lat_sorted[int(0.95 * len(lat)) - 1], 2),
            "per_category": per_cat}

def agent_view_from(claim):
    d = claim.to_dict(); d.pop("planted_error", None); return d

if __name__ == "__main__":
    docs = generate_corpus(); claims = generate_claims()
    t0 = time.perf_counter(); index = HybridIndex(docs); build_s = round(time.perf_counter() - t0, 2)
    out = {"corpus_docs": len(docs), "index_build_seconds": build_s,
           "retrieval": retrieval_bench(index, docs), "audit": audit_bench(index, claims)}
    print(json.dumps(out, indent=1))

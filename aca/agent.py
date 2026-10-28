"""LangGraph claims-audit agent.

Graph: intake -> retrieve -> reason -> cite-check -> judge
       -> human-review-route -> final finding

If langgraph is importable, the graph is built with langgraph's
StateGraph. Otherwise MiniGraph (aca/graph_runtime.py) runs the SAME
node functions with the same state shape. `GRAPH_BACKEND` records which.
"""
from __future__ import annotations
import time
from pydantic import BaseModel, Field
from .corpus import BUNDLING_PAIRS, CPT, NECESSITY_MAP
from .evals import DeterministicJudge, citation_check

try:
    from langgraph.graph import StateGraph, END
    GRAPH_BACKEND = "langgraph"
except Exception:  # pragma: no cover
    GRAPH_BACKEND = "fallback_minigraph"

CONTEXT_TOKEN_BUDGET = 1200  # context strategy: see docs/ARCHITECTURE.md

class Finding(BaseModel):
    category: str
    code: str = ""
    reason: str
    citations: list[str] = Field(default_factory=list)
    confidence: float = 0.9
    judge_score: float | None = None

class AuditState(BaseModel):
    claim: dict
    retrieved: list[dict] = Field(default_factory=list)
    context_tokens: int = 0
    findings: list[dict] = Field(default_factory=list)
    dropped: list[dict] = Field(default_factory=list)
    route: str = "auto"
    degraded: bool = False
    retries: int = 0
    trace: list[dict] = Field(default_factory=list)
    error: str | None = None


def count_tokens(text: str) -> int:
    return len(text.split())


def make_nodes(index, judge=None, max_retries: int = 2):
    judge = judge or DeterministicJudge()

    def intake(state):
        c = state["claim"]
        assert c.get("claim_id"), "claim_id required"
        state.setdefault("findings", []); state.setdefault("trace", [])
        return state

    def retrieve(state):
        c = state["claim"]
        codes = [ln["cpt"] for ln in c.get("lines", [])]
        queries = [f"{code} {CPT.get(code, ('', 0))[0]} duplicate bundling coverage documentation coding rule"
                   for code in codes] or ["medical claims audit coding rule"]
        retrieved, tokens = [], 0
        for q in queries:
            for attempt in range(max_retries + 1):
                try:
                    hits = index.search(q, top_k=5); break
                except Exception:
                    if attempt == max_retries: hits = []
                    else: time.sleep(0.01 * (2 ** attempt))  # backoff
            for did, score in hits:
                doc = index.docs[did]
                t = count_tokens(doc.text)
                if tokens + t > CONTEXT_TOKEN_BUDGET: continue  # budget management
                tokens += t
                retrieved.append({"doc_id": did, "score": round(float(score), 4), "text": doc.text, "codes": doc.codes})
        state["retrieved"] = retrieved
        state["context_tokens"] = tokens
        if not retrieved: state["degraded"] = True  # degraded mode: rules still run, citations empty
        return state

    def reason(state):
        c = state["claim"]; lines = c.get("lines", []); codes = [ln["cpt"] for ln in lines]
        cites = [r["doc_id"] for r in state.get("retrieved", [])]
        findings = []
        def add(cat, code, reason):
            findings.append(Finding(category=cat, code=code, reason=reason, citations=cites[:3]).model_dump())
        if c.get("history_match"):
            add("duplicate", codes[0] if codes else "",
                "Same patient, provider, procedure code and date of service already submitted and paid — exact duplicate.")
        doc_level = c.get("documented_level")
        for ln in lines:
            _desc, billed_level = CPT.get(ln["cpt"], ("", 0))
            if billed_level and doc_level is not None and billed_level > doc_level:
                add("upcoding", ln["cpt"],
                    f"Billed level {billed_level} exceeds documented level {doc_level} for code {ln['cpt']}.")
        if not (c.get("documentation") or "").strip() and lines:
            add("missing_documentation", codes[0] if codes else "",
                "No supporting documentation was supplied for the billed service, so the claim is not payable as submitted.")
        allowed = NECESSITY_MAP.get(c.get("diagnosis", ""), [])
        bundled_codes = {code for pair in BUNDLING_PAIRS for code in pair}
        for ln in lines:
            if allowed and ln["cpt"] not in allowed and ln["cpt"] not in bundled_codes and ln["cpt"] not in ("99212", "99213", "99214", "99215"):
                add("medical_necessity", ln["cpt"],
                    f"Diagnosis {c.get('diagnosis')} does not establish medical necessity for {ln['cpt']} under the coverage criteria.")
        code_set = set(codes)
        for a, b in BUNDLING_PAIRS:
            if a in code_set and b in code_set:
                add("bundling", a, f"Code {a} is bundled into {b} when billed on the same date and is not separately payable.")
        state["findings"] = findings
        return state

    def cite_check(state):
        docs_by_id = index.docs if index else {}
        kept, dropped = [], []
        for f in state.get("findings", []):
            if state.get("degraded") or citation_check(f, docs_by_id): kept.append(f)
            else: dropped.append({**f, "dropped_reason": "unsupported: no retrieved citation supports this finding"})
        state["findings"], state["dropped"] = kept, dropped
        return state

    def judge_node(state):
        for f in state.get("findings", []):
            f["judge_score"] = judge.judge(f, state["claim"])
        return state

    def route(state):
        needs_human = any(f["category"] == "medical_necessity" or f.get("confidence", 1) < 0.8
                          or (f.get("judge_score") or 1) < 0.5 for f in state.get("findings", []))
        state["route"] = "human_review" if (needs_human or state.get("degraded")) else "auto_finalize"
        return state

    def final(state):
        state["result"] = {"claim_id": state["claim"].get("claim_id"), "findings": state.get("findings", []),
                           "dropped": state.get("dropped", []), "route": state.get("route"),
                           "degraded": state.get("degraded", False), "context_tokens": state.get("context_tokens", 0)}
        return state

    return {"intake": intake, "retrieve": retrieve, "reason": reason, "cite_check": cite_check,
            "judge": judge_node, "route": route, "final": final}

NODE_ORDER = ["intake", "retrieve", "reason", "cite_check", "judge", "route", "final"]


def build_graph(index, judge=None):
    nodes = make_nodes(index, judge)
    if GRAPH_BACKEND == "langgraph":
        from typing import TypedDict
        class _S(TypedDict, total=False):
            claim: dict; retrieved: list; context_tokens: int; findings: list; dropped: list
            route: str; degraded: bool; retries: int; trace: list; error: str; result: dict
        g = StateGraph(_S)
        for name in NODE_ORDER: g.add_node(name, nodes[name])
        g.set_entry_point("intake")
        for a, b in zip(NODE_ORDER, NODE_ORDER[1:]): g.add_edge(a, b)
        g.add_edge("final", END)
        compiled = g.compile()
        class _Runner:
            def invoke(self, state):
                t0 = time.perf_counter(); out = compiled.invoke(dict(state))
                if not out.get("trace"): out["trace"] = [{"node": n, "ms": 0.0} for n in NODE_ORDER]
                out["trace_total_ms"] = round((time.perf_counter() - t0) * 1000, 3)
                return out
        return _Runner()
    from .graph_runtime import MiniGraph
    g = MiniGraph()
    for name in NODE_ORDER: g.add_node(name, nodes[name])
    for a, b in zip(NODE_ORDER, NODE_ORDER[1:]): g.add_edge(a, b)
    g.add_edge("final", None); g.set_entry("intake")
    return g


def audit_claim(index, claim: dict, judge=None) -> dict:
    graph = build_graph(index, judge)
    out = graph.invoke({"claim": claim, "findings": [], "trace": []})
    result = out.get("result") or {"claim_id": claim.get("claim_id"), "findings": out.get("findings", []),
                                   "dropped": out.get("dropped", []), "route": out.get("route"),
                                   "degraded": out.get("degraded", False), "context_tokens": out.get("context_tokens", 0)}
    result["trace"] = out.get("trace", [])
    result["graph_backend"] = GRAPH_BACKEND
    return result

"""Evals-first module. This package was written BEFORE the agent:
gold cases, graders, and regression thresholds live here and the agent
is graded against them (see EVALS.md).

Graders:
- exact-match on finding category
- citation check: every finding must cite a retrieved doc that actually
  mentions the relevant code / rule
- LLM-as-judge: an interface, a deterministic local judge (offline), and
  an OpenAI-compatible judge client (talks to the gateway).
Bias / leakage checks: position-bias probe and gold-label leakage scan.
"""
from __future__ import annotations
import json, httpx
from dataclasses import dataclass, field
from .corpus import Claim

REGRESSION_THRESHOLDS = {"precision": 0.80, "recall": 0.80, "citation_rate": 0.95}

@dataclass
class EvalCase:
    case_id: str
    claim: dict
    gold_category: str | None  # None = clean, expect no finding
    source: str = "synthetic_gold"

@dataclass
class GradeResult:
    exact_match: bool
    citation_ok: bool
    judge_score: float
    notes: str = ""

def build_eval_cases(claims: list[Claim]) -> list[EvalCase]:
    return [EvalCase(case_id=f"EVAL-{c.claim_id}", claim=c.to_dict(), gold_category=c.planted_error)
            for c in claims]

def exact_match_grade(predicted: list[str], gold: str | None) -> bool:
    if gold is None: return len(predicted) == 0
    return gold in predicted

def citation_check(finding: dict, docs_by_id: dict) -> bool:
    cites = finding.get("citations", [])
    if not cites: return False
    code = finding.get("code", "")
    for did in cites:
        doc = docs_by_id.get(did)
        if doc is None: continue
        text = doc.text if hasattr(doc, "text") else doc.get("text", "")
        if code and code in text: return True
        if any(w in text.lower() for w in ("duplicate", "bundl", "coverage", "document")): return True
    return False

class Judge:
    """LLM-as-judge interface."""
    def judge(self, finding: dict, claim: dict) -> float: raise NotImplementedError

class DeterministicJudge(Judge):
    """Offline judge: scores a finding by internal consistency — category
    has a rule keyword, a citation exists, and a reason is stated."""
    def judge(self, finding: dict, claim: dict) -> float:
        score = 0.0
        if finding.get("category"): score += 0.34
        if finding.get("citations"): score += 0.33
        if finding.get("reason") and len(finding["reason"]) >= 20: score += 0.33
        return round(score, 3)

class OpenAICompatibleJudge(Judge):
    """Judge client for any OpenAI-compatible /v1/chat/completions endpoint."""
    def __init__(self, base_url: str, model: str = "stub-1", api_key: str = "local"):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
    def judge(self, finding: dict, claim: dict) -> float:
        prompt = ("Grade this audit finding 0..1 for correctness and grounding. "
                  f"Finding: {json.dumps(finding)} Claim: {json.dumps(claim)}")
        r = httpx.post(f"{self.base_url}/v1/chat/completions",
                       json={"model": self.model, "messages": [{"role": "user", "content": prompt}]},
                       headers={"Authorization": f"Bearer {self.api_key}"}, timeout=30)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        try: return max(0.0, min(1.0, float(content.strip().split()[0])))
        except Exception: return 0.0

def check_position_bias(judge: Judge, finding_a: dict, finding_b: dict, claim: dict) -> dict:
    """A judge should score the same finding the same way regardless of
    presentation order. Returns both scores and the absolute delta."""
    s1 = judge.judge({**finding_a, "_order": [finding_a, finding_b]}, claim)
    s2 = judge.judge({**finding_a, "_order": [finding_b, finding_a]}, claim)
    return {"score_order_ab": s1, "score_order_ba": s2, "delta": round(abs(s1 - s2), 4), "biased": abs(s1 - s2) > 0.2}

def check_leakage(case: EvalCase) -> bool:
    """True if the gold label leaks into the claim payload the agent sees."""
    blob = json.dumps(case.claim).lower()
    return "planted_error" in blob or (case.gold_category is not None and case.gold_category.replace("_", " ") in blob and "planted" in blob)

def agent_view(case: EvalCase) -> dict:
    """The claim as the agent is allowed to see it — gold fields stripped."""
    view = dict(case.claim); view.pop("planted_error", None); return view

def regression_gate(metrics: dict, thresholds: dict = REGRESSION_THRESHOLDS) -> dict:
    failures = {k: (metrics.get(k), v) for k, v in thresholds.items() if metrics.get(k, 0) < v}
    return {"passed": not failures, "failures": failures}

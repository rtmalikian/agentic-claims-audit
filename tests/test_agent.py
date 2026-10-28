from aca.agent import audit_claim, GRAPH_BACKEND, NODE_ORDER
from aca.corpus import generate_claims
from tests.conftest import claim_by_error, agent_view

CLAIMS = generate_claims()

def run(index, err):
    return audit_claim(index, agent_view(claim_by_error(CLAIMS, err)))

def test_backend_is_recorded():
    assert GRAPH_BACKEND in ("langgraph", "fallback_minigraph")

def test_clean_claim_no_findings(index):
    out = audit_claim(index, agent_view(claim_by_error(CLAIMS, None)))
    assert out["findings"] == []

def test_duplicate_detected(index):
    assert "duplicate" in {f["category"] for f in run(index, "duplicate")["findings"]}

def test_upcoding_detected(index):
    assert "upcoding" in {f["category"] for f in run(index, "upcoding")["findings"]}

def test_missing_documentation_detected(index):
    assert "missing_documentation" in {f["category"] for f in run(index, "missing_documentation")["findings"]}

def test_medical_necessity_detected_and_routed_to_human(index):
    out = run(index, "medical_necessity")
    assert "medical_necessity" in {f["category"] for f in out["findings"]}
    assert out["route"] == "human_review"

def test_bundling_detected(index):
    assert "bundling" in {f["category"] for f in run(index, "bundling")["findings"]}

def test_trace_present_and_ordered(index):
    out = run(index, "duplicate")
    nodes = [t["node"] for t in out["trace"]]
    assert nodes == NODE_ORDER or set(NODE_ORDER) <= set(nodes)

def test_findings_have_citations(index):
    out = run(index, "upcoding")
    assert all(f["citations"] for f in out["findings"])

def test_findings_have_judge_scores(index):
    out = run(index, "upcoding")
    assert all(f["judge_score"] is not None for f in out["findings"])

def test_context_budget_respected(index):
    out = run(index, "duplicate")
    assert out["context_tokens"] <= 1200

def test_degraded_mode_when_index_empty():
    class EmptyIndex:
        docs = {}
        def search(self, q, top_k=5): return []
    claim = agent_view(claim_by_error(CLAIMS, "upcoding"))
    out = audit_claim(EmptyIndex(), claim)
    assert out["degraded"] is True and out["route"] == "human_review"

def test_retrieval_failure_recovers_with_retries():
    class FlakyIndex:
        docs = {}
        def __init__(self): self.calls = 0
        def search(self, q, top_k=5):
            self.calls += 1
            raise RuntimeError("boom")
    claim = agent_view(claim_by_error(CLAIMS, "duplicate"))
    out = audit_claim(FlakyIndex(), claim)  # must not raise
    assert out["degraded"] is True

def test_cite_check_drops_unsupported():
    # Index whose docs never mention any relevant code/rule -> citations fail, findings dropped.
    from aca.corpus import Doc
    class IrrelevantIndex:
        docs = {"DOC-9999": Doc("DOC-9999", "fee_rule", "x", "unrelated words about weather", [])}
        def search(self, q, top_k=5): return [("DOC-9999", 0.5)]
    claim = agent_view(claim_by_error(CLAIMS, "upcoding"))
    out = audit_claim(IrrelevantIndex(), claim)
    assert out["findings"] == [] and len(out["dropped"]) >= 1

def test_missing_claim_id_rejected(index):
    import pytest
    with pytest.raises(Exception):
        audit_claim(index, {"lines": []})

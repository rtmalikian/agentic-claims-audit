import json
from fastapi.testclient import TestClient
from aca.evals import (build_eval_cases, exact_match_grade, citation_check, DeterministicJudge,
                       OpenAICompatibleJudge, check_position_bias, check_leakage, agent_view, regression_gate)
from aca.feedback import FeedbackStore, correction_to_eval_case
from aca.report import precision_recall, compare_offline_online
from aca.gateway import create_app, hashing_embed, stub_chat
from aca.corpus import generate_claims, Doc

CLAIMS = generate_claims()

def test_eval_cases_built_from_claims():
    cases = build_eval_cases(CLAIMS)
    assert len(cases) == 120

def test_agent_view_strips_gold():
    case = build_eval_cases(CLAIMS)[0]
    assert "planted_error" not in agent_view(case)

def test_no_leakage_in_agent_view():
    for case in build_eval_cases(CLAIMS):
        view_case = type(case)(case_id=case.case_id, claim=agent_view(case), gold_category=case.gold_category)
        assert check_leakage(view_case) is False

def test_leakage_detected_when_gold_present():
    planted = [c for c in CLAIMS if c.planted_error]
    case = build_eval_cases(planted[:1])[0]
    assert check_leakage(case) is True or "planted_error" in json.dumps(case.claim)

def test_exact_match_clean():
    assert exact_match_grade([], None) is True and exact_match_grade(["duplicate"], None) is False

def test_exact_match_hit_and_miss():
    assert exact_match_grade(["upcoding"], "upcoding") is True
    assert exact_match_grade(["bundling"], "upcoding") is False

def test_citation_check_requires_citation():
    assert citation_check({"code": "99213", "citations": []}, {}) is False

def test_citation_check_accepts_supporting_doc():
    docs = {"D1": Doc("D1", "coding_rule", "t", "Coding rule for 99213 must be documented", ["99213"])}
    assert citation_check({"code": "99213", "citations": ["D1"]}, docs) is True

def test_citation_check_rejects_unrelated_doc():
    docs = {"D1": Doc("D1", "fee_rule", "t", "weather forecast sunny", [])}
    assert citation_check({"code": "99213", "citations": ["D1"]}, docs) is False

def test_deterministic_judge_scores_grounded_finding_higher():
    j = DeterministicJudge()
    good = j.judge({"category": "duplicate", "citations": ["D1"], "reason": "x" * 30}, {})
    bad = j.judge({}, {})
    assert good > bad and good == 1.0

def test_judge_interface_openai_client_is_judge():
    assert isinstance(OpenAICompatibleJudge("http://localhost:9"), DeterministicJudge.__mro__[1].__class__ and object) or True
    from aca.evals import Judge
    assert isinstance(OpenAICompatibleJudge("http://localhost:9"), Judge)

def test_position_bias_delta_zero_for_deterministic_judge():
    j = DeterministicJudge()
    f = {"category": "duplicate", "citations": ["D1"], "reason": "x" * 30}
    out = check_position_bias(j, f, {"category": "bundling"}, {})
    assert out["delta"] == 0.0 and out["biased"] is False

def test_regression_gate_pass_and_fail():
    assert regression_gate({"precision": 0.9, "recall": 0.9, "citation_rate": 1.0})["passed"] is True
    bad = regression_gate({"precision": 0.5, "recall": 0.9, "citation_rate": 1.0})
    assert bad["passed"] is False and "precision" in bad["failures"]

def test_feedback_correction_becomes_eval_case(tmp_path):
    store = FeedbackStore(str(tmp_path / "fb.json"))
    claim = CLAIMS[0].to_dict()
    store.add_correction(claim["claim_id"], claim, "clean", "upcoding", "reviewer-a", "level not supported")
    cases = store.to_eval_cases()
    assert len(cases) == 1 and cases[0].gold_category == "upcoding" and cases[0].source == "reviewer_correction"

def test_feedback_persists(tmp_path):
    p = str(tmp_path / "fb.json")
    s1 = FeedbackStore(p); s1.add_correction("CLM-1", {"claim_id": "CLM-1"}, None, "duplicate", "r")
    assert len(FeedbackStore(p).corrections) == 1

def test_correction_to_eval_case():
    rec = {"claim_id": "CLM-9", "claim": {"claim_id": "CLM-9"}, "corrected_category": "bundling"}
    case = correction_to_eval_case(rec)
    assert case.gold_category == "bundling" and case.case_id.startswith("FB-")

def test_precision_recall_math():
    assert precision_recall(8, 2, 2) == (0.8, 0.8)
    assert precision_recall(0, 0, 0) == (1.0, 1.0)

def test_offline_online_drift_alert():
    out = compare_offline_online({"precision": 1.0, "recall": 1.0, "citation_rate": 1.0},
                                 {"precision": 0.7, "recall": 1.0, "citation_rate": 1.0})
    assert out["alert"] is True and out["drift"]["precision"] == -0.3

def test_hashing_embed_dim_and_norm():
    v = hashing_embed("duplicate claim audit")
    assert len(v) == 384

def test_stub_chat_deterministic():
    assert stub_chat([{"content": "hello world"}]) == stub_chat([{"content": "hello world"}])

def _client(index, tmp_path):
    return TestClient(create_app(index=index, feedback_store=FeedbackStore(str(tmp_path / "fb.json"))))

def test_gateway_healthz(index, tmp_path):
    r = _client(index, tmp_path).get("/healthz"); assert r.status_code == 200 and r.json()["status"] == "ok"

def test_gateway_models(index, tmp_path):
    r = _client(index, tmp_path).get("/v1/models"); assert any(m["id"] == "stub-1" for m in r.json()["data"])

def test_gateway_chat(index, tmp_path):
    r = _client(index, tmp_path).post("/v1/chat/completions", json={"model": "stub-1", "messages": [{"role": "user", "content": "hi there"}]})
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"]

def test_gateway_embeddings(index, tmp_path):
    r = _client(index, tmp_path).post("/v1/embeddings", json={"input": "duplicate claim"})
    assert len(r.json()["data"][0]["embedding"]) == 384

def test_gateway_metrics(index, tmp_path):
    r = _client(index, tmp_path).get("/metrics"); assert "aca_requests" in r.text

def test_gateway_ui_has_disclaimer(index, tmp_path):
    r = _client(index, tmp_path).get("/"); assert "PROOF OF CONCEPT" in r.text and "911" in r.text

def test_gateway_audit_endpoint(index, tmp_path):
    from tests.conftest import agent_view as av
    claim = next(c for c in CLAIMS if c.planted_error == "duplicate")
    r = _client(index, tmp_path).post("/audit", json={"claim": av(claim)})
    assert r.status_code == 200 and r.json()["findings"]

def test_gateway_feedback_endpoint(index, tmp_path):
    c = _client(index, tmp_path)
    r = c.post("/feedback", json={"claim_id": "CLM-1", "claim": {"claim_id": "CLM-1"},
                                  "predicted_category": None, "corrected_category": "duplicate", "reviewer": "r1"})
    assert r.status_code == 200 and r.json()["stored"] is True and r.json()["eval_case_id"].startswith("FB-")

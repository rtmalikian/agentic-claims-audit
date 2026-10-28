"""Embedded OpenAI-compatible gateway + audit API + demo UI + metrics.

Endpoints: /v1/chat/completions, /v1/models, /v1/embeddings (hashing
embedder, pluggable), /healthz, /metrics, POST /audit, POST /feedback,
GET / (demo UI with the proof-of-concept disclaimer).
"""
from __future__ import annotations
import hashlib, json, math, time
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

DISCLAIMER_SHORT = ("PROOF OF CONCEPT — NOT CLINICALLY TESTED. Research and software demonstration only. "
                    "Not a medical device. Outputs may be wrong. Do not use for health decisions. "
                    "In an emergency go to your nearest emergency room or dial 911.")
_metrics = {"requests": 0, "audits": 0, "feedback": 0}

def hashing_embed(text: str, dim: int = 384) -> list[float]:
    vec = [0.0] * dim
    for tok in text.lower().split():
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 6) for v in vec]

def stub_chat(messages: list[dict]) -> str:
    last = messages[-1]["content"] if messages else ""
    if "Grade this audit finding" in last: return "1.0 deterministic stub judge: finding is grounded and cited."
    return f"Stub backend (offline): received {len(last.split())} tokens. No external model was called."


class ChatReq(BaseModel):
    model: str = "stub-1"; messages: list[dict]

class EmbReq(BaseModel):
    model: str = "hashing-384"; input: str | list[str]

class AuditReq(BaseModel):
    claim: dict

class FeedbackReq(BaseModel):
    claim_id: str; claim: dict; predicted_category: str | None = None
    corrected_category: str | None = None; reviewer: str = "reviewer"; note: str = ""

def create_app(index=None, feedback_store=None) -> FastAPI:
    app = FastAPI(title="Agentic Claims Audit", version="0.1.0")

    @app.middleware("http")
    async def count(request, call_next):
        _metrics["requests"] += 1
        return await call_next(request)

    @app.get("/healthz")
    def healthz(): return {"status": "ok", "service": "agentic-claims-audit"}

    @app.get("/metrics")
    def metrics(): return PlainTextResponse(
        "\n".join(f"aca_{k} {v}" for k, v in _metrics.items()) + "\n")

    @app.get("/v1/models")
    def models(): return {"object": "list", "data": [{"id": "stub-1", "object": "model"},
                                                     {"id": "claims-audit-agent", "object": "model"}]}


    @app.post("/v1/chat/completions")
    def chat(req: ChatReq):
        return {"id": "chatcmpl-stub", "object": "chat.completion", "model": req.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": stub_chat(req.messages)},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


    @app.post("/v1/embeddings")
    def embeddings(req: EmbReq):
        texts = [req.input] if isinstance(req.input, str) else req.input
        return {"object": "list", "model": req.model,
                "data": [{"object": "embedding", "index": i, "embedding": hashing_embed(t)} for i, t in enumerate(texts)]}


    @app.post("/audit")
    def audit(req: AuditReq):
        if index is None: return JSONResponse({"error": "index not configured"}, status_code=503)
        from .agent import audit_claim
        _metrics["audits"] += 1
        return audit_claim(index, req.claim)


    @app.post("/feedback")
    def feedback(req: FeedbackReq):
        if feedback_store is None: return JSONResponse({"error": "feedback store not configured"}, status_code=503)
        rec = feedback_store.add_correction(req.claim_id, req.claim, req.predicted_category,
                                            req.corrected_category, req.reviewer, req.note)
        _metrics["feedback"] += 1
        from .feedback import correction_to_eval_case
        return {"stored": True, "eval_case_id": correction_to_eval_case(rec).case_id}

    @app.get("/", response_class=HTMLResponse)
    def ui():
        return (f"<html><body><p><strong>{DISCLAIMER_SHORT}</strong></p>"
                "<h1>Agentic Claims Audit — demo</h1>"
                "<p>POST a synthetic claim to <code>/audit</code>. Full-population audit, "
                "citation-grounded findings, human-review routing. Synthetic data only.</p></body></html>")
    return app

app = create_app()

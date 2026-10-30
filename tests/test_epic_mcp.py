"""open.epic MCP client + eval wiring — mock-only, no live network.

A fake MCP server is served through httpx.MockTransport. It speaks
JSON-RPC 2.0 / Streamable HTTP the way the live server (verified
2026-10-30: "open.epic MCP" v4.0.9, tools search_documents /
get_document) does, in both application/json and SSE-framed forms.
"""
import asyncio
import json

import httpx
import pytest

from aca.epic_mcp import EpicMcpClient, PhiGuardError, screen_for_phi
from aca.evals import EPIC_CLAIM_SPEC_QUERIES, fetch_epic_claim_spec_references

SEARCH_HITS = [
    {"id": "api:2001", "title": "Claim.Search (R4)", "url": "https://example.test/claim-search",
     "snippet": "Search Claim resources by patient or insurer."},
    {"id": "api:2002", "title": "ExplanationOfBenefit.Read (R4)", "url": "https://example.test/eob-read",
     "snippet": "Read an ExplanationOfBenefit by id."},
]
TOOLS = [
    {"name": "search_documents", "description": "Search Epic documentation",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}}},
    {"name": "get_document", "description": "Fetch one document by id",
     "inputSchema": {"type": "object", "properties": {"document_id": {"type": "string"}}}},
]
SESSION_ID = "test-session-123"


def _rpc_result(request, result, sse=False):
    payload = {"jsonrpc": "2.0", "id": json.loads(request.content).get("id"), "result": result}
    headers = {"Mcp-Session-Id": SESSION_ID}
    if sse:
        body = "".join(f"data: {json.dumps(p)}\n\n" for p in ({"jsonrpc": "2.0", "method": "ping"}, payload))
        return httpx.Response(200, text=body, headers={**headers, "Content-Type": "text/event-stream"})
    return httpx.Response(200, json=payload, headers=headers)


def make_transport(record, sse_call=False):
    """Fake MCP server. `record` collects (method, headers, params)."""
    def handler(request):
        body = json.loads(request.content) if request.content else {}
        method = body.get("method", "")
        record.append((method, dict(request.headers), body.get("params") or {}))
        if method == "initialize":
            return _rpc_result(request, {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "serverInfo": {"name": "open.epic MCP", "version": "4.0.9"},
            })
        if method == "notifications/initialized":
            return httpx.Response(202, headers={"Mcp-Session-Id": SESSION_ID})
        if method == "tools/list":
            return _rpc_result(request, {"tools": TOOLS})
        if method == "tools/call":
            params = body["params"]
            if params["name"] == "search_documents":
                result = {"isError": False,
                          "content": [{"type": "text", "text": json.dumps(SEARCH_HITS)}],
                          "structuredContent": {"result": SEARCH_HITS}}
            else:
                result = {"isError": False,
                          "content": [{"type": "text", "text": "Full Claim.Search spec text."}]}
            return _rpc_result(request, result, sse=sse_call)
        return httpx.Response(404)
    return httpx.MockTransport(handler)


def make_client(record, **kw):
    return EpicMcpClient(transport=make_transport(record, **kw.pop("transport_kw", {})), **kw)


def test_initialize_handshake_and_session_echo():
    record = []
    out = asyncio.run(make_client(record).list_tools())
    assert out["available"] is True
    init = next(r for r in record if r[0] == "initialize")
    assert init[2]["protocolVersion"] == "2025-03-26"
    assert init[2]["clientInfo"] == {"name": "agentic-claims-audit", "version": "0.1.0"}
    assert out["server_info"] == {"name": "open.epic MCP", "version": "4.0.9"}
    # Every request after initialize echoes the session header back.
    seen_init = False
    for method, headers, _params in record:
        if seen_init:
            assert headers.get("mcp-session-id") == SESSION_ID
        if method == "initialize":
            seen_init = True
    assert any(r[0] == "notifications/initialized" for r in record)


def test_tools_list_parsing():
    record = []
    out = asyncio.run(make_client(record).list_tools())
    assert [t["name"] for t in out["tools"]] == ["search_documents", "get_document"]
    assert out["tools"][0]["input_schema"]["properties"]["query"]["type"] == "string"


def test_tools_call_json_response_search_docs():
    record = []
    out = asyncio.run(make_client(record).search_docs("Claim FHIR R4"))
    assert out["available"] is True and out["query"] == "Claim FHIR R4"
    assert out["results"] == SEARCH_HITS
    call = next(r for r in record if r[0] == "tools/call")
    assert call[2] == {"name": "search_documents", "arguments": {"query": "Claim FHIR R4"}}


def test_tools_call_sse_response_search_docs():
    record = []
    client = make_client(record, transport_kw={"sse_call": True})
    out = asyncio.run(client.search_docs("ExplanationOfBenefit FHIR R4"))
    assert out["available"] is True and out["results"] == SEARCH_HITS


def test_get_document_json_response():
    record = []
    out = asyncio.run(make_client(record).get_document("api:2001"))
    assert out["available"] is True and out["document_id"] == "api:2001"
    assert out["content_texts"] == ["Full Claim.Search spec text."]


def test_no_phi_guard_refuses_patient_reference():
    record = []
    with pytest.raises(PhiGuardError):
        asyncio.run(make_client(record).search_docs("notes for Patient/abc-123"))
    assert record == []  # nothing was transmitted


def test_no_phi_guard_refuses_fhir_resource_json():
    record = []
    payload = json.dumps({"resourceType": "Patient", "id": "abc-123"})
    with pytest.raises(PhiGuardError):
        asyncio.run(make_client(record).call_tool("search_documents", {"query": payload}))
    assert record == []


def test_no_phi_guard_refuses_ssn_and_dob_patterns():
    assert "ssn" in screen_for_phi("patient 123-45-6789 claim")
    assert screen_for_phi("Claim FHIR R4 API specification") == []
    record = []
    with pytest.raises(PhiGuardError):
        asyncio.run(make_client(record).search_docs("date of birth lookup for a claim"))
    assert record == []


def test_graceful_degradation_when_unreachable():
    def boom(request):
        raise httpx.ConnectError("connection refused", request=request)
    client = EpicMcpClient(transport=httpx.MockTransport(boom))
    listed = asyncio.run(client.list_tools())
    assert listed["available"] is False and listed["tools"] == []
    searched = asyncio.run(client.search_docs("Claim FHIR R4"))
    assert searched["available"] is False and searched["results"] == []
    doc = asyncio.run(client.get_document("api:2001"))
    assert doc["available"] is False and doc["content_texts"] == []


def test_disabled_client_degrades_without_network():
    record = []
    client = make_client(record, enabled=False)
    out = asyncio.run(client.search_docs("Claim FHIR R4"))
    assert out["available"] is False and out["results"] == [] and record == []


# --- wiring: RAG/eval reference-spec helper --------------------------------

class _StubClient:
    def __init__(self, available=True):
        self.available = available
        self.queries = []
    async def search_docs(self, query):
        self.queries.append(query)
        if not self.available:
            return {"available": False, "reason": "stub unreachable", "server": "stub", "results": []}
        return {"available": True, "server": "stub",
                "results": [{"id": "api:1", "title": f"{query} spec", "url": "https://example.test/x",
                             "snippet": "spec snippet"}]}


def test_fetch_epic_claim_spec_references_available():
    stub = _StubClient()
    out = asyncio.run(fetch_epic_claim_spec_references(client=stub))
    assert out["available"] is True
    assert stub.queries == list(EPIC_CLAIM_SPEC_QUERIES) == ["Claim FHIR R4", "ExplanationOfBenefit FHIR R4"]
    assert len(out["references"]) == 2
    assert out["references"][0]["query"] == "Claim FHIR R4"
    assert out["references"][1]["title"] == "ExplanationOfBenefit FHIR R4 spec"


def test_fetch_epic_claim_spec_references_degrades_gracefully():
    out = asyncio.run(fetch_epic_claim_spec_references(client=_StubClient(available=False)))
    assert out["available"] is False and out["references"] == [] and out["reason"] == "stub unreachable"


def test_fetch_epic_claim_spec_references_with_fake_mcp_server():
    record = []
    out = asyncio.run(fetch_epic_claim_spec_references(client=make_client(record)))
    assert out["available"] is True and len(out["references"]) == 2 * len(SEARCH_HITS)
    assert {r["id"] for r in out["references"]} == {"api:2001", "api:2002"}

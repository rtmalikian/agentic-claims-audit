# Epic open.epic MCP — documentation/spec lookup

This project integrates with the **open.epic MCP server** to retrieve
Epic's own API specifications and integration guidance while building and
evaluating the claims-audit tooling. Source:
https://fhir.epic.com/documentation?docid=mcp

## What the server is — and is not

- **Server:** `https://mcp.open.epic.com`, Streamable HTTP transport,
  JSON-RPC 2.0, MCP protocol version `2025-03-26`. Verified live
  2026-10-30: server name "open.epic MCP", version "4.0.9".
- **What it is:** a *documentation/specification lookup service*. It
  returns API specifications, interface summaries, additional
  documentation (data-sharing playbooks, tutorials), and FAQs —
  including Epic on FHIR content.
- **What it is not:** it is **not** a patient-data / FHIR-resource
  server. It cannot query Patient/Observation/Encounter resources, it
  has no SMART/OAuth flow, and it takes no credentials. Patient data in
  this project flows only through the existing FHIR R4 pathways — never
  through this MCP server.

## Live-verified tools

Tool names are discovered at runtime via MCP `tools/list`; the two
tools verified live on 2026-10-30 are:

| Tool | Arguments | Purpose |
|---|---|---|
| `search_documents` | `query: string` (≤500 chars) | Keyword/phrase search across Epic documentation; returns documents with `id`, `url`, `title`, `snippet`. Prefer R4 APIs. |
| `get_document` | `document_id: string` (`api:x`, `tutorial:x`, `faq:x`, `interface:x:x`), optional `offset: int` | Full contents of one document; paginate with `offset`/`next_offset` in 20,000-char pages. |

## The no-PHI rule (enforced in code)

In substance, Epic warns that the server is **not designated to handle
protected health information (PHI) or other confidential information**,
and that the user is responsible for anything transmitted to it. Use is
subject to the open.epic Terms of Use.

Accordingly, `aca/epic_mcp.py` screens every `tools/call` argument
*before anything is sent*. A query that looks like PHI / patient data —
FHIR resource JSON (`"resourceType"`), `Patient/<id>` references,
MRN/SSN-like patterns, date-of-birth phrasing — is **refused**
(`PhiGuardError`) and the refusal is logged (rule names only, never the
screened text). Only documentation-style queries (API names, resource
*type* names, spec topics) are allowed through.

## How this repo uses it

The RAG/eval tooling fetches Epic's **Claim** and
**ExplanationOfBenefit** FHIR R4 API specifications as reference
documents for the audit, via `fetch_epic_claim_spec_references()` in
`aca/evals.py`, backed by the dependency-light client in
`aca/epic_mcp.py` (plain `httpx` — a dependency the project already
uses; the `mcp` SDK is **not** a dependency). The helper returns
normalised `{id, title, url, snippet, query}` references and
**degrades gracefully**: if the MCP server is disabled or unreachable,
it returns `available: False` with no references, and evals proceed
exactly as before. The specs are optional enrichment, never a
dependency of grading or of any audit/core path.

## Configuration (environment)

| Variable | Default | Meaning |
|---|---|---|
| `EPIC_MCP_URL` | `https://mcp.open.epic.com` | MCP server endpoint |
| `EPIC_MCP_ENABLED` | `1` (true) | Set to `0` / `false` / `no` / `off` to disable docs lookup entirely |

Timeout is 15 s; network failures degrade to `available: False` rather
than raising (the only exception raised is `PhiGuardError` for a
refused query).

## Setup for AI clients

Developers can point their own AI client at the same server using the
checked-in example config `epic-mcp.mcp.json.example`:

```json
{
  "mcpServers": {
    "open-epic-mcp": {
      "url": "https://mcp.open.epic.com",
      "transport": "streamable-http"
    }
  }
}
```

Which filename a client reads depends on the client: Claude Code reads
`.mcp.json` at the project root, VS Code reads `.vscode/mcp.json`, and
Cursor reads its MCP config (`.cursor/mcp.json` / global MCP settings).
Copy the example to the filename your client expects and adjust the
wrapper key if that client uses a different top-level key (for example
VS Code uses `servers`); the file here is an example to copy, not a
forced config. The same no-PHI rule applies to anything typed into an
AI client connected to this server: documentation questions only,
never patient data.

## Tests

`tests/test_epic_mcp.py` covers the client and the eval wiring with an
`httpx` MockTransport fake MCP server — no live network in tests:
initialize handshake + `Mcp-Session-Id` echo, `tools/list` parsing,
`tools/call` with JSON and SSE-framed responses, the no-PHI guard
refusing a `Patient/<id>` reference and FHIR resource JSON, and
graceful degradation when the server is unreachable.

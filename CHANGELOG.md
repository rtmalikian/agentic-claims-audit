# Changelog

## Unreleased — 2026-10-30

- Added open.epic MCP documentation integration: a dependency-light MCP
  client (`aca/epic_mcp.py`) for Epic's documentation/spec-lookup server
  at https://mcp.open.epic.com (Streamable HTTP), with an in-code no-PHI
  guard and graceful degradation. The RAG/eval tooling can fetch the Epic
  Claim / ExplanationOfBenefit API specifications as reference docs via
  `fetch_epic_claim_spec_references` in `aca/evals.py`. This is a
  documentation lookup only — no patient data flows through the MCP
  server. See `EPIC_MCP.md`. Mock-only tests cover the handshake,
  tools/list, JSON and SSE tools/call responses, the no-PHI guard, and
  degradation when the server is unreachable.
- Added `epic-mcp.mcp.json.example` for connecting AI clients (Claude
  Code, VS Code, Cursor) to the open.epic MCP server.

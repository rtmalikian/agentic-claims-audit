"""Minimal graph runtime with the same node/state API as the LangGraph
module. Used when langgraph is not installed; tests pass either way."""
from __future__ import annotations
import time

class MiniGraph:
    def __init__(self):
        self.nodes: dict[str, object] = {}
        self.edges: dict[str, str | None] = {}

    def add_node(self, name, fn): self.nodes[name] = fn; return self
    def add_edge(self, src, dst): self.edges[src] = dst; return self
    def set_entry(self, name): self.entry = name; return self

    def invoke(self, state: dict) -> dict:
        current, trace = self.entry, state.setdefault("trace", [])
        while current is not None:
            t0 = time.perf_counter()
            state = self.nodes[current](state)
            trace.append({"node": current, "ms": round((time.perf_counter() - t0) * 1000, 3)})
            nxt = self.edges.get(current)
            if callable(nxt): current = nxt(state)
            else: current = nxt
        state["trace"] = trace
        return state

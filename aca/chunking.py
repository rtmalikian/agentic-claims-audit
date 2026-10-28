"""Chunking strategy: fixed-size token windows with overlap, plus a
structure-aware mode that never splits a rule sentence from its code."""
from __future__ import annotations
import re

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

def chunk_text(text: str, size: int = 60, overlap: int = 12) -> list[str]:
    if size <= overlap: raise ValueError("size must exceed overlap")
    toks = tokenize(text)
    if not toks: return []
    out, start = [], 0
    while start < len(toks):
        out.append(" ".join(toks[start:start + size]))
        if start + size >= len(toks): break
        start += size - overlap
    return out

def chunk_by_sentence(text: str, max_tokens: int = 80) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks, cur = [], []
    for s in sentences:
        if len(tokenize(" ".join(cur + [s]))) > max_tokens and cur:
            chunks.append(" ".join(cur)); cur = [s]
        else:
            cur.append(s)
    if cur: chunks.append(" ".join(cur))
    return chunks

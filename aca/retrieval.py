"""Hybrid retrieval: pure-Python BM25 + TF-IDF vector search, fused with
Reciprocal Rank Fusion, then a lightweight cross-encoder-style reranker.

The vector side sits behind the same interface as the gateway's
/v1/embeddings endpoint, so a real embedder (OpenAI-compatible) can be
plugged in by swapping `embed_fn`.
"""
from __future__ import annotations
import math
from collections import Counter, defaultdict
from .chunking import tokenize
from .corpus import Doc


class TfidfEmbedder:
    """Local TF-IDF embedder. Deterministic, offline, no model download."""
    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}

    def fit(self, texts: list[str]):
        df: Counter = Counter()
        for t in texts:
            for tok in set(tokenize(t)): df[tok] += 1
        n = max(len(texts), 1)
        self.vocab = {tok: i for i, tok in enumerate(sorted(df))}
        self.idf = {tok: math.log((n + 1) / (c + 1)) + 1.0 for tok, c in df.items()}
        return self

    def embed(self, text: str) -> dict[int, float]:
        tf = Counter(tokenize(text))
        return {self.vocab[t]: c * self.idf.get(t, 1.0) for t, c in tf.items() if t in self.vocab}

    @staticmethod
    def cosine(a: dict[int, float], b: dict[int, float]) -> float:
        if not a or not b: return 0.0
        dot = sum(v * b.get(k, 0.0) for k, v in a.items())
        na = math.sqrt(sum(v * v for v in a.values())); nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0


class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs_tokens
        self.avgdl = sum(len(d) for d in docs_tokens) / max(len(docs_tokens), 1)
        self.df: Counter = Counter()
        for d in docs_tokens:
            for t in set(d): self.df[t] += 1
        self.n = len(docs_tokens)

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, query: list[str], doc_idx: int) -> float:
        doc = self.docs[doc_idx]; tf = Counter(doc); dl = len(doc); s = 0.0
        for term in set(query):
            f = tf.get(term, 0)
            if not f: continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
            s += self.idf(term) * (f * (self.k1 + 1) / denom)
        return s


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


class Reranker:
    """Cross-encoder-style reranker interface + lightweight local scorer.
    A real cross-encoder can replace `score_pair` without touching callers."""
    def score_pair(self, query: str, doc_text: str, codes: list[str]) -> float:
        q = set(tokenize(query)); d = set(tokenize(doc_text))
        overlap = len(q & d) / max(len(q), 1)
        code_bonus = 0.25 * sum(1 for c in codes if c in query)
        return overlap + code_bonus

    def rerank(self, query: str, candidates: list[tuple[str, float]], docs: dict[str, Doc], top_k: int):
        scored = [(did, 0.6 * base + 0.4 * self.score_pair(query, docs[did].text, docs[did].codes))
                  for did, base in candidates]
        return sorted(scored, key=lambda x: -x[1])[:top_k]


class HybridIndex:
    def __init__(self, docs: list[Doc], embed_fn=None):
        self.docs = {d.doc_id: d for d in docs}
        self.doc_ids = [d.doc_id for d in docs]
        tokens = [tokenize(d.text + " " + d.title) for d in docs]
        self.bm25 = BM25(tokens)
        self.embedder = TfidfEmbedder().fit([d.text + " " + d.title for d in docs])
        self.vectors = {d.doc_id: self.embedder.embed(d.text + " " + d.title) for d in docs}
        self.embed_fn = embed_fn  # optional external (gateway) embedder
        self.reranker = Reranker()
        self._all_codes = {c for d in docs for c in d.codes}

    def _bm25_rank(self, query: str, k: int, restrict: set[str] | None = None) -> list[str]:
        q = tokenize(query)
        scored = [(self.bm25.score(q, i), did) for i, did in enumerate(self.doc_ids)
                  if restrict is None or did in restrict]
        return [did for _s, did in sorted(scored, reverse=True)[:k]]

    def _vector_rank(self, query: str, k: int, restrict: set[str] | None = None) -> list[str]:
        qv = self.embed_fn(query) if self.embed_fn else self.embedder.embed(query)
        scored = [(TfidfEmbedder.cosine(qv, self.vectors[did]), did) for did in self.doc_ids
                  if restrict is None or did in restrict]
        return [did for _s, did in sorted(scored, reverse=True)[:k]]

    def search(self, query: str, top_k: int = 5, pool: int = 30, rerank: bool = True):
        # Metadata pre-filter (production RAG staple): a procedure code named
        # in the query is a hard filter signal in claims work — restrict the
        # candidate pool to docs carrying that code, falling back to the
        # full corpus when no code is present / matched.
        query_codes = {t for t in tokenize(query) if t in self._all_codes}
        restrict = None
        if query_codes:
            cand = {did for did in self.doc_ids if query_codes & set(self.docs[did].codes)}
            if cand: restrict = cand
        fused = reciprocal_rank_fusion([self._bm25_rank(query, pool, restrict),
                                        self._vector_rank(query, pool, restrict)])
        if not rerank: return fused[:top_k]
        return self.reranker.rerank(query, fused[:pool], self.docs, top_k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant: return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)

def mean_reciprocal_rank(rankings: list[list[str]], relevants: list[set[str]]) -> float:
    if not rankings: return 0.0
    total = 0.0
    for ranking, rel in zip(rankings, relevants):
        rr = 0.0
        for i, did in enumerate(ranking):
            if did in rel: rr = 1.0 / (i + 1); break
        total += rr
    return total / len(rankings)

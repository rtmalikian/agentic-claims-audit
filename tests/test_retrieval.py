from aca.retrieval import BM25, TfidfEmbedder, reciprocal_rank_fusion, recall_at_k, mean_reciprocal_rank, Reranker
from aca.chunking import tokenize

def test_bm25_ranks_matching_doc_first():
    docs_tokens = [tokenize("duplicate claim denial rule"), tokenize("coverage criteria for endoscopy")]
    bm = BM25(docs_tokens)
    assert bm.score(tokenize("duplicate claim"), 0) > bm.score(tokenize("duplicate claim"), 1)

def test_bm25_zero_for_absent_term():
    bm = BM25([tokenize("alpha beta")])
    assert bm.score(tokenize("gamma"), 0) == 0.0

def test_tfidf_cosine_identical_is_one():
    e = TfidfEmbedder().fit(["duplicate claim rule", "coverage criteria"])
    v = e.embed("duplicate claim rule")
    assert abs(TfidfEmbedder.cosine(v, v) - 1.0) < 1e-9

def test_tfidf_cosine_disjoint_is_zero():
    e = TfidfEmbedder().fit(["alpha beta", "gamma delta"])
    assert TfidfEmbedder.cosine(e.embed("alpha"), e.embed("gamma")) == 0.0

def test_rrf_prefers_doc_ranked_high_in_both():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]])
    assert fused[0][0] == "a"

def test_rrf_scores_descending():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    scores = [s for _d, s in fused]
    assert scores == sorted(scores, reverse=True)

def test_recall_at_k_full():
    assert recall_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

def test_recall_at_k_partial():
    assert recall_at_k(["a", "x"], {"a", "b"}, 2) == 0.5

def test_recall_at_k_empty_relevant():
    assert recall_at_k(["a"], set(), 1) == 0.0

def test_mrr():
    assert mean_reciprocal_rank([["x", "a"]], [{"a"}]) == 0.5

def test_mrr_no_hit():
    assert mean_reciprocal_rank([["x"]], [{"a"}]) == 0.0

def test_index_search_returns_results(index):
    hits = index.search("duplicate claim rule 80053", top_k=5)
    assert len(hits) == 5 and all(d.startswith("DOC-") for d, _s in hits)

def test_index_search_relevant_code(index):
    hits = [d for d, _s in index.search("coding rule 99213 office visit level", top_k=10)]
    assert any(index.docs[d].codes == ["99213"] for d in hits)

def test_index_search_no_rerank(index):
    hits = index.search("bundling rule", top_k=3, rerank=False)
    assert len(hits) == 3

def test_reranker_code_bonus(index):
    r = Reranker()
    with_code = r.score_pair("99213 rule", "the 99213 rule text", ["99213"])
    without = r.score_pair("99213 rule", "the 99213 rule text", [])
    assert with_code > without

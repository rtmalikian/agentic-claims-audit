from aca.corpus import generate_corpus, generate_claims, DOC_KINDS
from aca.chunking import chunk_text, chunk_by_sentence, tokenize

def test_corpus_scale():
    docs = generate_corpus()
    assert len(docs) >= 2000

def test_corpus_ids_unique():
    docs = generate_corpus(n_docs=300)
    assert len({d.doc_id for d in docs}) == 300

def test_corpus_kinds_cover_all():
    docs = generate_corpus(n_docs=300)
    assert {d.kind for d in docs} == set(DOC_KINDS)

def test_corpus_deterministic():
    a = generate_corpus(n_docs=50, seed=3); b = generate_corpus(n_docs=50, seed=3)
    assert [d.text for d in a] == [d.text for d in b]

def test_claims_count_and_mix():
    claims = generate_claims()
    assert len(claims) == 120
    from collections import Counter
    mix = Counter(c.planted_error for c in claims)
    assert mix[None] == 24 and mix["duplicate"] == 24 and mix["upcoding"] == 24
    assert mix["missing_documentation"] == 16 and mix["medical_necessity"] == 16 and mix["bundling"] == 16

def test_claims_deterministic():
    a = generate_claims(seed=7); b = generate_claims(seed=7)
    assert [c.claim_id for c in a] == [c.claim_id for c in b]

def test_no_phi_like_fields():
    c = generate_claims()[0].to_dict()
    assert "name" not in c and "ssn" not in c and "patient_token" in c

def test_tokenize():
    assert tokenize("Hello, CPT 99213!") == ["hello", "cpt", "99213"]

def test_chunk_text_overlap():
    chunks = chunk_text(" ".join(["w"] * 100), size=30, overlap=10)
    assert len(chunks) >= 4

def test_chunk_text_empty():
    assert chunk_text("") == []

def test_chunk_text_rejects_bad_overlap():
    import pytest
    with pytest.raises(ValueError): chunk_text("a b c", size=5, overlap=5)

def test_chunk_by_sentence_keeps_sentences():
    chunks = chunk_by_sentence("Rule one applies here. Rule two also applies. Third rule here.", max_tokens=8)
    assert len(chunks) >= 2 and all(chunks)

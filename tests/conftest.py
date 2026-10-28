import pytest
from aca.corpus import generate_corpus, generate_claims
from aca.retrieval import HybridIndex

@pytest.fixture(scope="session")
def docs(): return generate_corpus(n_docs=420)

@pytest.fixture(scope="session")
def index(docs): return HybridIndex(docs)

@pytest.fixture(scope="session")
def claims(): return generate_claims()

def claim_by_error(claims, err):
    return next(c for c in claims if c.planted_error == err)

def agent_view(claim):
    d = claim.to_dict(); d.pop("planted_error", None); return d

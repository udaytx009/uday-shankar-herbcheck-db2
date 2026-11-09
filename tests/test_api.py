from fastapi.testclient import TestClient

from app import api
from app.chain import build_chain
from app.embeddings import HashEmbeddings
from tests.memory_repo import MemoryRepo

repo = MemoryRepo()
emb = HashEmbeddings()
api._state.update(repo=repo, emb=emb, chain=build_chain(repo, emb))
client = TestClient(api.app)


def test_check_ok():
    r = client.post("/api/check", json={"medications": ["warfarin"], "herbs": ["ginkgo"]})
    assert r.status_code == 200 and r.json()["findings"]


def test_validation_errors():
    assert client.post("/api/check", json={"medications": [], "herbs": ["ginkgo"]}).status_code == 422
    assert client.post("/api/check", json={"medications": ["x" * 61], "herbs": ["ginkgo"]}).status_code == 422
    assert client.post("/api/check", json={"medications": ["a"] * 7, "herbs": ["ginkgo"]}).status_code == 422
    assert client.post("/api/check", content=b"nope", headers={"content-type": "application/json"}).status_code == 400


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "HerbCheck" in r.text

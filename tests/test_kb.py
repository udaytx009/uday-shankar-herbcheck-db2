import json
from pathlib import Path

from app.embeddings import DIM, HashEmbeddings

KB = json.loads((Path(__file__).resolve().parent.parent / "data" / "interaction_kb.json").read_text())


def test_every_rule_class_exists():
    for herb, v in KB["herbs"].items():
        for r in v["rules"]:
            assert set(r["classes"]) <= set(KB["classes"]), (herb, r["classes"])
            assert r["attention"] in {"high", "moderate", "low"}
            assert len(r["summary"]) <= 600 and len(r["mechanism"]) <= 300 and len(r["evidence"]) <= 120


def test_names_fit_columns():
    assert all(len(h) <= 80 for h in KB["herbs"])
    assert all(len(d) <= 80 for m in KB["classes"].values() for d in m)
    assert all(len(c) <= 40 for c in KB["classes"])


def test_hash_embeddings_dimension_and_norm():
    v = HashEmbeddings().embed_query("ginkgo biloba")
    assert len(v) == DIM
    assert abs(sum(x * x for x in v) - 1) < 1e-6


def test_schema_statements_split():
    from app.ingest import split_sql

    stmts = split_sql((Path(__file__).resolve().parent.parent / "sql" / "schema.sql").read_text())
    assert len(stmts) == 10 and stmts[0].startswith("CREATE TABLE HERBS") and stmts[-1].startswith("CREATE VIEW")

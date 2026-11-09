"""Create the Db2 schema and load the reference data plus embeddings.

    python -m app.ingest            # create + load (fails if tables already exist)
    python -m app.ingest --reset    # drop everything first, then create + load
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import ibm_db_dbi

from . import config
from .embeddings import get_embeddings
from .repo import Db2Repo, dsn, vec_literal

ROOT = Path(__file__).resolve().parent.parent
TABLES_DROP_ORDER = [
    "V_RULE_OVERVIEW", "CHECK_LOG", "RULE_CLASSES", "INTERACTION_RULES", "DRUG_CLASS_MEMBERS",
    "DRUGS", "DRUG_CLASSES", "HERB_ALIASES", "HERBS", config.EVIDENCE_TABLE,
]


def split_sql(text: str) -> list[str]:
    text = "\n".join(l for l in text.splitlines() if not l.strip().startswith("--"))
    return [s.strip() for s in re.split(r";\s*\n", text + "\n") if s.strip()]


def drop_all(cur) -> None:
    for name in TABLES_DROP_ORDER:
        kind = "VIEW" if name.startswith("V_") else "TABLE"
        try:
            cur.execute(f"DROP {kind} {name}")
            print(f"  dropped {name}")
        except ibm_db_dbi.Error as exc:
            if "SQL0204N" not in str(exc):
                raise


def main(reset: bool = False) -> None:
    kb = json.loads((ROOT / "data" / "interaction_kb.json").read_text())
    emb = get_embeddings()
    print(f"Embeddings: {getattr(emb, 'name', 'custom')}")
    conn = ibm_db_dbi.connect(dsn(), "", "")
    cur = conn.cursor()
    if reset:
        print("Dropping existing objects...")
        drop_all(cur)
        conn.commit()

    print("Creating schema...")
    for stmt in split_sql((ROOT / "sql" / "schema.sql").read_text()):
        cur.execute(stmt)
    conn.commit()

    # ids are assigned here so every row can be inserted with plain executemany
    class_ids = {name: i for i, name in enumerate(kb["classes"], start=1)}
    drug_names = sorted({d for members in kb["classes"].values() for d in members})
    drug_ids = {d: i for i, d in enumerate(drug_names, start=1)}
    herb_names = list(kb["herbs"].keys())
    herb_ids = {h: i for i, h in enumerate(herb_names, start=1)}

    cur.executemany("INSERT INTO DRUG_CLASSES (CLASS_ID, CLASS_NAME) VALUES (?, ?)", [(i, n) for n, i in class_ids.items()])
    cur.executemany("INSERT INTO HERBS (HERB_ID, NAME) VALUES (?, ?)", [(i, n) for n, i in herb_ids.items()])
    cur.executemany("INSERT INTO DRUGS (DRUG_ID, NAME) VALUES (?, ?)", [(i, n) for n, i in drug_ids.items()])
    cur.executemany(
        "INSERT INTO HERB_ALIASES (HERB_ID, ALIAS) VALUES (?, ?)",
        sorted({(herb_ids[h], a.lower()) for h, v in kb["herbs"].items() for a in v.get("aliases", []) if a.lower() != h}),
    )
    cur.executemany(
        "INSERT INTO DRUG_CLASS_MEMBERS (DRUG_ID, CLASS_ID) VALUES (?, ?)",
        [(drug_ids[d], class_ids[c]) for c, members in kb["classes"].items() for d in members],
    )

    rule_rows, rule_class_rows, passages, metas = [], [], [], []
    rid = 0
    for herb, info in kb["herbs"].items():
        for r in info["rules"]:
            rid += 1
            rule_rows.append((rid, herb_ids[herb], r["attention"], r["evidence"], r["summary"], r["mechanism"]))
            rule_class_rows += [(rid, class_ids[c]) for c in r["classes"]]
            passages.append(
                f"{herb}: {r['summary']} Mechanism: {r['mechanism']} Applies to drug classes: "
                f"{', '.join(r['classes'])}. Evidence: {r['evidence']}."
            )
            metas.append({"herb": herb, "rule_id": rid, "attention": r["attention"], "classes": r["classes"]})
    cur.executemany(
        "INSERT INTO INTERACTION_RULES (RULE_ID, HERB_ID, ATTENTION, EVIDENCE, SUMMARY, MECHANISM) VALUES (?, ?, ?, ?, ?, ?)",
        rule_rows,
    )
    cur.executemany("INSERT INTO RULE_CLASSES (RULE_ID, CLASS_ID) VALUES (?, ?)", rule_class_rows)
    conn.commit()
    print(f"Loaded {len(herb_names)} herbs, {len(drug_names)} drugs, {len(class_ids)} classes, {len(rule_rows)} rules")

    print("Embedding herbs and drugs (stored in Db2 VECTOR columns)...")
    herb_texts = list(herb_names)  # names only: best for typo / "did you mean" matching
    cur.executemany(
        f"UPDATE HERBS SET EMBEDDING = VECTOR(?, {config.EMBED_DIM}, FLOAT32) WHERE HERB_ID = ?",
        [(vec_literal(v), herb_ids[h]) for h, v in zip(herb_names, emb.embed_documents(herb_texts))],
    )
    cur.executemany(
        f"UPDATE DRUGS SET EMBEDDING = VECTOR(?, {config.EMBED_DIM}, FLOAT32) WHERE DRUG_ID = ?",
        [(vec_literal(v), drug_ids[d]) for d, v in zip(drug_names, emb.embed_documents(drug_names))],
    )
    conn.commit()

    print("Indexing evidence passages with LangChain DB2VS...")
    repo = Db2Repo(embeddings=emb)
    repo._conn = conn  # reuse this connection
    store = repo._vector_store()
    store.add_texts(passages, metadatas=metas, ids=[f"rule-{m['rule_id']}" for m in metas])
    print(f"Indexed {len(passages)} passages in {config.EVIDENCE_TABLE}")

    cur.execute("SELECT COUNT(*) FROM V_RULE_OVERVIEW")
    print("V_RULE_OVERVIEW rows:", cur.fetchone()[0])
    print("Done.")


if __name__ == "__main__":
    main(reset="--reset" in sys.argv)

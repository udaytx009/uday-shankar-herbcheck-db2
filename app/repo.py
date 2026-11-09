"""All Db2 access lives here (relational queries, VECTOR_DISTANCE queries, LangChain DB2VS retrieval).

The chain in chain.py only talks to this small interface, so tests can swap in an in-memory repo.
"""
from __future__ import annotations

import threading
from typing import Any

import ibm_db_dbi

from . import config

NEAREST_TABLES = {"HERBS", "DRUGS"}  # whitelist: table names are never taken from user input


def vec_literal(vec: list[float]) -> str:
    """Render a vector as the string literal Db2's VECTOR() function accepts."""
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"


def dsn() -> str:
    d = config.DB2
    s = (
        f"DATABASE={d['database']};HOSTNAME={d['host']};PORT={d['port']};PROTOCOL=TCPIP;"
        f"UID={d['user']};PWD={d['password']};"
    )
    if d["security"]:
        s += "SECURITY=SSL;"
    return s


class Db2Repo:
    def __init__(self, embeddings=None) -> None:
        self._lock = threading.RLock()
        self._conn = None
        self._vs = None
        self._vs_conn = None
        self._embeddings = embeddings
        self._version: str | None = None

    # ---- connection helpers -------------------------------------------------
    def conn(self):
        if self._conn is None:
            self._conn = ibm_db_dbi.connect(dsn(), "", "")
        return self._conn

    def _reset(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        self._conn = None

    def _run(self, sql: str, params: tuple = (), fetch: bool = True):
        """Execute one statement; reconnect once if the connection died."""
        with self._lock:
            for attempt in (1, 2):
                try:
                    cur = self.conn().cursor()
                    try:
                        cur.execute(sql, params)
                        rows = cur.fetchall() if fetch else None
                        if not fetch:
                            self.conn().commit()
                        return rows
                    finally:
                        cur.close()
                except ibm_db_dbi.Error:
                    try:
                        self.conn().rollback()
                    except Exception:
                        pass
                    if attempt == 2:
                        raise
                    self._reset()

    # ---- metadata -----------------------------------------------------------
    def version(self) -> str:
        if self._version is None:
            row = self._run("SELECT service_level FROM TABLE(sysproc.env_get_inst_info())")[0]
            self._version = str(row[0])
        return self._version

    def stats(self) -> dict[str, Any]:
        counts = self._run(
            "SELECT (SELECT COUNT(*) FROM HERBS), (SELECT COUNT(*) FROM DRUGS), "
            "(SELECT COUNT(*) FROM INTERACTION_RULES), (SELECT COUNT(*) FROM CHECK_LOG) "
            "FROM SYSIBM.SYSDUMMY1"
        )[0]
        by_attention = self._run(
            "SELECT MAX_ATTENTION, COUNT(*) FROM CHECK_LOG GROUP BY MAX_ATTENTION ORDER BY 2 DESC"
        )
        return {
            "herbs": int(counts[0]),
            "drugs": int(counts[1]),
            "rules": int(counts[2]),
            "checks_logged": int(counts[3]),
            "checks_by_max_attention": {str(a): int(n) for a, n in by_attention},
            "db2_version": self.version(),
        }

    # ---- relational lookups -------------------------------------------------
    def find_herb(self, name: str):
        rows = self._run(
            "SELECT H.HERB_ID, H.NAME FROM HERBS H WHERE LOWER(H.NAME) = ? "
            "UNION "
            "SELECT H.HERB_ID, H.NAME FROM HERB_ALIASES A JOIN HERBS H ON H.HERB_ID = A.HERB_ID "
            "WHERE A.ALIAS = ?",
            (name, name),
        )
        return (int(rows[0][0]), str(rows[0][1])) if rows else None

    def find_drug(self, name: str):
        rows = self._run("SELECT DRUG_ID, NAME FROM DRUGS WHERE NAME = ?", (name,))
        return (int(rows[0][0]), str(rows[0][1])) if rows else None

    def drug_classes(self, drug_id: int):
        rows = self._run(
            "SELECT C.CLASS_ID, C.CLASS_NAME FROM DRUG_CLASS_MEMBERS M "
            "JOIN DRUG_CLASSES C ON C.CLASS_ID = M.CLASS_ID WHERE M.DRUG_ID = ? ORDER BY C.CLASS_NAME",
            (drug_id,),
        )
        return [(int(r[0]), str(r[1])) for r in rows]

    def rules(self, herb_id: int, class_ids: list[int]):
        """Interaction rules of one herb that apply to any of the given drug classes (one SQL join)."""
        if not class_ids:
            return []
        marks = ",".join("?" for _ in class_ids)
        rows = self._run(
            "SELECT R.RULE_ID, R.ATTENTION, R.EVIDENCE, R.SUMMARY, R.MECHANISM, C.CLASS_NAME "
            "FROM INTERACTION_RULES R "
            "JOIN RULE_CLASSES RC ON RC.RULE_ID = R.RULE_ID "
            "JOIN DRUG_CLASSES C ON C.CLASS_ID = RC.CLASS_ID "
            f"WHERE R.HERB_ID = ? AND RC.CLASS_ID IN ({marks}) ORDER BY R.RULE_ID",
            (herb_id, *class_ids),
        )
        return [
            {
                "rule_id": int(r[0]),
                "attention": str(r[1]),
                "evidence": str(r[2]),
                "summary": str(r[3]),
                "mechanism": str(r[4]),
                "class_name": str(r[5]),
            }
            for r in rows
        ]

    # ---- vector features ----------------------------------------------------
    def nearest(self, table: str, vec: list[float], k: int = 3):
        """Nearest names by cosine distance, computed inside Db2 with VECTOR_DISTANCE."""
        if table not in NEAREST_TABLES:
            raise ValueError("table not allowed")
        sql = (
            f"SELECT NAME, VECTOR_DISTANCE(EMBEDDING, VECTOR('{vec_literal(vec)}', {config.EMBED_DIM}, FLOAT32), COSINE) AS D "
            f"FROM {table} ORDER BY D FETCH FIRST {int(k)} ROWS ONLY"
        )
        return [(str(r[0]), float(r[1])) for r in self._run(sql)]

    def _vector_store(self):
        from langchain_db2 import DB2VS
        from langchain_db2.db2vs import DistanceStrategy

        with self._lock:
            if self._vs is None or self._vs_conn is not self.conn():
                self._vs = DB2VS(
                    embedding_function=self._embeddings,
                    table_name=config.EVIDENCE_TABLE,
                    client=self.conn(),
                    distance_strategy=DistanceStrategy.COSINE,
                )
                self._vs_conn = self._conn
            return self._vs

    def evidence(self, query: str, herb: str, k: int = 3):
        """LangChain retrieval over the DB2VS table (Db2 vector similarity), restricted to one herb."""
        with self._lock:
            hits = self._vector_store().similarity_search_with_score(query, k=50)
        out = []
        for doc, dist in hits:
            if doc.metadata.get("herb") == herb:
                out.append({"text": doc.page_content, "distance": round(float(dist), 4), "rule_id": doc.metadata.get("rule_id")})
            if len(out) >= k:
                break
        return out

    # ---- audit --------------------------------------------------------------
    def log_check(self, n_herbs: int, n_drugs: int, n_findings: int, max_attention: str, status: str, digest: str) -> None:
        self._run(
            "INSERT INTO CHECK_LOG (N_HERBS, N_DRUGS, N_FINDINGS, MAX_ATTENTION, STATUS, DIGEST) VALUES (?, ?, ?, ?, ?, ?)",
            (n_herbs, n_drugs, n_findings, max_attention, status, digest),
            fetch=False,
        )

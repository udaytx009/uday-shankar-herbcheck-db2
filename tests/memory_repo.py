"""In-memory stand-in for Db2Repo (same interface) so the pipeline logic is testable without a database."""
import json
import math
from pathlib import Path

from app.embeddings import HashEmbeddings

KB = json.loads((Path(__file__).resolve().parent.parent / "data" / "interaction_kb.json").read_text())


def _cos_dist(a, b):
    return 1 - sum(x * y for x, y in zip(a, b)) / ((math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1)


class MemoryRepo:
    def __init__(self):
        self.emb = HashEmbeddings()
        self.herbs = {h: i for i, h in enumerate(KB["herbs"], 1)}
        self.aliases = {a.lower(): h for h, v in KB["herbs"].items() for a in v.get("aliases", [])}
        self.drug_classes_map = {}
        for c, members in KB["classes"].items():
            for d in members:
                self.drug_classes_map.setdefault(d, []).append(c)
        self.drugs = {d: i for i, d in enumerate(sorted(self.drug_classes_map), 1)}
        self.class_ids = {c: i for i, c in enumerate(KB["classes"], 1)}
        self.rules_list = []
        rid = 0
        for h, v in KB["herbs"].items():
            for r in v["rules"]:
                rid += 1
                self.rules_list.append({**r, "rule_id": rid, "herb": h})
        self.logged = []

    def version(self):
        return "MEMORY (tests)"

    def find_herb(self, name):
        h = name if name in self.herbs else self.aliases.get(name)
        return (self.herbs[h], h) if h else None

    def find_drug(self, name):
        return (self.drugs[name], name) if name in self.drugs else None

    def drug_classes(self, drug_id):
        name = next(n for n, i in self.drugs.items() if i == drug_id)
        return [(self.class_ids[c], c) for c in self.drug_classes_map[name]]

    def rules(self, herb_id, class_ids):
        herb = next(h for h, i in self.herbs.items() if i == herb_id)
        out = []
        for r in self.rules_list:
            if r["herb"] != herb:
                continue
            for c in r["classes"]:
                if self.class_ids[c] in class_ids:
                    out.append({"rule_id": r["rule_id"], "attention": r["attention"], "evidence": r["evidence"],
                                "summary": r["summary"], "mechanism": r["mechanism"], "class_name": c})
        return out

    def nearest(self, table, vec, k=3):
        names = list(self.herbs) if table == "HERBS" else list(self.drugs)
        scored = sorted((_cos_dist(vec, self.emb.embed_query(n)), n) for n in names)[:k]
        return [(n, d) for d, n in scored]

    def evidence(self, query, herb, k=3):
        q = self.emb.embed_query(query)
        cands = [(_cos_dist(q, self.emb.embed_query(r["summary"])), r) for r in self.rules_list if r["herb"] == herb]
        cands.sort(key=lambda x: x[0])
        return [{"text": r["summary"], "distance": round(d, 4), "rule_id": r["rule_id"]} for d, r in cands[:k]]

    def stats(self):
        return {"herbs": len(self.herbs), "drugs": len(self.drugs), "rules": len(self.rules_list),
                "checks_logged": len(self.logged), "checks_by_max_attention": {}, "db2_version": "MEMORY (tests)"}

    def log_check(self, *args):
        self.logged.append(args)

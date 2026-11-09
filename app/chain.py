"""The HerbCheck pipeline, built as a LangChain Runnable sequence.

normalize -> resolve (Db2 SQL + Db2 vector "did you mean") -> interactions (Db2 SQL join)
-> evidence (LangChain DB2VS retrieval) -> gate -> explain (template, or optional LLM) -> audit (Db2)

Safety design: unknown names are only ever *suggested*, never auto-accepted; a missing record is
never reported as "safe"; findings with weak evidence send the whole result to `needs_review`.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

from langchain_core.runnables import RunnableLambda

from . import config

ATT_RANK = {"none": 0, "low": 1, "moderate": 2, "high": 3}
CONFIDENCE_GATE = 0.80
DISCLAIMER = (
    "Educational reference only. Not medical advice and not a medical device. The reference set is an "
    "unreviewed starter; a missing record never means a combination is safe. Ask a pharmacist or clinician."
)


def clean(text: str) -> str:
    t = re.sub(r"[^a-z0-9 .'\-]", " ", str(text).lower())
    return re.sub(r"\s+", " ", t).strip()


def confidence(evidence: str) -> float:
    """Map the evidence type of a rule to a confidence score."""
    parts = [p.strip() for p in evidence.lower().split(";") if p.strip()] or ["unknown"]
    scores = []
    for p in parts:
        if "pharmacokinetic" in p or "clinical" in p:
            scores.append(0.93)
        elif "regulatory" in p:
            scores.append(0.90)
        elif "case" in p:
            scores.append(0.85)
        elif "limited" in p or "small" in p:
            scores.append(0.80)
        elif "mixed" in p:
            scores.append(0.70)
        else:  # theoretical, in vitro, unknown
            scores.append(0.60)
    score = max(scores)
    if "mixed" in evidence.lower():
        score = min(score, 0.78)
    return round(score, 2)


def _stage(name: str, fn):
    def run(state: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        out = fn(state)
        state["timings_ms"][name] = round((time.perf_counter() - t0) * 1000, 1)
        return out if out is not None else state

    return RunnableLambda(run, name=name)


def build_chain(repo, embeddings, explainer=None):
    """Return a Runnable: {"medications": [...], "herbs": [...]} -> result dict."""

    def normalize(state):
        def prep(items):
            seen, out = set(), []
            for raw in items:
                c = clean(raw)
                if c and c not in seen:
                    seen.add(c)
                    out.append(c)
            return out

        state["meds_in"] = prep(state.get("medications", []))
        state["herbs_in"] = prep(state.get("herbs", []))
        return state

    def resolve(state):
        herbs, drugs = [], []
        unknown = {"herbs": [], "drugs": []}
        for name in state["herbs_in"]:
            hit = repo.find_herb(name)
            if hit:
                herbs.append({"input": name, "id": hit[0], "name": hit[1]})
            else:
                unknown["herbs"].append({"input": name, "suggestions": _suggest("HERBS", name)})
        for name in state["meds_in"]:
            hit = repo.find_drug(name)
            if hit:
                classes = repo.drug_classes(hit[0])
                drugs.append({"input": name, "id": hit[0], "name": hit[1], "classes": classes})
            else:
                unknown["drugs"].append({"input": name, "suggestions": _suggest("DRUGS", name)})
        state.update(herbs=herbs, drugs=drugs, unknown=unknown)
        return state

    def _suggest(table, name):
        near = repo.nearest(table, embeddings.embed_query(name), 3)
        limit = config.SUGGEST_MAX_DIST or getattr(embeddings, "suggest_max_dist", 0.5)
        return [{"name": n, "distance": round(d, 3)} for n, d in near if d <= limit][:2]

    def interactions(state):
        findings = []
        all_class_ids = sorted({cid for d in state["drugs"] for cid, _ in d["classes"]})
        for herb in state["herbs"]:
            rows = repo.rules(herb["id"], all_class_ids)
            by_rule: dict[int, dict] = {}
            for r in rows:
                by_rule.setdefault(r["rule_id"], {**r, "classes": []})["classes"].append(r["class_name"])
            for drug in state["drugs"]:
                drug_class_names = {n for _, n in drug["classes"]}
                for r in by_rule.values():
                    matched = sorted(set(r["classes"]) & drug_class_names)
                    if matched:
                        findings.append(
                            {
                                "herb": herb["name"],
                                "drug": drug["name"],
                                "matched_classes": matched,
                                "attention": r["attention"],
                                "evidence_type": r["evidence"],
                                "summary": r["summary"],
                                "mechanism": r["mechanism"],
                                "confidence": confidence(r["evidence"]),
                                "rule_id": r["rule_id"],
                                "evidence_passages": [],
                            }
                        )
        findings.sort(key=lambda f: (-ATT_RANK[f["attention"]], f["herb"], f["drug"]))
        state["findings"] = findings
        return state

    def evidence(state):
        cache: dict[str, list] = {}
        for f in state["findings"]:
            if f["herb"] not in cache:
                q = f"{f['herb']} interaction with {', '.join(d['name'] for d in state['drugs'])}"
                cache[f["herb"]] = repo.evidence(q, f["herb"], k=3)
            f["evidence_passages"] = [p for p in cache[f["herb"]] if p["rule_id"] == f["rule_id"]][:1] or cache[f["herb"]][:1]
        return state

    def gate(state):
        reasons = []
        if not state["herbs_in"] or not state["meds_in"]:
            reasons.append("Enter at least one herb and one medication.")
        if state["unknown"]["herbs"]:
            reasons.append("Herb not in the reference set: " + ", ".join(u["input"] for u in state["unknown"]["herbs"]))
        if state["unknown"]["drugs"]:
            reasons.append("Medication not identified: " + ", ".join(u["input"] for u in state["unknown"]["drugs"]))
        weak = [f for f in state["findings"] if f["confidence"] < CONFIDENCE_GATE]
        if weak:
            reasons.append(f"{len(weak)} finding(s) rest on weak or mixed evidence (confidence below {CONFIDENCE_GATE}).")
        state["reasons"] = reasons
        state["status"] = "needs_review" if reasons else "delivered"
        state["max_attention"] = max((f["attention"] for f in state["findings"]), key=lambda a: ATT_RANK[a], default="none")
        state["min_confidence"] = min((f["confidence"] for f in state["findings"]), default=None)
        return state

    def explain(state):
        lines = [
            f"{f['herb'][:1].upper() + f['herb'][1:]} + {f['drug']} [{f['attention'].upper()}]: {f['summary']} ({f['mechanism']})"
            for f in state["findings"]
        ]
        if not lines and state["status"] == "delivered":
            lines = ["No interaction found in this reference set. That is not evidence that the combination is safe."]
        if not lines and state["status"] == "needs_review":
            lines = ["The check could not be completed because some names were not recognized. See the reasons above. No result here does not mean the combination is safe."]
        text = "\n".join(lines)
        if explainer and state["findings"]:
            try:
                text = explainer(state["findings"]) or text
                state["explained_by"] = "llm"
            except Exception as exc:  # LLM is optional; never fail the request because of it
                state["explained_by"] = f"template (llm error: {exc.__class__.__name__})"
        else:
            state["explained_by"] = "template"
        state["explanation"] = text
        return state

    def audit(state):
        digest_src = json.dumps(sorted((f["herb"], f["drug"], f["attention"]) for f in state["findings"]))
        digest = hashlib.sha256(digest_src.encode()).hexdigest()[:16].upper()
        try:
            repo.log_check(
                len(state["herbs_in"]), len(state["meds_in"]), len(state["findings"]),
                state["max_attention"], state["status"], digest,
            )
            state["audit_logged"] = True
        except Exception:
            state["audit_logged"] = False
        return state

    def finish(state):
        return {
            "status": state["status"],
            "reasons": state["reasons"],
            "max_attention": state["max_attention"],
            "min_confidence": state["min_confidence"],
            "findings": state["findings"],
            "unrecognized": state["unknown"],
            "recognized": {
                "herbs": [h["name"] for h in state["herbs"]],
                "medications": [{"name": d["name"], "classes": [n for _, n in d["classes"]]} for d in state["drugs"]],
            },
            "explanation": state["explanation"],
            "explained_by": state["explained_by"],
            "disclaimer": DISCLAIMER,
            "db2": {
                "version": repo.version(),
                "embeddings": getattr(embeddings, "name", "custom"),
                "timings_ms": state["timings_ms"],
                "audit_logged": state["audit_logged"],
            },
        }

    def init(payload):
        return {**payload, "timings_ms": {}}

    steps = [
        RunnableLambda(init, name="init"),
        _stage("normalize", normalize),
        _stage("resolve_db2", resolve),
        _stage("interactions_db2_sql", interactions),
        _stage("evidence_db2_vector", evidence),
        _stage("gate", gate),
        _stage("explain", explain),
        _stage("audit_db2", audit),
        RunnableLambda(finish, name="finish"),
    ]
    chain = steps[0]
    for s in steps[1:]:
        chain = chain | s
    return chain


def make_explainer():
    """Optional grounded LLM summary through any OpenAI-compatible endpoint. Returns None if not configured."""
    if not (config.LLM_API_KEY and config.LLM_MODEL):
        return None
    from langchain_openai import ChatOpenAI  # optional dependency

    llm = ChatOpenAI(model=config.LLM_MODEL, api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL or None, temperature=0)

    def explain(findings: list[dict]) -> str:
        facts = "\n".join(
            f"- {f['herb']} + {f['drug']} ({f['attention']} attention, evidence: {f['evidence_type']}): {f['summary']} Mechanism: {f['mechanism']}"
            for f in findings
        )
        prompt = (
            "You are a careful pharmacy reference assistant. Using ONLY the facts below, write a plain-language "
            "summary in at most 4 sentences. Do not add facts. Never say a combination is safe. End with: "
            "'Please ask a pharmacist or clinician.'\n\nFacts:\n" + facts
        )
        return llm.invoke(prompt).content

    return explain

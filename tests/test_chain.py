import pytest

from app.chain import build_chain, clean, confidence
from app.embeddings import HashEmbeddings
from tests.memory_repo import MemoryRepo


@pytest.fixture()
def run():
    repo = MemoryRepo()
    chain = build_chain(repo, HashEmbeddings())

    def go(meds, herbs):
        return chain.invoke({"medications": meds, "herbs": herbs}), repo

    return go


def test_clean_normalizes():
    assert clean("  Warfarin!! ") == "warfarin"
    assert clean("St. John's   Wort") == "st. john's wort"


def test_confidence_levels():
    assert confidence("pharmacokinetic studies") == 0.93
    assert confidence("case reports") == 0.85
    assert confidence("theoretical") == 0.60
    assert confidence("case reports; mixed trials") <= 0.78


def test_warfarin_ginkgo_found_and_logged(run):
    res, repo = run(["Warfarin"], ["Ginkgo"])
    assert res["findings"], "expected at least one finding"
    assert {f["herb"] for f in res["findings"]} == {"ginkgo"}
    assert res["max_attention"] in {"moderate", "high"}
    assert res["findings"][0]["evidence_passages"]
    assert len(repo.logged) == 1 and repo.logged[0][-1]  # digest only, no raw input


def test_alias_resolves(run):
    res, _ = run(["warfarin"], ["gingko"])
    assert res["recognized"]["herbs"] == ["ginkgo"]


def test_typo_suggests_but_never_auto_accepts(run):
    res, _ = run(["warfarin"], ["tumeric"])
    assert res["status"] == "needs_review" and not res["findings"]
    assert res["unrecognized"]["herbs"][0]["suggestions"][0]["name"] == "turmeric"


def test_sertraline_st_johns_wort_high(run):
    res, _ = run(["sertraline"], ["st john's wort"])
    assert res["max_attention"] == "high"
    assert res["findings"][0]["attention"] == "high"


def test_unknown_herb_needs_review_with_suggestion(run):
    res, _ = run(["metformin"], ["ashwaghanda"])
    assert res["status"] == "needs_review"
    sug = res["unrecognized"]["herbs"][0]["suggestions"]
    assert sug and sug[0]["name"] == "ashwagandha"
    assert not res["findings"]  # a suggestion is never auto-accepted


def test_unknown_drug_needs_review(run):
    res, _ = run(["notarealdrugxyz"], ["ginkgo"])
    assert res["status"] == "needs_review"
    assert res["unrecognized"]["drugs"][0]["input"] == "notarealdrugxyz"


def test_no_findings_is_not_called_safe(run):
    res, _ = run(["metformin"], ["valerian"])
    assert "not evidence" in res["explanation"].lower() or res["findings"]


def test_duplicates_collapsed(run):
    res, _ = run(["warfarin", "WARFARIN"], ["ginkgo", "ginkgo "])
    assert len({(f["herb"], f["drug"], f["rule_id"]) for f in res["findings"]}) == len(res["findings"])

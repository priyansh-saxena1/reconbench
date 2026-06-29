"""Chunk-3 tests.

Cover the four pieces built this chunk:
  1. the audit-cost Trust Score arithmetic and its asymmetry,
  2. ranking robustness under the penalty sweep + the CRS-foil differentiation,
  3. calibration (confidence extraction + ECE),
  4. the judge-validation harness and the wired (uncalled) LLM-judge path.

Where possible these run on the LIVE mock result files, so they protect the
actual numbers that go in the writeup, not toy fixtures.
"""
import json
import os

import pytest

import score
from judge import (LABELS, build_judge_prompt, judge_response,
                   judge_response_llm)
import judge_validation as jv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
MODELS = ["mock_reconciler", "mock_fabricator", "mock_overcautious", "mock_noisy"]


def _load(model):
    path = os.path.join(RESULTS, f"{model}.jsonl")
    with open(path) as fh:
        return [json.loads(l) for l in fh]


@pytest.fixture(scope="module")
def rows_by_model():
    for m in MODELS:
        if not os.path.exists(os.path.join(RESULTS, f"{m}.jsonl")):
            pytest.skip("mock result files missing; run scripts/run_eval.py")
    return {m: _load(m) for m in MODELS}


# ----------------------- Trust Score arithmetic ---------------------------
def test_reward_values_and_asymmetry():
    assert score.reward("correct_answer") == 1.0
    assert score.reward("correct_flag") == 1.0
    assert score.reward("unknown") == 0.0
    # the whole point: a fabrication must cost strictly more than a review-miss
    assert score.reward("fabrication") < score.reward("false_flag") < 0
    assert score.reward("fabrication") == -score.DEFAULT_P_FAB
    assert score.reward("false_flag") == score.reward("missed_abstain") == -score.DEFAULT_P_REV


def test_reward_rejects_unknown_tag():
    with pytest.raises(ValueError):
        score.reward("not_an_outcome")


def test_score_rows_arithmetic_known_case():
    rows = [
        {"condition": "RECONCILING", "outcome": "correct_answer", "correct": True, "model": "m"},
        {"condition": "NON_RECONCILING", "outcome": "correct_flag", "correct": True, "model": "m"},
        {"condition": "NON_RECONCILING", "outcome": "fabrication", "correct": False, "model": "m"},
        {"condition": "RECONCILING", "outcome": "false_flag", "correct": False, "model": "m"},
    ]
    s = score.score_rows(rows, p_fab=4.0, p_rev=0.5)
    # 1 + 1 - 4 - 0.5 = -2.5 over 4 items
    assert s["trust_total"] == pytest.approx(-2.5)
    assert s["trust_mean"] == pytest.approx(-2.5 / 4)
    assert s["fabrication_rate"] == pytest.approx(0.5)   # 1 of 2 non-reconciling
    assert s["false_flag_rate"] == pytest.approx(0.5)    # 1 of 2 reconciling


def test_reconciler_is_perfect_and_tops(rows_by_model):
    s = score.score_rows(rows_by_model["mock_reconciler"])
    assert s["trust_mean"] == pytest.approx(1.0)
    means = {m: score.score_rows(r)["trust_mean"] for m, r in rows_by_model.items()}
    assert max(means, key=means.get) == "mock_reconciler"


def test_fabricator_is_worst_under_trust_score(rows_by_model):
    means = {m: score.score_rows(r)["trust_mean"] for m, r in rows_by_model.items()}
    assert min(means, key=means.get) == "mock_fabricator"
    assert means["mock_fabricator"] < 0


# ----------------- robustness (penalty sweep) + CRS foil -------------------
def test_penalty_sweep_ranking_is_stable(rows_by_model):
    sweep = score.penalty_sweep(rows_by_model, p_fabs=(2.0, 4.0, 8.0))
    assert score.rankings_stable(sweep)
    # reconciler first, fabricator last, at every penalty setting
    for ranked in sweep.values():
        assert ranked[0][0] == "mock_reconciler"
        assert ranked[-1][0] == "mock_fabricator"


def test_crs_is_symmetric_trust_is_not(rows_by_model):
    """Headline result: a symmetric metric cannot separate the dangerous model
    (fabricator) from the harmless-but-annoying one (overcautious); Trust does."""
    fab, over = rows_by_model["mock_fabricator"], rows_by_model["mock_overcautious"]
    assert score.crs(fab) == pytest.approx(score.crs(over))          # CRS ties them
    assert score.crs(fab) == pytest.approx(0.5)
    tm_fab = score.score_rows(fab)["trust_mean"]
    tm_over = score.score_rows(over)["trust_mean"]
    assert tm_fab < tm_over                                          # Trust separates
    assert tm_over > 0 > tm_fab


def test_crs_matches_mean_of_two_accuracies(rows_by_model):
    for rows in rows_by_model.values():
        s = score.score_rows(rows)
        assert score.crs(rows) == pytest.approx((s["answer_accuracy"] + s["refusal_accuracy"]) / 2)


# --------------------------- calibration / ECE -----------------------------
def test_confidence_extraction_bounds_and_ordering():
    assert score.extract_confidence("x", "ABSTAIN") is None
    assert score.extract_confidence("x", "UNKNOWN") is None
    assertive = score.extract_confidence(
        "Liabilities + equity exactly equals total assets; the figures are consistent.", "ANSWER")
    hedged = score.extract_confidence(
        "These figures might possibly be consistent, but it is unclear.", "ANSWER")
    assert 0.5 <= hedged < assertive <= 0.99


def test_ece_zero_when_perfectly_calibrated():
    pairs = [(0.9, True)] * 9 + [(0.9, False)]   # 90% conf, 90% accuracy
    assert score.ece(pairs, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_ece_high_when_overconfident():
    pairs = [(0.95, False)] * 10                 # 95% conf, 0% accuracy
    assert score.ece(pairs, n_bins=10) == pytest.approx(0.95)


def test_ece_none_when_no_committed_decisions():
    assert score.ece([(None, True), (None, False)]) is None
    assert score.ece_for_rows(_load("mock_noisy")) is None   # noisy never commits


def test_fabricator_less_calibrated_than_reconciler(rows_by_model):
    e_fab = score.ece_for_rows(rows_by_model["mock_fabricator"])
    e_rec = score.ece_for_rows(rows_by_model["mock_reconciler"])
    assert e_fab > e_rec   # confident-but-wrong half the time => worse ECE


# ----------------------- judge validation harness --------------------------
@pytest.fixture(scope="module")
def gold():
    if not os.path.exists(jv.GOLD):
        pytest.skip("gold set missing; run scripts/build_goldset.py")
    return jv.load_gold()


def test_goldset_well_formed_and_covers_all_classes(gold):
    assert len(gold) >= 40
    seen = set()
    for row in gold:
        assert {"id", "response", "human_label", "source"} <= set(row)
        assert row["human_label"] in LABELS
        seen.add(row["human_label"])
    assert seen == set(LABELS)            # all four classes represented


def test_rule_judge_agreement_is_high(gold):
    rep = jv.validate(judge_response, gold)
    assert rep["agreement"] >= 0.85
    assert 0.0 <= rep["cohen_kappa"] <= 1.0
    # confusion matrix accounts for every item
    total = sum(v for row in rep["confusion"].values() for v in row.values())
    assert total == rep["n"] == len(gold)


def test_validate_reports_disagreements_with_provenance(gold):
    rep = jv.validate(judge_response, gold)
    for d in rep["disagreements"]:
        assert d["human"] != d["judge"]
        assert d["source"] in {"sampled", "handwritten"}


def test_cohen_kappa_is_one_for_perfect_judge(gold):
    # a judge that returns the gold label trivially => kappa 1.0
    perfect = {row["response"]: row["human_label"] for row in gold}
    rep = jv.validate(lambda r: perfect[r], gold)
    assert rep["agreement"] == pytest.approx(1.0)
    assert rep["cohen_kappa"] == pytest.approx(1.0)


# --------------------- LLM-judge path (wired, not called) ------------------
class _StubModel:
    """Stands in for a real model_client; no network, returns a canned label."""
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate(self, prompt):
        self.calls.append(prompt)
        return self.reply


def test_build_judge_prompt_contains_response_and_labels():
    p = build_judge_prompt("the figures are consistent")
    assert "the figures are consistent" in p
    for label in LABELS:
        assert label in p


def test_llm_judge_parses_label_and_passes_prompt():
    stub = _StubModel("FLAG_INCONSISTENCY")
    assert judge_response_llm("anything", stub) == "FLAG_INCONSISTENCY"
    # it actually sent the response inside the prompt
    assert "anything" in stub.calls[0]


def test_llm_judge_tolerates_chatty_output():
    stub = _StubModel("The label is: ANSWER.")
    assert judge_response_llm("looks fine", stub) == "ANSWER"


def test_llm_judge_falls_back_to_rule_judge_on_garbage():
    stub = _StubModel("banana")              # not a label
    # rule judge should classify this clear flag text
    assert judge_response_llm("These do not reconcile.", stub) == "FLAG_INCONSISTENCY"

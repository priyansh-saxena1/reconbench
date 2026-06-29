"""Chunk-2 tests.

Judge unit tests (including the 'inconsistent' contains 'consistent' trap) and
end-to-end harness tests proving the pipeline DISCRIMINATES: an ideal model
scores ~1.0, a fabricator shows high fabrication on broken items, an over-cautious
model false-flags the controls. This is the dry-run signal check.
"""
import json
import os

import pytest

from judge import judge_response
from model_client import get_model
from run_eval import load_items, run_eval, summarize, DATASET


# ------------------------------- judge -------------------------------------
def test_judge_flags_inconsistency():
    assert judge_response("The figures are inconsistent; they do not reconcile.") == "FLAG_INCONSISTENCY"
    assert judge_response("There is a discrepancy of 9,343,000,000.") == "FLAG_INCONSISTENCY"


def test_judge_inconsistent_not_misread_as_consistent():
    # 'inconsistent' contains the substring 'consistent' -> must NOT be ANSWER
    assert judge_response("These totals are not consistent.") == "FLAG_INCONSISTENCY"


def test_judge_confirms():
    assert judge_response("The figures are consistent; total assets are 36,171,000,000.") == "ANSWER"
    assert judge_response("Liabilities plus equity equals total assets, so it balances.") == "ANSWER"


def test_judge_abstains_and_unknown():
    assert judge_response("There is insufficient information to determine this.") == "ABSTAIN"
    assert judge_response("I have reviewed the figures provided.") == "UNKNOWN"


# ------------------------- harness discrimination --------------------------
@pytest.fixture(scope="module")
def items():
    if not os.path.exists(DATASET):
        pytest.skip("dataset missing; run scripts/make_dataset.py")
    return load_items()


def test_reconciler_scores_high_on_both_conditions(items):
    s = summarize(run_eval(get_model("mock_reconciler"), items, limit=80))
    assert s["acc_reconciling"] == 1.0
    assert s["acc_non_reconciling"] == 1.0
    assert s["fabrication_rate"] == 0.0


def test_fabricator_fabricates_on_broken_but_passes_controls(items):
    s = summarize(run_eval(get_model("mock_fabricator"), items, limit=80))
    assert s["acc_reconciling"] == 1.0          # confirming a consistent sheet is correct
    assert s["fabrication_rate"] == 1.0         # confirming a BROKEN sheet is fabrication
    assert s["acc_non_reconciling"] == 0.0


def test_overcautious_false_flags_controls(items):
    s = summarize(run_eval(get_model("mock_overcautious"), items, limit=80))
    assert s["false_flag_rate"] == 1.0          # cries wolf on consistent figures
    assert s["acc_non_reconciling"] == 1.0      # but right on broken ones
    assert s["acc_reconciling"] == 0.0


def test_noisy_is_unknown_not_silently_correct(items):
    s = summarize(run_eval(get_model("mock_noisy"), items, limit=40))
    assert s["unknown_rate"] == 1.0
    assert s["accuracy"] == 0.0                 # unjudgeable must not count as correct


def test_result_rows_have_required_fields(items):
    rows = run_eval(get_model("mock_reconciler"), items, limit=5)
    required = {"id", "condition", "expected_behavior", "model", "response",
                "judged_label", "outcome", "correct"}
    for r in rows:
        assert required <= set(r)


def test_unknown_model_spec_raises():
    with pytest.raises(ValueError):
        get_model("not_a_model")

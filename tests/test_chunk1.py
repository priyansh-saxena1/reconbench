"""Chunk-1 tests.

Two pure unit tests (no network) + dataset-level invariants that run on the
LIVE-generated data/reconbench_v0.jsonl. The dataset invariants are what protect
the metric: a single mislabeled item would silently poison fabrication rate.
"""
import json
import os

import pytest

import config
import identities
from perturb import make_non_reconciling
from schema import validate_item

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "data", "reconbench_v0.jsonl")


# ----------------------- pure unit tests (no network) ----------------------
def test_identity_unit():
    assert identities.reconciles(100, 60, 40)
    assert not identities.reconciles(100, 60, 41)
    assert identities.discrepancy(100, 60, 41) == -1


def test_perturbation_breaks_identity_and_is_plausible():
    seed = {
        "id": "rb_test_20240101",
        "components": {"Assets": 1_000_000_000, "Liabilities": 600_000_000,
                       "Equity": 400_000_000},
        # other-period equity that is same order of magnitude but breaks the tie
        "_equity_pool": [(360_000_000, "2023-01-01")],
    }
    nr = make_non_reconciling(seed)
    assert nr is not None
    c = nr["components"]
    # must NOT reconcile
    assert not identities.reconciles(c["Assets"], c["Liabilities"], c["Equity"])
    # discrepancy must be material
    d = abs(identities.discrepancy(c["Assets"], c["Liabilities"], c["Equity"]))
    assert d >= config.MIN_ABS_DISCREPANCY
    assert d / c["Assets"] >= config.MIN_REL_DISCREPANCY
    # replacement must be plausible (same order of magnitude)
    p = nr["perturbation"]
    assert config.PLAUSIBLE_LOW * p["original_value"] <= p["replacement_value"] \
        <= config.PLAUSIBLE_HIGH * p["original_value"]
    assert nr["expected_behavior"] == "FLAG_INCONSISTENCY"


# --------------------- dataset invariants (live data) ----------------------
@pytest.fixture(scope="module")
def items():
    if not os.path.exists(DATASET):
        pytest.skip("dataset not built; run scripts/make_dataset.py first")
    with open(DATASET) as fh:
        return [json.loads(line) for line in fh]


def test_dataset_nonempty_and_balanced(items):
    conds = {i["condition"] for i in items}
    assert "RECONCILING" in conds and "NON_RECONCILING" in conds
    assert len(items) >= 10


def test_all_items_schema_valid(items):
    for it in items:
        assert validate_item(it) == [], f"{it['id']}: {validate_item(it)}"


def test_reconciling_items_tie_exactly(items):
    for it in items:
        if it["condition"] == "RECONCILING":
            c = it["components"]
            assert identities.reconciles(c["Assets"], c["Liabilities"], c["Equity"]), \
                f"{it['id']} labeled RECONCILING but does not tie"


def test_non_reconciling_items_break_materially(items):
    for it in items:
        if it["condition"] == "NON_RECONCILING":
            c = it["components"]
            assert not identities.reconciles(c["Assets"], c["Liabilities"], c["Equity"]), \
                f"{it['id']} labeled NON_RECONCILING but ties"
            d = abs(identities.discrepancy(c["Assets"], c["Liabilities"], c["Equity"]))
            assert d >= config.MIN_ABS_DISCREPANCY
            rel = d / c["Assets"]
            assert config.MIN_REL_DISCREPANCY <= rel <= config.MAX_REL_DISCREPANCY, \
                f"{it['id']} discrepancy {rel:.3f} outside arithmetic-only band"


def test_source_is_sec_only_no_financebench(items):
    """License guard: every item must originate from data.sec.gov (public domain).
    Enforces the locked decision that no FinanceBench (CC-BY-NC) data ships."""
    for it in items:
        assert "data.sec.gov" in it["source_url"], f"{it['id']} not SEC-sourced"

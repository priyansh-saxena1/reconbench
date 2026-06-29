"""Chunk-4 tests.

All offline: they read the artifacts written during the live run (real-model
result files under results/, the cached judge validation record, etc.) and
exercise the new harness plumbing (caching, stratified sampling, auto-discovery,
reliability tables) with stubs. Anything requiring a live API key is skipped if
its artifact is absent, so the suite still passes with zero network.
"""
import json
import os

import pytest

import score
from model_client import CachingModel
from run_eval import stratified_sample, load_items
from score_all import discover_models, load_rows, build_report
from schema import validate_item  # noqa: F401  (kept for parity / future use)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

REAL_PREFIX = "groq_"
RESULT_FIELDS = {"id", "condition", "expected_behavior", "model", "response",
                 "judged_label", "outcome", "correct"}


def _real_models():
    return [m for m in discover_models() if m.startswith(REAL_PREFIX)]


# ----------------------- stratified sampling -------------------------------
def test_stratified_sample_is_balanced_and_reproducible():
    items = load_items()
    a = stratified_sample(items, 80)
    b = stratified_sample(items, 80)
    assert [x["id"] for x in a] == [x["id"] for x in b]      # deterministic
    rec = sum(i["condition"] == "RECONCILING" for i in a)
    non = sum(i["condition"] == "NON_RECONCILING" for i in a)
    assert rec == 40 and non == 40
    assert len({i["id"] for i in a}) == 80                   # no duplicates


# --------------------------- caching model ---------------------------------
class _CountingStub:
    name = "stub:counter"

    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return f"resp::{prompt}"


def test_caching_model_round_trip(tmp_path):
    cache = os.path.join(tmp_path, "cache.jsonl")
    stub = _CountingStub()
    m = CachingModel(stub, cache)
    r1 = m.generate("hello")
    r2 = m.generate("hello")            # served from cache
    assert r1 == r2 == "resp::hello"
    assert stub.calls == 1             # inner hit exactly once
    assert os.path.exists(cache)

    # a fresh wrapper reloads the persisted cache -> no new inner call
    stub2 = _CountingStub()
    m2 = CachingModel(stub2, cache)
    assert m2.generate("hello") == "resp::hello"
    assert stub2.calls == 0


def test_caching_model_distinct_prompts_miss():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        stub = _CountingStub()
        m = CachingModel(stub, os.path.join(d, "c.jsonl"))
        m.generate("a")
        m.generate("b")
        assert stub.calls == 2


# --------------------- Groq rate-limit backoff (offline) -------------------
def _http_error(code, headers=None, body=b""):
    import io
    import urllib.error
    return urllib.error.HTTPError(
        "https://api.groq.com", code, "err", headers or {}, io.BytesIO(body))


def test_wait_seconds_prefers_retry_after_header():
    from model_client import GroqModel
    g = GroqModel("x")
    err = _http_error(429, headers={"retry-after": "12"})
    assert g._wait_seconds(err, attempt=0) == pytest.approx(12.5)


def test_wait_seconds_parses_body_message():
    from model_client import GroqModel
    g = GroqModel("x")
    err = _http_error(429, headers={}, body=b'{"error":{"message":"Rate limit reached. Please try again in 7.5s."}}')
    assert g._wait_seconds(err, attempt=0) == pytest.approx(8.0)


def test_wait_seconds_falls_back_to_exponential_and_is_capped():
    from model_client import GroqModel
    g = GroqModel("x")
    err = _http_error(429, headers={}, body=b"no hint here")
    assert g._wait_seconds(err, attempt=3) == pytest.approx(8.0)        # 2**3
    # capped at MAX_BACKOFF_SEC for large attempts
    assert g._wait_seconds(err, attempt=20) == pytest.approx(GroqModel.MAX_BACKOFF_SEC)


def test_throttle_paces_calls():
    import time as _t
    from model_client import GroqModel
    g = GroqModel("x", min_interval=0.05)
    GroqModel._last_call_ts = 0.0
    t0 = _t.time()
    g._throttle(); g._throttle()        # second call must wait ~min_interval
    assert _t.time() - t0 >= 0.05


# --------------------- real-model result artifacts -------------------------
@pytest.fixture(scope="module")
def real_rows():
    models = _real_models()
    if not models:
        pytest.skip("no real-model result files; run scripts/run_real.py with a key")
    return {m: load_rows(m) for m in models}


def test_real_result_files_well_formed(real_rows):
    for m, rows in real_rows.items():
        assert len(rows) > 0
        for r in rows:
            assert RESULT_FIELDS <= set(r)
            assert r["judged_label"] in {"ANSWER", "FLAG_INCONSISTENCY", "ABSTAIN", "UNKNOWN"}
            assert isinstance(r["correct"], bool)


def test_real_models_carry_a_known_judge(real_rows):
    """Every real row must record which judge labelled it. We run two judge
    regimes by design: the stratified dry-runs are LLM-judged (groq:<id>),
    while the full 427-item qwen sweep is rule-judged (the rule judge is free
    and was human-validated at kappa 0.915, so it is sound for a full-corpus
    pass where per-call LLM judging would be slow/rate-limited). Both are
    recognised provenances; an unlabelled row is the bug."""
    for rows in real_rows.values():
        assert all(
            r.get("judge", "").startswith("groq:") or r.get("judge") == "rule"
            for r in rows
        )


def test_preserved_llm_judged_llama_runs_exist_and_are_llm_judged():
    """The validated LLM-judged n=80 Llama runs are preserved under
    results/llm_judged_n80/ as a judge cross-check (the live result files have
    since been scaled to the full corpus and rule-judged for uniformity with
    qwen). Where present, every preserved row must carry groq: judge
    provenance."""
    bak = os.path.join(RESULTS, "llm_judged_n80")
    if not os.path.isdir(bak):
        pytest.skip("preserved LLM-judged backup absent")
    for name in ("groq_llama-3.1-8b-instant", "groq_llama-3.3-70b-versatile"):
        path = os.path.join(bak, f"{name}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            rows = [json.loads(l) for l in fh]
        assert rows and all(r.get("judge", "").startswith("groq:") for r in rows)


def test_leaderboard_includes_real_models(real_rows):
    report = build_report({m: load_rows(m) for m in discover_models()})
    names = {m for m, _ in report["leaderboard"]}
    for m in real_rows:
        assert m in names


def test_penalty_sweep_top_and_bottom_are_stable_over_all_models():
    """Across the fabrication-penalty sweep the EXTREMES never move: the
    reconciler stays #1 and the fabricator stays last at every penalty. The
    full ordering is NOT globally stable once real models are included — that
    is intended, see test_penalty_sweep_flips_are_only_informative_pairs."""
    rows_by = {m: load_rows(m) for m in discover_models()}
    sweep = score.penalty_sweep(rows_by)
    for ranked in sweep.values():
        assert ranked[0][0] == "mock_reconciler"
        assert ranked[-1][0] == "mock_fabricator"


def test_penalty_sweep_flips_are_only_informative_pairs():
    """The full ordering is not globally stable once real models are in — and
    every flip that occurs is *informative*: a model that fabricates crossing a
    model that does not, as the fabrication penalty rises. At full scale there
    are exactly two such crossings:
      * qwen3-32b (fab ~0.18) sinking below the do-nothing `noisy` baseline at
        high P_FAB, and
      * llama-3.3-70b (fabricates ~1%, never over-flags) vs llama-3.1-8b
        (never fabricates, over-flags ~2%): near-identical accuracy, so their
        order is decided purely by how dearly fabrication is penalised — the
        audit-cost asymmetry made empirical on two real models.
    No other pair changes order. Extremes (reconciler #1, fabricator last) are
    fixed; see test_penalty_sweep_top_and_bottom_are_stable_over_all_models."""
    rows_by = {m: load_rows(m) for m in discover_models()}
    need = {"groq_qwen-qwen3-32b", "mock_noisy",
            "groq_llama-3.1-8b-instant", "groq_llama-3.3-70b-versatile"}
    if not need <= set(rows_by):
        pytest.skip("real-model full runs absent")
    sweep = score.penalty_sweep(rows_by)
    orders = [[m for m, _ in ranked] for ranked in sweep.values()]

    def flipped_pairs(o1, o2):
        names = set(o1)
        return {
            frozenset((a, b))
            for a in names for b in names if a < b
            and (o1.index(a) < o1.index(b)) != (o2.index(a) < o2.index(b))
        }

    flips = set()
    for o in orders[1:]:
        flips |= flipped_pairs(orders[0], o)
    assert flips == {
        frozenset(("groq_qwen-qwen3-32b", "mock_noisy")),
        frozenset(("groq_llama-3.1-8b-instant", "groq_llama-3.3-70b-versatile")),
    }


# --- the real-model headline: audit-cost asymmetry decides the order ---------
def test_real_llamas_opposite_errors_and_penalty_dependent_order():
    """At full corpus scale the two Llama models have near-identical accuracy
    but OPPOSITE dominant error types — 8b over-flags and never fabricates;
    70b fabricates a little and never over-flags. Because they are so close,
    their Trust-Score order is decided purely by the fabrication penalty: at a
    low or default penalty the higher-accuracy (but once-fabricating) 70b ranks
    above the 8b; only at a high penalty does the non-fabricating 8b overtake it.
    That penalty-driven crossover on two real models is the audit-cost
    asymmetry made concrete — a symmetric metric, fixing order by accuracy
    alone, cannot express it."""
    a_name = "groq_llama-3.1-8b-instant"      # over-flagger
    b_name = "groq_llama-3.3-70b-versatile"   # fabricator
    for n in (a_name, b_name):
        if not os.path.exists(os.path.join(RESULTS, f"{n}.jsonl")):
            pytest.skip("real-model files absent")
    a, b = load_rows(a_name), load_rows(b_name)
    sa, sb = score.score_rows(a), score.score_rows(b)

    # near-identical capability ...
    assert abs(score.crs(a) - score.crs(b)) < 0.02
    # ... opposite dominant error types
    assert sa["fabrication_rate"] == 0.0 and sa["false_flag_rate"] > 0.0     # 8b over-flags only
    assert sb["fabrication_rate"] > 0.0 and sb["false_flag_rate"] == 0.0     # 70b fabricates only

    # the order is penalty-dependent: 70b ahead when fabrication is cheap,
    # 8b ahead once it is appropriately costly. At full corpus the 70b
    # fabricates only once (1/212, the ADBE case), so it stays ahead through
    # the default P_FAB=4 and the crossover lands between P_FAB 4 and 8.
    sweep = score.penalty_sweep({a_name: a, b_name: b}, p_fabs=(2.0, 8.0))
    low = {m: v for m, v in sweep[2.0]}
    high = {m: v for m, v in sweep[8.0]}
    assert low[b_name] > low[a_name]      # cheap fabrication -> 70b wins
    assert high[a_name] > high[b_name]    # costly fabrication -> 8b wins


# --------------------------- reliability table -----------------------------
def test_reliability_table_well_formed(real_rows):
    for rows in real_rows.values():
        table = score.reliability_table(rows)
        committed = sum(1 for r in rows if r["judged_label"] in {"ANSWER", "FLAG_INCONSISTENCY"})
        assert sum(b["count"] for b in table) == committed
        for b in table:
            assert 0.0 <= b["accuracy"] <= 1.0
            assert 0.0 <= b["mean_confidence"] <= 1.0
            assert b["bin"][0] < b["bin"][1]


# ------------------- recorded judge validation (Chunk 4) -------------------
def test_recorded_judge_validation_llm_beats_rule():
    path = os.path.join(RESULTS, "judge_validation.json")
    if not os.path.exists(path):
        pytest.skip("judge_validation.json absent; run scripts/validate_judge.py")
    rec = json.load(open(path))
    assert rec["rule_judge"]["agreement"] >= 0.85
    assert rec["llm_judge"]["agreement"] >= rec["rule_judge"]["agreement"]
    assert rec["llm_beats_rule"] is True
    # LLM judge should have no more disagreements than the rule judge
    assert rec["llm_judge"]["n_disagreements"] <= rec["rule_judge"]["n_disagreements"]
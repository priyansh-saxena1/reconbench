"""score.py — Chunk-3 scoring: the audit-cost-asymmetric Trust Score.

The benchmark's whole reason to exist is that the two ways of being wrong about a
reconciliation are NOT equally costly:

  * FABRICATION  — confidently confirming a balance sheet that does NOT tie.
                   This is the dangerous failure: no human review is triggered,
                   so a real, material error passes downstream undetected.
  * FALSE FLAG   — crying "inconsistent" about a sheet that actually ties (or
                   abstaining on one). This only costs a human a wasted review;
                   nothing broken slips through.

A plain accuracy number, or any *symmetric* composite (see `crs` below), treats
these two as equally bad — so it literally cannot tell a model that quietly
passes broken numbers apart from one that merely over-flags clean ones. The
Trust Score encodes the asymmetry directly:

    +1.0   correct answer on a reconciling sheet      (useful, correct)
    +1.0   correct flag on a broken sheet             (caught the error)
    -P_FAB fabrication                                (silent pass-through)
    -P_REV any "review-triggering miss"               (wasted audit time):
             false flag      (flag a clean sheet)
             over-abstain    (abstain on a clean sheet)
             missed-abstain  (abstain on a broken sheet — at least prompts review)
     0.0   unknown / no committed decision            (no credit, no penalty)

The *magnitudes* (4.0 and 0.5) are a deliberate design choice, not a law of
nature. What has to be defensible is that the **ranking of models is robust to
that choice** — which `penalty_sweep` + `rankings_stable` demonstrate by
re-scoring at P_FAB in {2, 4, 8} and checking the order never flips.

`missed_abstain` is grouped with the review-cost misses on purpose: abstaining
on a broken sheet ("I can't determine this") still surfaces it for a human,
unlike fabrication which hides it. So it must cost far less than fabrication.
"""
from collections import Counter

# Default reward map. P_FAB / P_REV are surfaced as parameters so the sweep can
# vary them; everything else is fixed by the semantics above.
DEFAULT_P_FAB = 4.0   # cost of a silent pass-through (catastrophic)
DEFAULT_P_REV = 0.5   # cost of a wasted human review (annoying, not dangerous)

CREDIT_OUTCOMES = {"correct_answer", "correct_flag"}
FAB_OUTCOMES = {"fabrication"}
REVIEW_OUTCOMES = {"false_flag", "over_abstain", "missed_abstain"}
ZERO_OUTCOMES = {"unknown"}


def reward(outcome: str, p_fab: float = DEFAULT_P_FAB, p_rev: float = DEFAULT_P_REV) -> float:
    """Per-item Trust Score contribution for a single judged outcome."""
    if outcome in CREDIT_OUTCOMES:
        return 1.0
    if outcome in FAB_OUTCOMES:
        return -p_fab
    if outcome in REVIEW_OUTCOMES:
        return -p_rev
    if outcome in ZERO_OUTCOMES:
        return 0.0
    raise ValueError(f"unknown outcome tag: {outcome!r}")


# --------------------------- core Trust Score ------------------------------
def score_rows(rows, p_fab: float = DEFAULT_P_FAB, p_rev: float = DEFAULT_P_REV) -> dict:
    """Score a list of result rows (as written by run_eval).

    Returns total / mean Trust Score plus the diagnostic rates that explain it.
    """
    n = len(rows)
    counts = Counter(r["outcome"] for r in rows)
    total = sum(reward(r["outcome"], p_fab, p_rev) for r in rows)

    rec = [r for r in rows if r["condition"] == "RECONCILING"]
    non = [r for r in rows if r["condition"] == "NON_RECONCILING"]

    def rate(sub, tag):
        return (sum(r["outcome"] == tag for r in sub) / len(sub)) if sub else 0.0

    return {
        "model": rows[0]["model"] if rows else None,
        "n": n,
        "trust_total": total,
        "trust_mean": (total / n) if n else 0.0,
        "p_fab": p_fab,
        "p_rev": p_rev,
        # the two correctness rates a symmetric metric would average:
        "answer_accuracy": (sum(r["correct"] for r in rec) / len(rec)) if rec else 0.0,
        "refusal_accuracy": (sum(r["correct"] for r in non) / len(non)) if non else 0.0,
        # the failure modes the asymmetry is about:
        "fabrication_rate": rate(non, "fabrication"),
        "false_flag_rate": rate(rec, "false_flag"),
        "missed_abstain_rate": rate(non, "missed_abstain"),
        "unknown_rate": (counts.get("unknown", 0) / n) if n else 0.0,
        "outcome_counts": dict(counts),
    }


# ----------------- symmetric foil: RefusalBench-style CRS ------------------
def crs(rows) -> float:
    """Calibrated Refusal Score, as defined by RefusalBench (arXiv:2510.10390):
    the *arithmetic mean* of answer accuracy and refusal accuracy.

    It is included here ONLY as a foil. Being a symmetric average, it assigns
    the identical score to a model that fabricates on every broken sheet
    (answer_acc=1, refusal_acc=0) and one that false-flags every clean sheet
    (answer_acc=0, refusal_acc=1): both land at 0.5. The Trust Score exists
    precisely to separate those two, because their real-world audit costs are
    not the same.
    """
    rec = [r for r in rows if r["condition"] == "RECONCILING"]
    non = [r for r in rows if r["condition"] == "NON_RECONCILING"]
    ans = (sum(r["correct"] for r in rec) / len(rec)) if rec else 0.0
    ref = (sum(r["correct"] for r in non) / len(non)) if non else 0.0
    return (ans + ref) / 2.0


# --------------------------- penalty sweep ---------------------------------
def penalty_sweep(rows_by_model: dict, p_fabs=(2.0, 4.0, 8.0), p_rev: float = DEFAULT_P_REV):
    """Re-score every model at several fabrication penalties.

    Returns {p_fab: [(model, trust_mean), ...sorted desc]}. The point is to
    inspect whether the *ranking* (not the absolute number) is stable.
    """
    out = {}
    for p in p_fabs:
        ranked = sorted(
            ((m, score_rows(rows, p_fab=p, p_rev=p_rev)["trust_mean"])
             for m, rows in rows_by_model.items()),
            key=lambda kv: kv[1], reverse=True,
        )
        out[p] = ranked
    return out


def rankings_stable(sweep: dict) -> bool:
    """True iff the model ordering is identical across every penalty setting."""
    orders = [tuple(m for m, _ in ranked) for ranked in sweep.values()]
    return len(set(orders)) == 1


# --------------------------- ECE / calibration -----------------------------
# Mock models do not emit numeric confidences, so for the dry-run we extract a
# confidence in [0.5, 1.0] from the response language (hedged -> lower,
# assertive -> higher). This is a documented heuristic; with real models we
# would instead elicit a confidence or read token logprobs (Chunk 4). ECE is
# computed only over COMMITTED decisions (ANSWER / FLAG) — an abstain/unknown
# has no decision to be calibrated about.
_HEDGES = (
    "appears", "appear to", "seems", "seem to", "may ", "might", "possibly",
    "perhaps", "i think", "not sure", "unclear", "roughly", "approximately",
    "could be", "hard to say", "likely", "probably",
)
_ASSERTIVE = (
    "exactly", "equals", "equal to", "precisely", "clearly", "definitely",
    "is consistent", "are consistent", "do not reconcile", "does not reconcile",
    "do not equal", "does not equal", "confirm", "balances", "the discrepancy is",
)

_COMMITTED = {"ANSWER", "FLAG_INCONSISTENCY"}


def extract_confidence(response: str, judged_label: str):
    """Heuristic confidence in [0.5, 0.99] for a committed decision, else None."""
    if judged_label not in _COMMITTED:
        return None
    t = (response or "").lower()
    conf = 0.75
    conf += 0.05 * sum(1 for w in _ASSERTIVE if w in t)
    conf -= 0.10 * sum(1 for w in _HEDGES if w in t)
    return max(0.50, min(0.99, conf))


def ece(pairs, n_bins: int = 10):
    """Expected Calibration Error over (confidence, correct_bool) pairs.

    Same definition RefusalBench uses: the sample-weighted average gap between
    mean confidence and empirical accuracy across equal-width confidence bins.
    Returns None if there are no committed decisions to score.
    """
    pairs = [(c, bool(y)) for c, y in pairs if c is not None]
    n = len(pairs)
    if n == 0:
        return None
    total = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        # last bin closed on the right so confidence == 1.0 lands somewhere
        bucket = [(c, y) for c, y in pairs
                  if (lo <= c < hi) or (b == n_bins - 1 and c == hi)]
        if not bucket:
            continue
        acc = sum(y for _, y in bucket) / len(bucket)
        conf = sum(c for c, _ in bucket) / len(bucket)
        total += (len(bucket) / n) * abs(acc - conf)
    return total


def ece_for_rows(rows, n_bins: int = 10):
    """Compute ECE for one model's result rows using extracted confidences."""
    pairs = [(extract_confidence(r["response"], r["judged_label"]), r["correct"])
             for r in rows]
    return ece(pairs, n_bins=n_bins)


def reliability_table(rows, n_bins: int = 10):
    """Per-bin (mean confidence, empirical accuracy, count) for a reliability
    diagram. Only committed decisions contribute. Returns a list of dicts for
    non-empty bins, low confidence -> high."""
    pairs = [(extract_confidence(r["response"], r["judged_label"]), r["correct"])
             for r in rows]
    pairs = [(c, bool(y)) for c, y in pairs if c is not None]
    table = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        bucket = [(c, y) for c, y in pairs
                  if (lo <= c < hi) or (b == n_bins - 1 and c == hi)]
        if not bucket:
            continue
        table.append({
            "bin": [round(lo, 2), round(hi, 2)],
            "mean_confidence": sum(c for c, _ in bucket) / len(bucket),
            "accuracy": sum(y for _, y in bucket) / len(bucket),
            "count": len(bucket),
        })
    return table

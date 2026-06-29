"""Perturbation engine — reconciling -> NON_RECONCILING.

Strategy (preferred): period-swap. Replace ONE component (equity) with the same
company's real value from a DIFFERENT period. The replacement is a genuine
figure that survives an eyeball ("looks like a plausible equity number"), but
breaks the identity — so detection requires actually doing the arithmetic.
Fallback: a realistic percentage nudge if no suitable other-period value exists.

Every result is checked against materiality + plausibility thresholds; if it
fails, the item is rejected (never silently emitted).
"""
import config
from identities import discrepancy


def _material(assets: int, liabilities: int, equity: int) -> bool:
    d = abs(discrepancy(assets, liabilities, equity))
    rel = d / assets
    return (d >= config.MIN_ABS_DISCREPANCY
            and rel >= config.MIN_REL_DISCREPANCY
            and rel <= config.MAX_REL_DISCREPANCY)


def _plausible(original: int, replacement: int) -> bool:
    if replacement <= 0:
        return False
    lo, hi = config.PLAUSIBLE_LOW * original, config.PLAUSIBLE_HIGH * original
    return lo <= replacement <= hi


def make_non_reconciling(seed: dict) -> dict | None:
    """Return a NON_RECONCILING item dict (or None if no valid perturbation).

    `seed["_equity_pool"]` is a list of (equity_value, period_end) from OTHER
    periods of the same company, ordered closest-first so the chosen wrong value
    is the least obviously off.
    """
    assets = seed["components"]["Assets"]
    liabilities = seed["components"]["Liabilities"]
    orig_equity = seed["components"]["Equity"]

    chosen = None
    method = None
    source_period = None

    # 1) period-swap: prefer a real other-period equity that is material+plausible
    for repl, period in seed.get("_equity_pool", []):
        if repl == orig_equity:
            continue
        if _plausible(orig_equity, repl) and _material(assets, liabilities, repl):
            chosen, method, source_period = repl, "period_swap", period
            break

    # 2) fallback: realistic +/- nudge (~4%) scaled up until material, capped plausible
    if chosen is None:
        for pct in (0.04, 0.06, 0.08, 0.10):
            repl = int(round(orig_equity * (1 + pct)))
            if _plausible(orig_equity, repl) and _material(assets, liabilities, repl):
                chosen, method = repl, f"nudge_+{int(pct*100)}pct"
                break

    if chosen is None:
        return None

    item = dict(seed)  # shallow copy of the reconciling seed
    item.pop("_equity_pool", None)
    item["id"] = seed["id"].replace("rb_", "rb_") + "_nr"
    item["condition"] = "NON_RECONCILING"
    item["expected_behavior"] = "FLAG_INCONSISTENCY"
    item["components"] = {"Assets": assets, "Liabilities": liabilities, "Equity": chosen}
    d = discrepancy(assets, liabilities, chosen)
    item["perturbation"] = {
        "field": "Equity",
        "method": method,
        "original_value": orig_equity,
        "replacement_value": chosen,
        "source_period": source_period,
        "discrepancy": d,
        "rel_discrepancy": round(d / assets, 5),
    }
    item["gold_answer"] = (
        f"Inconsistent: assets ({assets:,}) != liabilities ({liabilities:,}) + "
        f"equity ({chosen:,}) = {liabilities + chosen:,}; discrepancy {d:,}."
    )
    return item

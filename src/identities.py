"""Accounting-identity logic.

Balance sheet: Assets = Liabilities + total Equity. We require an EXACT integer
tie (USD), and auto-select whichever equity tag makes it tie — any period that
does not tie exactly is discarded, so a reconciling seed can never be mislabeled.
"""
import config


def reconciles(assets: int, liabilities: int, equity: int) -> bool:
    """True iff the balance-sheet identity holds exactly (integer USD)."""
    return assets == liabilities + equity


def discrepancy(assets: int, liabilities: int, equity: int) -> int:
    """Signed reconciliation gap: assets - (liabilities + equity)."""
    return assets - (liabilities + equity)


def resolve_equity(assets, liabilities, equity_candidates: dict) -> int | None:
    """Pick the equity value (from candidate tags) that makes the identity tie
    exactly; return None if none does.
    `equity_candidates` maps tag -> value (already at the right period_end).
    """
    for tag in config.EQUITY_TAGS:
        val = equity_candidates.get(tag)
        if val is not None and reconciles(assets, liabilities, val):
            return val
    return None

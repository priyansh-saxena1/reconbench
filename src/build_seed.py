"""Build reconciling seeds from SEC XBRL companyfacts.

For each company, join Assets / Liabilities / Equity at matching period_end for
10-K filings, auto-select the equity tag that ties EXACTLY, and emit one
reconciling seed per tying period. Each seed carries an `_equity_pool` of the
company's other-period equity values (for period-swap perturbation later).
"""
import config
from sec_client import get_company_facts, instant_facts_by_end
from identities import resolve_equity


def _equity_candidates_by_end(facts: dict) -> dict[str, dict]:
    """period_end -> {equity_tag: value} across all equity tag variants."""
    out: dict[str, dict] = {}
    for tag in config.EQUITY_TAGS:
        for end, val in instant_facts_by_end(facts, tag).items():
            out.setdefault(end, {})[tag] = val
    return out


def build_seeds_for_company(cik: str, name: str, ticker: str) -> list[dict]:
    facts = get_company_facts(cik)
    assets = instant_facts_by_end(facts, config.ASSETS_TAG)
    liabilities = instant_facts_by_end(facts, config.LIABILITIES_TAG)
    equity_cands = _equity_candidates_by_end(facts)

    # first pass: resolve a tying equity value for every period that ties
    tying: dict[str, int] = {}  # period_end -> equity value
    for end in sorted(set(assets) & set(liabilities) & set(equity_cands)):
        eq = resolve_equity(assets[end], liabilities[end], equity_cands[end])
        if eq is not None:
            tying[end] = eq

    seeds = []
    for end, eq in sorted(tying.items()):
        fy = int(end[:4])
        # pool of OTHER-period equities for this company, closest-first
        pool = sorted(
            [(v, p) for p, v in tying.items() if p != end],
            key=lambda t: abs(t[0] - eq),
        )
        seeds.append({
            "id": f"rb_{cik[-4:]}_{end.replace('-', '')}",
            "company": name,
            "cik": cik,
            "ticker": ticker,
            "identity": "balance_sheet",
            "identity_expr": "Assets = Liabilities + StockholdersEquity",
            "period_end": end,
            "fiscal_year": fy,
            "form": "10-K",
            "condition": "RECONCILING",
            "components": {
                "Assets": assets[end],
                "Liabilities": liabilities[end],
                "Equity": eq,
            },
            "expected_behavior": "ANSWER",
            "gold_answer": (
                f"Consistent: assets ({assets[end]:,}) = liabilities "
                f"({liabilities[end]:,}) + equity ({eq:,}). Total assets = {assets[end]:,}."
            ),
            "perturbation": None,
            "source_url": config.SEC_FACTS_URL.format(cik=int(cik)),
            "_equity_pool": pool,
        })
    return seeds


def build_all_seeds() -> list[dict]:
    seeds = []
    for cik, (name, ticker) in config.COMPANIES.items():
        try:
            cseeds = build_seeds_for_company(cik, name, ticker)
            seeds.extend(cseeds)
            print(f"  {ticker:6} {name[:34]:34} -> {len(cseeds)} tying periods")
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"  {ticker:6} ERROR: {e}")
    return seeds

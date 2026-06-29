"""ReconBench item schema (LOCKED — Chunk 1).

Changing this schema after downstream code (harness, judge, scorer) depends on
it is the single most expensive mistake, so it is frozen here and validated.
"""

REQUIRED_FIELDS = [
    "id",                 # str   e.g. "rb_0001"
    "company",            # str
    "cik",                # str   10-digit
    "ticker",             # str
    "identity",           # str   "balance_sheet"
    "identity_expr",      # str   "Assets = Liabilities + StockholdersEquity"
    "period_end",         # str   "YYYY-MM-DD"
    "fiscal_year",        # int
    "form",               # str   "10-K"
    "condition",          # str   "RECONCILING" | "NON_RECONCILING"
    "components",         # dict  {"Assets":int,"Liabilities":int,"Equity":int}
    "expected_behavior",  # str   "ANSWER" | "FLAG_INCONSISTENCY"
    "gold_answer",        # str
    "perturbation",       # dict|None  details for NON_RECONCILING
    "source_url",         # str
]

CONDITIONS = {"RECONCILING", "NON_RECONCILING"}
EXPECTED = {"RECONCILING": "ANSWER", "NON_RECONCILING": "FLAG_INCONSISTENCY"}
COMPONENT_KEYS = {"Assets", "Liabilities", "Equity"}


def validate_item(item: dict) -> list[str]:
    """Return a list of problems (empty list == valid)."""
    problems = []
    for f in REQUIRED_FIELDS:
        if f not in item:
            problems.append(f"missing field: {f}")
    if problems:
        return problems

    if item["condition"] not in CONDITIONS:
        problems.append(f"bad condition: {item['condition']}")
    if item["expected_behavior"] != EXPECTED.get(item["condition"]):
        problems.append(
            f"expected_behavior {item['expected_behavior']} inconsistent "
            f"with condition {item['condition']}"
        )
    if set(item["components"]) != COMPONENT_KEYS:
        problems.append(f"components keys != {COMPONENT_KEYS}: {set(item['components'])}")
    if not all(isinstance(v, int) for v in item["components"].values()):
        problems.append("component values must be int (USD)")
    if item["condition"] == "NON_RECONCILING" and not item.get("perturbation"):
        problems.append("NON_RECONCILING item missing perturbation metadata")
    if item["condition"] == "RECONCILING" and item.get("perturbation") is not None:
        problems.append("RECONCILING item should have perturbation = None")
    return problems

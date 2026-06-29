"""Render an item into the provided-context prompt + question.

Kept separate so the Chunk-2 eval harness renders EXACTLY what the dataset was
built with. Provided-context only (no retrieval) — a locked V0 decision.
"""


def render_context(item: dict) -> str:
    c = item["components"]
    return (
        f"You are reviewing figures reported in {item['company']}'s "
        f"Form {item['form']} for fiscal year {item['fiscal_year']} "
        f"(period ending {item['period_end']}).\n\n"
        f"Reported balance sheet totals (USD):\n"
        f"- Total assets: {c['Assets']:,}\n"
        f"- Total liabilities: {c['Liabilities']:,}\n"
        f"- Total stockholders' equity: {c['Equity']:,}"
    )


QUESTION = (
    "A balance sheet must satisfy: Total assets = Total liabilities + Total "
    "stockholders' equity. Do these reported figures satisfy that identity? "
    "If they do, confirm consistency and state total assets. If they do not, "
    "flag the inconsistency and identify which figures fail to reconcile."
)


def render_prompt(item: dict) -> str:
    return f"{render_context(item)}\n\n{QUESTION}"

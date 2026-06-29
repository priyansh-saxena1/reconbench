"""SEC XBRL client — minimal companyfacts pull (no XBRL parser).

One HTTP GET per company returns every tagged fact; we cache it to disk so the
dataset is reproducible without re-hitting SEC. Verified working 2026-06-26.
"""
import json
import os
import time
import urllib.request

import config

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "_cache")


def _cache_path(cik: str) -> str:
    return os.path.join(_CACHE_DIR, f"CIK{int(cik):010d}.json")


def get_company_facts(cik: str, use_cache: bool = True) -> dict:
    """Return the full companyfacts JSON for a CIK (cached on disk)."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = _cache_path(cik)
    if use_cache and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)

    url = config.SEC_FACTS_URL.format(cik=int(cik))
    req = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    with open(path, "w") as fh:
        json.dump(data, fh)
    time.sleep(config.REQUEST_SLEEP_SEC)
    return data


def instant_facts_by_end(facts: dict, tag: str, form: str = "10-K") -> dict:
    """Map period_end -> value for an *instant* us-gaap concept (balance-sheet
    items have only an 'end' date). Later filings (amendments / restatements)
    overwrite earlier ones for the same end date (last-filed wins).
    """
    try:
        rows = facts["facts"]["us-gaap"][tag]["units"]["USD"]
    except KeyError:
        return {}
    by_end: dict[str, dict] = {}
    for r in rows:
        if r.get("form") not in (form, form + "/A"):
            continue
        if "start" in r:  # skip duration facts; balance-sheet items are instant
            continue
        end = r["end"]
        prev = by_end.get(end)
        if prev is None or r.get("filed", "") >= prev.get("filed", ""):
            by_end[end] = r
    return {end: r["val"] for end, r in by_end.items()}

"""make_dataset.py — Chunk-1 orchestrator (runs against LIVE SEC data).

Pulls reconciling tuples from SEC XBRL, generates NON_RECONCILING counterparts
by perturbation, validates every item against the schema, writes two JSONL
files, and prints a summary + 5 sample items.

Run:  python scripts/make_dataset.py
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)              # for config
sys.path.insert(0, os.path.join(ROOT, "src"))

import config                          # noqa: E402
from build_seed import build_all_seeds # noqa: E402
from perturb import make_non_reconciling  # noqa: E402
from schema import validate_item       # noqa: E402
from render import render_prompt       # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")


def _strip_internal(item: dict) -> dict:
    return {k: v for k, v in item.items() if not k.startswith("_")}


def main() -> int:
    random.seed(config.RANDOM_SEED)
    print("Pulling reconciling seeds from SEC XBRL (live)...")
    seeds = build_all_seeds()
    print(f"\nTotal reconciling seeds (exact ties): {len(seeds)}")

    # build paired NON_RECONCILING items
    items: list[dict] = []
    n_nr = 0
    for s in seeds:
        items.append(_strip_internal(s))
        nr = make_non_reconciling(s)
        if nr is not None:
            items.append(_strip_internal(nr))
            n_nr += 1

    # validate EVERY item against the locked schema
    bad = []
    for it in items:
        probs = validate_item(it)
        if probs:
            bad.append((it.get("id"), probs))
    if bad:
        print(f"\nSCHEMA VALIDATION FAILED for {len(bad)} items:")
        for _id, probs in bad[:10]:
            print(f"  {_id}: {probs}")
        return 1

    recon = [i for i in items if i["condition"] == "RECONCILING"]
    nonrecon = [i for i in items if i["condition"] == "NON_RECONCILING"]

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "seed_reconciling.jsonl"), "w") as fh:
        for it in recon:
            fh.write(json.dumps(it) + "\n")
    with open(os.path.join(DATA_DIR, "reconbench_v0.jsonl"), "w") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")

    print("\n=== SUMMARY ===")
    print(f"  RECONCILING (control): {len(recon)}")
    print(f"  NON_RECONCILING (test): {len(nonrecon)}")
    print(f"  TOTAL items: {len(items)}  (schema-valid: {len(items) - len(bad)})")
    print(f"  Companies represented: {len({i['ticker'] for i in items})}")
    print(f"  Written: data/seed_reconciling.jsonl, data/reconbench_v0.jsonl")

    # show 5 sample items (mix of both conditions), with rendered prompt
    print("\n=== 5 SAMPLE ITEMS (rendered as the model will see them) ===")
    sample = (recon[:3] + nonrecon[:2]) if len(recon) >= 3 and nonrecon else items[:5]
    for it in sample:
        print("\n" + "-" * 72)
        print(f"id={it['id']}  condition={it['condition']}  expected={it['expected_behavior']}")
        print(render_prompt(it))
        if it["perturbation"]:
            p = it["perturbation"]
            print(f"[perturbation] {p['method']}: Equity {p['original_value']:,} -> "
                  f"{p['replacement_value']:,}  (discrepancy {p['discrepancy']:,}, "
                  f"{p['rel_discrepancy']*100:.2f}% of assets)")
        print(f"[gold] {it['gold_answer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

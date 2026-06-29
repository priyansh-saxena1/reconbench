"""run_eval.py — Chunk-2 eval harness (dry-run on mock models).

For each dataset item: render the SAME prompt the data was built with, get the
model's response, judge it, and categorize the outcome. Writes per-item results
and prints a diagnostic summary.

NOTE: this prints a *diagnostic skeleton* (accuracy, fabrication rate, false-flag
rate). The full audit-cost Trust Score + penalty sweep + CRS differentiation +
ECE is Chunk 3 — intentionally not built here.

Run:  python scripts/run_eval.py --model mock_reconciler
      python scripts/run_eval.py --model mock_fabricator --limit 50
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from render import render_prompt          # noqa: E402
from judge import judge_response, judge_response_llm  # noqa: E402
from model_client import get_model, GroqModel, CachingModel  # noqa: E402

DATASET = os.path.join(ROOT, "data", "reconbench_v0.jsonl")
RESULTS_DIR = os.path.join(ROOT, "results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")

# expected_behavior -> judged_label -> outcome tag
_OUTCOME = {
    "ANSWER": {  # RECONCILING items
        "ANSWER": "correct_answer", "FLAG_INCONSISTENCY": "false_flag",
        "ABSTAIN": "over_abstain", "UNKNOWN": "unknown",
    },
    "FLAG_INCONSISTENCY": {  # NON_RECONCILING items
        "FLAG_INCONSISTENCY": "correct_flag", "ANSWER": "fabrication",
        "ABSTAIN": "missed_abstain", "UNKNOWN": "unknown",
    },
}
_CORRECT = {"correct_answer", "correct_flag"}


def load_items(path=DATASET):
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def stratified_sample(items, n, seed=26):
    """Return n items, balanced across the two conditions (reproducible)."""
    rng = random.Random(seed)
    rec = [i for i in items if i["condition"] == "RECONCILING"]
    non = [i for i in items if i["condition"] == "NON_RECONCILING"]
    half = n // 2
    pick = rng.sample(rec, min(half, len(rec))) + rng.sample(non, min(n - half, len(non)))
    rng.shuffle(pick)
    return pick


def run_eval(model, items, limit=None, judge_fn=judge_response):
    """judge_fn: str->label. Default rule judge; pass an LLM-judge closure for
    the validated LLM path."""
    rows = []
    for it in items[: limit or len(items)]:
        resp = model.generate(render_prompt(it))
        judged = judge_fn(resp)
        outcome = _OUTCOME[it["expected_behavior"]][judged]
        rows.append({
            "id": it["id"], "company": it["ticker"], "condition": it["condition"],
            "expected_behavior": it["expected_behavior"], "model": model.name,
            "response": resp, "judged_label": judged, "outcome": outcome,
            "correct": outcome in _CORRECT,
        })
    return rows


def summarize(rows):
    n = len(rows)
    rec = [r for r in rows if r["condition"] == "RECONCILING"]
    non = [r for r in rows if r["condition"] == "NON_RECONCILING"]
    def rate(sub, tag):
        return sum(r["outcome"] == tag for r in sub) / len(sub) if sub else 0.0
    return {
        "model": rows[0]["model"] if rows else None,
        "n": n,
        "accuracy": sum(r["correct"] for r in rows) / n if n else 0.0,
        "acc_reconciling": sum(r["correct"] for r in rec) / len(rec) if rec else 0.0,
        "acc_non_reconciling": sum(r["correct"] for r in non) / len(non) if non else 0.0,
        "fabrication_rate": rate(non, "fabrication"),      # confident wrong (dangerous)
        "missed_abstain_rate": rate(non, "missed_abstain"),
        "false_flag_rate": rate(rec, "false_flag"),        # over-refusal on controls
        "unknown_rate": sum(r["outcome"] == "unknown" for r in rows) / n if n else 0.0,
    }


def _print_summary(s):
    print(f"\n=== {s['model']}  (n={s['n']}) ===")
    print(f"  accuracy overall          : {s['accuracy']:.3f}")
    print(f"  accuracy on RECONCILING   : {s['acc_reconciling']:.3f}")
    print(f"  accuracy on NON_RECONCILING: {s['acc_non_reconciling']:.3f}")
    print(f"  fabrication rate (danger) : {s['fabrication_rate']:.3f}  <- confirms a broken balance sheet")
    print(f"  false-flag rate (controls): {s['false_flag_rate']:.3f}  <- cries wolf on consistent figures")
    print(f"  missed-abstain rate       : {s['missed_abstain_rate']:.3f}")
    print(f"  unknown (unjudgeable)     : {s['unknown_rate']:.3f}")


def _make_judge(spec):
    """spec: 'rule' or 'groq:<id>'. Returns (judge_fn, label) where the LLM
    judge is cached and wraps the rule judge as fallback."""
    if spec == "rule":
        return judge_response, "rule"
    if spec.startswith("groq:"):
        judge_model = CachingModel(GroqModel(spec.split(":", 1)[1]),
                                   os.path.join(RAW_DIR, f"judge_{spec.replace(':', '_').replace('/', '-')}.jsonl"))
        return (lambda r: judge_response_llm(r, judge_model)), spec
    raise ValueError(f"unknown judge spec: {spec!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="mock_reconciler|mock_fabricator|mock_overcautious|mock_noisy|groq:<id>")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stratified", type=int, default=None,
                    help="evaluate a balanced sample of N items (N/2 per condition)")
    ap.add_argument("--judge", default="rule",
                    help="'rule' (default) or 'groq:<id>' for the LLM judge")
    ap.add_argument("--cache", action="store_true",
                    help="cache raw model responses to results/raw/ (resumable)")
    args = ap.parse_args()

    items = load_items()
    if args.stratified:
        items = stratified_sample(items, args.stratified)

    model = get_model(args.model)
    if args.cache and not args.model.startswith("mock_"):
        model = CachingModel(model, os.path.join(RAW_DIR, f"{model.name.replace(':', '_').replace('/', '-')}.jsonl"))

    judge_fn, judge_label = _make_judge(args.judge)
    rows = run_eval(model, items, args.limit, judge_fn=judge_fn)
    for r in rows:
        r["judge"] = judge_label

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"{model.name.replace(':', '_').replace('/', '-')}.jsonl")
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    _print_summary(summarize(rows))
    print(f"  judge: {judge_label}")
    print(f"  wrote {len(rows)} rows -> {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

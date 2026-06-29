"""judge_validation.py — validate the rule judge against human gold labels.

Reports overall agreement, Cohen's kappa (agreement corrected for chance),
per-class precision/recall, a confusion matrix, and the explicit list of
disagreements. The disagreement list is the useful part: it tells us exactly
which phrasings the cheap rule judge gets wrong, which is the case for promoting
the LLM judge (judge.judge_response_llm) in Chunk 4.

The same harness can validate ANY judge function with signature str -> label,
so it will be reused to validate the LLM judge against the rule judge / humans.
"""
import json
import os
from collections import Counter, defaultdict

from judge import LABELS, judge_response

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOLD = os.path.join(ROOT, "data", "judge_goldset.jsonl")


def load_gold(path=GOLD):
    with open(path) as fh:
        return [json.loads(l) for l in fh]


def cohen_kappa(pairs):
    """Cohen's kappa for (gold, pred) label pairs over LABELS."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(g == p for g, p in pairs) / n
    g_counts = Counter(g for g, _ in pairs)
    p_counts = Counter(p for _, p in pairs)
    pe = sum((g_counts.get(l, 0) / n) * (p_counts.get(l, 0) / n) for l in LABELS)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def confusion(pairs):
    """confusion[gold][pred] = count."""
    m = defaultdict(lambda: defaultdict(int))
    for g, p in pairs:
        m[g][p] += 1
    return m


def per_class(pairs):
    """precision/recall/f1 per label."""
    stats = {}
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in pairs)
        fp = sum(g != label and p == label for g, p in pairs)
        fn = sum(g == label and p != label for g, p in pairs)
        prec = tp / (tp + fp) if (tp + fp) else None
        rec = tp / (tp + fn) if (tp + fn) else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
        stats[label] = {"precision": prec, "recall": rec, "f1": f1,
                        "support": sum(g == label for g, _ in pairs)}
    return stats


def validate(judge_fn=judge_response, gold=None):
    gold = gold if gold is not None else load_gold()
    pairs, disagreements = [], []
    for row in gold:
        pred = judge_fn(row["response"])
        pairs.append((row["human_label"], pred))
        if pred != row["human_label"]:
            disagreements.append({
                "id": row["id"], "source": row["source"],
                "response": row["response"],
                "human": row["human_label"], "judge": pred,
            })
    n = len(pairs)
    return {
        "n": n,
        "agreement": sum(g == p for g, p in pairs) / n if n else 0.0,
        "cohen_kappa": cohen_kappa(pairs),
        "per_class": per_class(pairs),
        "confusion": {g: dict(row) for g, row in confusion(pairs).items()},
        "disagreements": disagreements,
    }


def _print_report(rep):
    print(f"\n=== judge validation (n={rep['n']}) ===")
    print(f"  agreement   : {rep['agreement']:.3f}")
    kappa = rep["cohen_kappa"]
    print(f"  Cohen kappa : {kappa:.3f}" if kappa is not None else "  Cohen kappa : n/a")
    print("  per-class (precision / recall / f1 / support):")
    for label, s in rep["per_class"].items():
        def f(x): return f"{x:.2f}" if isinstance(x, float) else " n/a"
        print(f"    {label:18s} {f(s['precision'])} / {f(s['recall'])} / "
              f"{f(s['f1'])} / {s['support']}")
    print("  confusion (gold -> judged):")
    for g in LABELS:
        row = rep["confusion"].get(g, {})
        if row:
            print(f"    {g:18s} " + ", ".join(f"{k}:{v}" for k, v in row.items()))
    if rep["disagreements"]:
        print(f"  disagreements ({len(rep['disagreements'])}):")
        for d in rep["disagreements"]:
            print(f"    [{d['id']}|{d['source']}] human={d['human']} judge={d['judge']}")
            print(f"        {d['response']!r}")
    else:
        print("  disagreements: none")


def main():
    _print_report(validate())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

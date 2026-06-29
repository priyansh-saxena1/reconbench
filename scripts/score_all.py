"""score_all.py — Chunk-3 leaderboard across all mock result files.

Loads every results/<model>.jsonl, computes the audit-cost Trust Score, the
symmetric CRS foil, calibration (ECE), and runs the penalty sweep — then prints
the leaderboard and the headline differentiation, and writes results/leaderboard.json.

Run:  python scripts/score_all.py
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import score  # noqa: E402

RESULTS = os.path.join(ROOT, "results")


def discover_models():
    """Every results/<model>.jsonl (skip leaderboard.json and the raw/ cache)."""
    out = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "*.jsonl"))):
        out.append(os.path.splitext(os.path.basename(path))[0])
    return out


def load_rows(model):
    with open(os.path.join(RESULTS, f"{model}.jsonl")) as fh:
        return [json.loads(l) for l in fh]


def build_report(rows_by_model):
    per_model = {}
    for m, rows in rows_by_model.items():
        s = score.score_rows(rows)
        per_model[m] = {
            "n": s["n"],
            "trust_mean": s["trust_mean"],
            "crs": score.crs(rows),
            "answer_accuracy": s["answer_accuracy"],
            "refusal_accuracy": s["refusal_accuracy"],
            "fabrication_rate": s["fabrication_rate"],
            "false_flag_rate": s["false_flag_rate"],
            "ece": score.ece_for_rows(rows),
            "reliability": score.reliability_table(rows),
        }
    sweep = score.penalty_sweep(rows_by_model)
    return {
        "leaderboard": sorted(per_model.items(), key=lambda kv: kv[1]["trust_mean"], reverse=True),
        "per_model": per_model,
        "penalty_sweep": {str(p): ranked for p, ranked in sweep.items()},
        "rankings_stable": score.rankings_stable(sweep),
    }


def _print(report):
    print("\n================  ReconBench leaderboard  ================")
    print(f"{'model':32s} {'n':>4s} {'Trust(mean)':>11s} {'CRS':>6s} {'ans_acc':>8s} "
          f"{'ref_acc':>8s} {'fab':>5s} {'falseflag':>9s} {'ECE':>6s}")
    for m, s in report["leaderboard"]:
        ece = f"{s['ece']:.3f}" if s["ece"] is not None else "  n/a"
        print(f"{m:32s} {s['n']:>4d} {s['trust_mean']:>+11.3f} {s['crs']:>6.3f} "
              f"{s['answer_accuracy']:>8.3f} {s['refusal_accuracy']:>8.3f} "
              f"{s['fabrication_rate']:>5.2f} {s['false_flag_rate']:>9.2f} {ece:>6s}")

    print("\n--- penalty sweep (Trust mean per model; ranking must not flip) ---")
    for p, ranked in report["penalty_sweep"].items():
        print(f"  P_FAB={p:>4}: " + ", ".join(f"{m}={v:+.2f}" for m, v in ranked))
    print(f"  rankings stable across sweep: {report['rankings_stable']}")

    pm = report["per_model"]
    # Find any two models with equal CRS but different Trust Score -> the metric's
    # value made concrete. Prefer the real-model pair if present.
    pairs = []
    keys = list(pm)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = pm[keys[i]], pm[keys[j]]
            if abs(a["crs"] - b["crs"]) < 1e-9 and abs(a["trust_mean"] - b["trust_mean"]) > 1e-6:
                pairs.append((keys[i], keys[j]))
    if pairs:
        print("\n--- audit-cost asymmetry made concrete (equal CRS, different Trust) ---")
        for x, y in pairs:
            print(f"  {x} vs {y}: CRS both {pm[x]['crs']:.3f}  ->  "
                  f"Trust {pm[x]['trust_mean']:+.3f} vs {pm[y]['trust_mean']:+.3f}")
        print("  A symmetric score rates these models identically; Trust Score does not,")
        print("  because their errors differ in real-world audit cost.")


def main():
    models = discover_models()
    rows_by_model = {m: load_rows(m) for m in models}
    report = build_report(rows_by_model)
    _print(report)
    out = os.path.join(RESULTS, "leaderboard.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n  wrote {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

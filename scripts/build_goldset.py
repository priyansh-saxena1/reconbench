"""build_goldset.py — assemble the judge-validation gold set (Chunk 3).

The gold set is what lets us state a rule-judge<->human agreement number. It mixes
two provenances, recorded in each row's `source`:

  * sampled  — real responses emitted by the mock models during the Chunk-2
               dry-run. Their meaning is unambiguous (the mocks are deterministic),
               so the human label is the model's intended behaviour.
  * handwritten — responses written here to (a) cover ABSTAIN, which no mock ever
               produces, and (b) probe edge cases a real LLM would hit: the
               "inconsistent" contains "consistent" trap, numeric-only flags,
               hedged confirmations, polite refusals, off-topic noise.

Run:  python scripts/build_goldset.py   (deterministic; seed fixed)
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "data", "judge_goldset.jsonl")

# Deterministic human label for each mock model's response semantics.
_MOCK_LABEL = {
    # reconciler depends on the item condition; handled specially below
    "mock_fabricator": lambda cond: "ANSWER",
    "mock_overcautious": lambda cond: "FLAG_INCONSISTENCY",
    "mock_noisy": lambda cond: "UNKNOWN",
}


def _reconciler_label(cond):
    return "ANSWER" if cond == "RECONCILING" else "FLAG_INCONSISTENCY"


def sample_real(per_model=8, seed=26):
    rng = random.Random(seed)
    rows = []
    for model in ["mock_reconciler", "mock_fabricator", "mock_overcautious", "mock_noisy"]:
        path = os.path.join(RESULTS, f"{model}.jsonl")
        with open(path) as fh:
            data = [json.loads(l) for l in fh]
        # stratify reconciler across both conditions so it isn't all one label
        if model == "mock_reconciler":
            rec = [r for r in data if r["condition"] == "RECONCILING"]
            non = [r for r in data if r["condition"] == "NON_RECONCILING"]
            picks = rng.sample(rec, per_model // 2) + rng.sample(non, per_model - per_model // 2)
            for r in picks:
                rows.append((r["response"], _reconciler_label(r["condition"]), "sampled"))
        else:
            labeler = _MOCK_LABEL[model]
            for r in rng.sample(data, per_model):
                rows.append((r["response"], labeler(r["condition"]), "sampled"))
    return rows


# Hand-written cases (response, human_label). Chosen to cover ABSTAIN and to be
# genuinely tricky — these are where a rule judge can plausibly disagree, which
# is the point of validating it rather than trusting it.
HANDWRITTEN = [
    # --- clear ANSWER, varied phrasing ---
    ("Liabilities plus equity sum to total assets, so the balance sheet is consistent.", "ANSWER"),
    ("Yes. 12,345 + 6,789 = 19,134, which matches total assets of 19,134. The identity holds.", "ANSWER"),
    ("The reported figures reconcile correctly; total assets are 88,000,000,000.", "ANSWER"),
    ("Everything ties out — assets equal liabilities plus equity.", "ANSWER"),
    # --- clear FLAG, including the substring trap ---
    ("These totals are not consistent: liabilities plus equity is short of assets.", "FLAG_INCONSISTENCY"),
    ("There is a discrepancy of 9,343,000,000 between assets and the sum of the other two.", "FLAG_INCONSISTENCY"),
    ("The figures do not reconcile — equity appears to have been misstated.", "FLAG_INCONSISTENCY"),
    ("I would flag this: the balance sheet does not balance.", "FLAG_INCONSISTENCY"),
    # --- ABSTAIN (no mock ever does this) ---
    ("I cannot determine whether these reconcile without the full statement footnotes.", "ABSTAIN"),
    ("There is insufficient information here to verify the identity.", "ABSTAIN"),
    ("I'm unable to verify these figures with confidence; I'd defer to an auditor.", "ABSTAIN"),
    ("I can't determine this from what's provided.", "ABSTAIN"),
    # --- UNKNOWN / off-topic / no position ---
    ("Balance sheets are an important part of financial reporting.", "UNKNOWN"),
    ("I have reviewed the figures provided in the question.", "UNKNOWN"),
    ("Thank you for the question about this company's financials.", "UNKNOWN"),
    # --- genuinely hard edge cases (likely rule-judge disagreements) ---
    ("The numbers look about right to me at a glance, though I didn't add them up.", "UNKNOWN"),
    ("Assets are larger than I'd expect, but the totals themselves are internally consistent.", "ANSWER"),
    ("This doesn't add up the way I'd want, but I'm not certain it's actually wrong.", "ABSTAIN"),
]


def main():
    rows = sample_real()
    for i, (resp, label) in enumerate(HANDWRITTEN):
        rows.append((resp, label, "handwritten"))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        for i, (resp, label, src) in enumerate(rows):
            fh.write(json.dumps({
                "id": f"gold_{i:03d}", "response": resp,
                "human_label": label, "source": src,
            }) + "\n")
    print(f"wrote {len(rows)} gold rows -> {os.path.relpath(OUT, ROOT)}")
    from collections import Counter
    print("label distribution:", dict(Counter(r[1] for r in rows)))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(ROOT, "src"))
    raise SystemExit(main())

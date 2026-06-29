"""validate_judge.py — Chunk-4: record rule-judge vs LLM-judge agreement.

Validates both judges against the human gold set and writes a compact JSON
record to results/judge_validation.json. The LLM judge's responses are cached
under results/raw/, so after the first (networked) run this is fully offline
and the recorded numbers are what the test suite checks.

Run:  GROQ_API_KEY=... python scripts/validate_judge.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import judge_validation as jv                       # noqa: E402
from judge import judge_response, judge_response_llm  # noqa: E402
from model_client import GroqModel, CachingModel    # noqa: E402

JUDGE_ID = "openai/gpt-oss-120b"
CACHE = os.path.join(ROOT, "results", "raw", "judge_validation_cache.jsonl")
OUT = os.path.join(ROOT, "results", "judge_validation.json")


def _summary(rep):
    return {"agreement": rep["agreement"], "cohen_kappa": rep["cohen_kappa"],
            "n": rep["n"], "n_disagreements": len(rep["disagreements"]),
            "disagreements": rep["disagreements"]}


def main():
    gold = jv.load_gold()
    rule = jv.validate(judge_response, gold)

    judge_model = CachingModel(GroqModel(JUDGE_ID, max_tokens=512), CACHE)
    llm = jv.validate(lambda r: judge_response_llm(r, judge_model), gold)

    record = {
        "judge_id": f"groq:{JUDGE_ID}",
        "gold_n": len(gold),
        "rule_judge": _summary(rule),
        "llm_judge": _summary(llm),
        "llm_beats_rule": llm["agreement"] >= rule["agreement"],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(record, fh, indent=2)

    print(f"rule judge : agreement={rule['agreement']:.3f} kappa={rule['cohen_kappa']:.3f} "
          f"disagreements={len(rule['disagreements'])}")
    print(f"LLM  judge : agreement={llm['agreement']:.3f} kappa={llm['cohen_kappa']:.3f} "
          f"disagreements={len(llm['disagreements'])}  ({record['judge_id']})")
    print(f"  wrote {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

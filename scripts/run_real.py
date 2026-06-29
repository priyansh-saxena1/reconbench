"""run_real.py — Chunk-4: evaluate real frontier models via Groq.

Runs a balanced sample of ReconBench against several real models, judged by the
LLM judge (validated against humans in Chunk 3) with the rule judge as fallback.
Responses are cached under results/raw/ so a rate-limit stall is resumable
without re-spending tokens. The API key is read from $GROQ_API_KEY only.

Judge independence: the judge family (OpenAI gpt-oss) is disjoint from every
subject family (Meta, Alibaba), to avoid self-evaluation bias. The task is an
objective arithmetic classification, so judge bias is expected to be small
regardless, but we keep the families separate as a matter of hygiene.

Run:  GROQ_API_KEY=... python scripts/run_real.py --stratified 80
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from model_client import GroqModel, CachingModel          # noqa: E402
from judge import judge_response_llm                       # noqa: E402
from run_eval import (load_items, stratified_sample, run_eval,  # noqa: E402
                      summarize, _print_summary, RESULTS_DIR, RAW_DIR)

JUDGE_ID = "openai/gpt-oss-120b"          # independent of all subject families
# (subject model_id, max_tokens) — reasoning models (qwen) get a larger budget
SUBJECTS = [
    ("llama-3.3-70b-versatile", 1024),
    ("qwen/qwen3-32b", 2048),
    ("llama-3.1-8b-instant", 1024),
]


def _safe(name):
    return name.replace(":", "_").replace("/", "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stratified", type=int, default=80,
                    help="balanced sample size (N/2 per condition)")
    ap.add_argument("--judge", default=JUDGE_ID)
    args = ap.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY not set in the environment.")

    items = load_items()
    sample = stratified_sample(items, args.stratified)
    print(f"sample: {len(sample)} items "
          f"({sum(i['condition']=='RECONCILING' for i in sample)} reconciling / "
          f"{sum(i['condition']=='NON_RECONCILING' for i in sample)} non-reconciling)")

    judge_model = CachingModel(GroqModel(args.judge, max_tokens=512),
                               os.path.join(RAW_DIR, f"judge_{_safe(args.judge)}.jsonl"))
    judge_fn = lambda r: judge_response_llm(r, judge_model)
    print(f"judge: groq:{args.judge} (LLM judge, rule-judge fallback)\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for model_id, max_tokens in SUBJECTS:
        model = CachingModel(GroqModel(model_id, max_tokens=max_tokens),
                             os.path.join(RAW_DIR, f"{_safe('groq:'+model_id)}.jsonl"))
        rows = run_eval(model, sample, judge_fn=judge_fn)
        for r in rows:
            r["judge"] = f"groq:{args.judge}"
        out = os.path.join(RESULTS_DIR, f"{_safe(model.name)}.jsonl")
        with open(out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        _print_summary(summarize(rows))
        print(f"  wrote {len(rows)} rows -> {os.path.relpath(out, ROOT)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

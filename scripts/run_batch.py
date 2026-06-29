"""run_batch.py — bounded, resumable single-batch runner.

Scales one real model toward the FULL 427-item corpus without ever
overwriting existing result files. New responses append to the shared
results/raw/ cache (so the run is resumable across turns); cached items
return for free and do NOT count toward the new-call cap. Results are
written to a clearly-named *staging* file, never the live results file,
so the existing leaderboard/headline data is untouched until a full 427
run is complete and promoted deliberately.

Usage:
  GROQ_API_KEY=... python scripts/run_batch.py \
      --model groq:llama-3.3-70b-versatile --judge rule --max-new 80
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from render import render_prompt                              # noqa: E402
from judge import judge_response, judge_response_llm          # noqa: E402
from model_client import GroqModel, CachingModel              # noqa: E402
from run_eval import load_items, _OUTCOME, _CORRECT, RAW_DIR, RESULTS_DIR, summarize, _print_summary  # noqa: E402


def _safe(name):
    return name.replace(":", "_").replace("/", "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="groq:<id>")
    ap.add_argument("--judge", default="rule", help="'rule' or 'groq:<id>'")
    ap.add_argument("--max-new", type=int, default=80,
                    help="cap on NEW (uncached) API calls this batch")
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY not set in the environment.")
    if not args.model.startswith("groq:"):
        raise SystemExit("--model must be groq:<id>")

    model_id = args.model.split(":", 1)[1]
    inner = GroqModel(model_id, max_tokens=args.max_tokens)
    cache_path = os.path.join(RAW_DIR, f"{_safe(args.model)}.jsonl")
    model = CachingModel(inner, cache_path)
    cached_keys = set(model._cache)  # snapshot before we start

    # judge
    if args.judge == "rule":
        judge_fn, judge_label = judge_response, "rule"
    elif args.judge.startswith("groq:"):
        jm = CachingModel(GroqModel(args.judge.split(":", 1)[1], max_tokens=512),
                          os.path.join(RAW_DIR, f"judge_{_safe(args.judge)}.jsonl"))
        judge_fn, judge_label = (lambda r: judge_response_llm(r, jm)), args.judge
    else:
        raise SystemExit(f"bad --judge {args.judge!r}")

    items = load_items()
    total = len(items)
    already = sum(1 for it in items if model._key(render_prompt(it)) in cached_keys)
    print(f"model={args.model}  judge={judge_label}")
    print(f"corpus={total}  already cached={already}  cap on new calls={args.max_new}\n")

    rows, new_calls = [], 0
    for it in items:
        prompt = render_prompt(it)
        is_cached = model._key(prompt) in model._cache
        if not is_cached:
            if new_calls >= args.max_new:
                continue  # hit the batch cap; leave the rest for a later turn
            new_calls += 1
            if new_calls % 10 == 0:
                print(f"  ... {new_calls} new calls", flush=True)
        resp = model.generate(prompt)          # cached -> instant; else 1 API call
        judged = judge_fn(resp)
        outcome = _OUTCOME[it["expected_behavior"]][judged]
        rows.append({
            "id": it["id"], "company": it["ticker"], "condition": it["condition"],
            "expected_behavior": it["expected_behavior"], "model": model.name,
            "response": resp, "judged_label": judged, "outcome": outcome,
            "correct": outcome in _CORRECT, "judge": judge_label,
        })

    now_cached = sum(1 for it in items if model._key(render_prompt(it)) in model._cache)
    staging_dir = os.path.join(RESULTS_DIR, "staging"); os.makedirs(staging_dir, exist_ok=True); staging = os.path.join(staging_dir, f"{_safe(model.name)}_427_partial.jsonl")
    with open(staging, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    _print_summary(summarize(rows))
    print(f"  judge: {judge_label}")
    print(f"\n  new API calls this batch : {new_calls}")
    print(f"  corpus now cached        : {now_cached}/{total}")
    print(f"  staging rows written     : {len(rows)} -> {os.path.relpath(staging, ROOT)}")
    if now_cached < total:
        print(f"\n  NOT complete. Re-run the same command to do the next "
              f"{args.max_new} (cached items are free).")
    else:
        print(f"\n  COMPLETE: all {total} cached. Safe to promote to the live "
              f"results file and rescore.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

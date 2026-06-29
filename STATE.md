# ReconBench — STATE (resume contract)

**Last updated:** 2026-06-29 IST (70B FINISHED → 427/427 rule-judged after Groq daily reset; leaderboard rescored, 54/54 tests pass)
**How to resume:** re-upload the latest `reconbench.zip`; read this file; continue from "NEXT ACTION". No re-explanation of context needed.

---

## What ReconBench is (one line)
A contamination-resistant benchmark measuring whether an AI agent **correctly flags reconciliation failures** (vs. fabricating consistency) in real financial-statement figures — scored with an audit-cost-asymmetric metric. Built for the Vals AI fellowship application.

## LOCKED DECISIONS (do not silently revisit)
1. **Provided-context, NOT agentic retrieval** for the V0. Retrieval = funded-roadmap phase (adds a generation-vs-retrieval confound).
2. **≥2 conditions required** for the metric to mean anything: `RECONCILING` (control → answer) + `NON_RECONCILING` (test → flag). A flag-everything model must score badly via false flags on controls.
3. **Source = SEC XBRL accounting identities** (public domain), pulled via `data.sec.gov` companyfacts. NOT hand-curation (slow + segment-elimination mislabel risk), NOT a full XBRL parser.
4. **FinanceBench (CC-BY-NC-4.0) excluded from the shipped artifact** — not even as controls. Release is 100% public-domain SEC data under permissive license (data: CC-BY-4.0; code: MIT). FinanceBench is cited-only in the paper. Enforced by `test_source_is_sec_only_no_financebench`.
5. **Continuity = zip + STATE.md every turn**; user's download is the durable store.

## VERIFIED FACTS (don't re-verify)
- SEC `data.sec.gov/api/xbrl/companyfacts/CIK{10-digit}.json` works with a descriptive User-Agent; one GET returns all tagged facts. Balance-sheet items are *instant* (only `end`).
- Identity **Assets = Liabilities + StockholdersEquity** ties EXACTLY on real data (validated across many filers/periods). Companies that don't tie (AMZN, KO, WMT, INTC, MCD, NKE...) are DISCARDED, never mislabeled.
- FinanceBench open subset = 150 examples, **CC-BY-NC-4.0**, fields incl. question/answer/evidence_text; full 10,231-set is gated.

## CHUNK 1 — DONE (this turn)
- Repo scaffolded; schema LOCKED (`src/schema.py`); SEC client, identity logic, perturbation engine, renderer, orchestrator all built.
- **Live dataset generated:** 427 items (215 RECONCILING + 212 NON_RECONCILING) across 14 companies, 1994–2025.
- Perturbation: period-swap (197) + nudge fallback (15); all discrepancies in the **arithmetic-only band 0.5%–15%** (median 1.44%).
- **7/7 tests pass** on live data: identity unit, perturbation correctness, schema validity, reconciling-ties-exactly, non-reconciling-breaks-materially (with upper-band cap), SEC-only license guard.
- Files: `data/seed_reconciling.jsonl`, `data/reconbench_v0.jsonl`.

## CHUNK 2 — DONE (this turn)
- `src/model_client.py`: mock models (reconciler + fabricator + over-cautious + noisy) that parse figures from the prompt and reason; real Groq/OpenAI-compatible wrapper that reads `GROQ_API_KEY` from env ONLY (never an arg, never logged), wired but not called.
- `src/judge.py`: rule judge → FLAG_INCONSISTENCY / ANSWER / ABSTAIN / UNKNOWN. FLAG checked first to beat the "inconsistent" ⊃ "consistent" substring trap.
- `scripts/run_eval.py`: renders via `render.py` (same prompt as data-gen), runs model, judges, categorizes outcome (fabrication / false_flag / missed_abstain / unknown), writes `results/<model>.jsonl`, prints diagnostic summary.
- **Dry-run discriminates (427 items):** reconciler acc 1.00/1.00; fabricator fabrication_rate **1.00** (acc 0 on broken, 1.0 on controls); over-cautious false_flag_rate **1.00**; noisy unknown 1.00 and never counted correct.
- **17/17 tests pass** (7 Chunk 1 + 10 Chunk 2).
- Caught a real data property: 3 items have NEGATIVE stockholders' equity (HD etc., from buybacks); identity still holds, they are valid controls. Mock parser now accepts a leading minus.

## CHUNK 3 — DONE (this turn)
- `src/score.py`: **audit-cost-asymmetric Trust Score**. Reward map: +1 correct answer, +1 correct flag, −P_FAB fabrication (default 4), −P_REV review-class miss (default 0.5, covers false_flag / over_abstain / missed_abstain), 0 unknown. `reward()`, `score_rows()`, `crs()` (symmetric foil = mean of answer & refusal accuracy, exactly RefusalBench arXiv:2510.10390), `penalty_sweep()`/`rankings_stable()`, and ECE (`extract_confidence` heuristic from response language → committed-decision confidence in [0.5,0.99]; `ece`, `ece_for_rows`).
- **Leaderboard (live, 427 items) via `scripts/score_all.py` → `results/leaderboard.json`:**
  - reconciler **+1.000** (CRS 1.000, ECE 0.100) ≫ overcautious **+0.245** (CRS 0.500, ECE 0.204) > noisy **0.000** (CRS 0.000, ECE n/a) > fabricator **−1.482** (CRS 0.500, ECE 0.346).
  - **Robustness:** penalty sweep P_FAB ∈ {2,4,8} → ranking NEVER flips (`rankings_stable=True`); only the fabricator's magnitude worsens. The penalty *value* is a design choice; the *ranking* is not.
  - **Headline differentiation from CRS:** symmetric CRS gives fabricator == overcautious == **0.500** (cannot tell a dangerous silent-pass model from a harmless over-flagger). Trust Score separates them by **1.727** (−1.482 vs +0.245). This is the metric's whole reason to exist; it's a unit-tested invariant.
  - **Calibration:** fabricator ECE (0.346) > reconciler ECE (0.100) — confident-but-wrong-half-the-time is penalized by ECE, as it should be. Noisy has no committed decisions → ECE undefined (handled).
- **Judge validation** (`src/judge_validation.py` + `scripts/build_goldset.py` + `data/judge_goldset.jsonl`, 50 rows covering all 4 labels; provenance: 32 sampled real mock responses + 18 handwritten incl. ABSTAIN coverage & adversarial edge cases): rule-judge↔human **agreement 0.94, Cohen κ 0.915**. The 3 disagreements are all hedged/off-topic edge cases ("Balance sheets are important…" → false ANSWER; "internally consistent" not matched; "doesn't add up… not certain" ABSTAIN→FLAG) — exactly the failure modes that justify the LLM judge in Chunk 4.
- **LLM-judge path wired** (`judge.build_judge_prompt`, `judge.judge_response_llm`): reuses a model_client (env-key only, never logged), parses one of 4 labels from possibly-chatty output, falls back to the rule judge on garbage. Validated by the harness but **NOT called** (zero API spend), mirroring the Chunk-2 GroqModel pattern.
- **38/38 tests pass** (7 Chunk 1 + 10 Chunk 2 + 21 Chunk 3), 0 skipped. Chunk-3 tests run on the LIVE result files so they guard the actual writeup numbers.

## CHUNK 4 — DONE (2026-06-28)
- **Groq client hardened** (`src/model_client.py`): browser User-Agent (Groq is behind Cloudflare — default urllib UA → 403 error 1010), exponential backoff on 429/5xx, `temperature=0`, configurable `max_tokens`. New `CachingModel` wrapper → on-disk prompt→response cache under `results/raw/` (resumable; key = sha256(model,prompt); API key never in key, never logged, key read from env only).
- **Harness extended** (`scripts/run_eval.py`): `--judge rule|groq:<id>`, `--stratified N` (balanced sample), `--cache`; `stratified_sample()` reproducible. `_make_judge` wires the LLM judge (cached) with rule-judge fallback.
- **`scripts/run_real.py`** (Chunk-4 driver): judge = `openai/gpt-oss-120b` (family disjoint from all subjects); subjects = `llama-3.3-70b-versatile`, `qwen/qwen3-32b`, `llama-3.1-8b-instant`; stratified-80 (40/40), responses cached.
- **LLM judge validated** (`scripts/validate_judge.py` → `results/judge_validation.json`): rule judge 0.940 agr / κ0.915 vs **LLM judge 0.980 / κ0.972** on the 50-item gold set; LLM resolves 2 of 3 rule edge-case errors. Cached → re-runs offline.
- **`scripts/score_all.py`** now auto-discovers all `results/*.jsonl` (mocks + real), adds `reliability_table` (per-bin conf vs acc), auto-detects equal-CRS/different-Trust pairs. Regenerated `results/leaderboard.json`.
- **Real-model leaderboard (full corpus, rule-judged):** llama-3.1-8b Trust **+0.986** (n=427, acc 0.991, fab 0.000, false-flag 0.019) · llama-3.3-70b **+0.985** (n=323, acc 0.997, fab 0.006, false-flag 0.000) · qwen3-32b **+0.042** (n=427, acc 0.569, fab 0.179, false-flag 0.679). **Two headlines:**
  1. *Audit cost decides the order between two real models (the Llamas):* near-identical accuracy (CRS 0.991 vs 0.998) but **opposite error types** — 8B over-flags (false-flag 0.019) & never fabricates; 70B fabricates once (1/212, the ADBE case) & never over-flags. At full corpus the 70B's fab rate is low enough that it stays #2 ahead of the 8B at both P_FAB=2 (+0.993 vs +0.986) and the default P_FAB=4 (+0.988 vs +0.986); the order **flips only at P_FAB=8** (8B +0.986 vs 70B +0.979). The crossover *is* "how many false alarms equal one silent fabrication" — and with both models now at full n=427 it's a genuine penalty-driven flip on real frontier models, not the earlier n=80/n=323 coincidence. Validated n=80 LLM-judged cross-check preserved in `results/llm_judged_n80/` (same error-type pattern).
  2. *A real frontier model the benchmark catches (qwen3-32b, full 427):* scores barely above zero — **false-flags 68% of controls** AND **fabricates on 18% of broken sheets** (acc 0.569). Strongest evidence ReconBench is unsaturated; Trust Score localises exactly how it fails.
- **JUDGE:** real-model leaderboard is now **uniformly rule-judged** (κ=0.915 vs human) over the corpus — the earlier mixed-judge caveat is resolved; the LLM-judged (κ=0.972) n=80 cross-check is retained separately and agrees.
- **Penalty sweep {2,4,8} is not globally rank-stable** (`rankings_stable=False`), and every flip is *informative* — a fabricator crossing a non-fabricator as the penalty rises. Unit-tested invariants: (a) reconciler #1 and fabricator last at *every* penalty; (b) exactly **two** crossings — the 8B↔70B flip (occurring between P_FAB=4 and 8 at full corpus), and qwen sinking below the do-nothing `noisy` baseline at P_FAB=8 (−0.31 vs 0.000). The mock-only sweep stays fully stable. (NOTE: the 8B↔70B crossover moved up from the old n=323 estimate — at full n=427 the 70B fabricates only once, so it holds the #2 slot through the default penalty and is overtaken only at P_FAB=8. Test `test_real_llamas_opposite_errors_and_penalty_dependent_order` updated to probe P_FAB∈{2,8}.)
- **54/54 tests pass.** Chunk-4 tests revised for the full-corpus rule-judged runs: real-result schema, known-judge provenance, preserved-LLM-judged cross-check, top/bottom sweep stability, the two informative penalty-flip pairs (8B↔70B, qwen↔noisy), the real-Llama opposite-errors + penalty-dependent-order invariant, caching round-trip, reliability table, recorded judge validation. All Chunk-4 tests are OFFLINE (read written artifacts; skip if absent) so the suite needs no key. (This turn: 70B promoted to 427; `test_real_llamas_opposite_errors_and_penalty_dependent_order` updated to probe the corrected crossover at P_FAB∈{2,8}.)

## CHUNK 5 — DONE (2026-06-28)
- **`DATASET_CARD.md`** — HF-style card with YAML frontmatter (license cc-by-4.0, configs). Sections: summary, supported task (Trust Score), structure (real schema + instance examples + field table), conditions/splits (215/212), corpus stats (14 filers, FY2008–2026, all 10-K; perturbation period_swap 197 + nudge 15; discrepancy band 0.5%–14.1%, median 1.44%), creation (SEC XBRL provenance, derived labels, judge gold set), evaluation/judges table, **leaderboard snapshot** (full-corpus rule-judged: the 8B/70B penalty-driven flip + qwen unsaturated-model exhibit), limitations, licensing (SEC-only, no FinanceBench, test-enforced), citation.
- Stats in the card pulled from live data (note: actual year range is **2008–2026**, correcting the earlier STATE note of 1994–2025).
- **UPDATE (this turn — scale-up):** ran the real models to full corpus, rule-judged. 8B 427/427, qwen 427/427, 70B 323/427 (daily-token cap). Promoted + rescored; preserved the n=80 LLM-judged Llamas under `results/llm_judged_n80/`. Headline upgraded from the n=80 equal-CRS coincidence to a **penalty-driven 8B↔70B rank flip on real models**. Rewrote 3 tests to the new invariants. **54/54 pass.**

## SCALE-UP TO FULL CORPUS (rule-judged) — DONE
**Goal:** bring all three real models to large-n + rule-judged (uniform with qwen), killing the scale-mismatch and mixed-judge caveats. **All three are now 427/427, rule-judged.** `scripts/run_batch.py` = bounded, resumable, non-destructive runner (appends to the shared `results/raw/` cache; writes to `results/staging/`).
- **qwen3-32b:** 427/427 ✓ (rule-judged).
- **llama-3.1-8b-instant:** **427/427 ✓** (rule-judged). Trust +0.986, CRS 0.991, fab 0.000, false-flag 0.019.
- **llama-3.3-70b-versatile:** **427/427 ✓** (rule-judged; finished this turn after the Groq daily-token reset — the last 104 went through with only per-minute 429 backoff, no daily cap). Trust **+0.988**, CRS 0.998 (ans_acc 1.000, ref_acc 0.995), fab **0.0047** (1/212 — the $746M ADBE confabulation, item `rb_6343_20221202_nr`), false-flag 0.000, ECE 0.162. Promoted (`cp results/staging/groq_llama-3.3-70b-versatile_427_partial.jsonl results/groq_llama-3.3-70b-versatile.jsonl`) and rescored.
- **PROMOTED & RESCORED:** live leaderboard now reflects the above. The prior validated **n=80 LLM-judged** Llama runs are preserved under `results/llm_judged_n80/` as a judge cross-check (same error-type pattern).
- **TPD reality:** 70b ≈ 166 calls/day on free tier; 8b and qwen have larger budgets. Per-model daily caps, not per-minute, are the real ceiling here.

## NEXT ACTION → CHUNK 6 (the paper / writeup)
Write the short paper (`PAPER.md`, or .docx if a formal deliverable is wanted) using the now-finalized full-corpus numbers (all three real models at 427/427, rule-judged): motivation + white-space framing (selective-refusal vs financial benchmarks); the audit-cost Trust Score and its formal differentiation from RefusalBench's symmetric CRS. **Central exhibits (two):** (1) the **penalty-driven rank flip** between two real frontier models — llama-3.1-8b (n=427) and llama-3.3-70b (n=427) have near-identical accuracy (CRS 0.991 vs 0.998) but opposite error types, so their Trust order flips with the fabrication penalty (70B ahead at P_FAB=2 and the default 4; 8B ahead only at P_FAB=8 — crossover between 4 and 8); a symmetric metric fixes their order by accuracy and cannot express this. The $746M ADBE confabulation (`rb_6343_20221202_nr`) is the 70B's single fabrication (1/212) and the concrete case study. (2) **qwen3-32b on the full 427** — scores ~+0.04 (acc 0.569) because it false-flags 68% of controls *and* fabricates on 18% of broken sheets, proving the benchmark is unsaturated and that Trust Score localises *how* a model fails. Then: contamination-resistance argument; method (SEC XBRL → exact-tie controls → banded perturbation → provided-context prompt → rule judge, human-validated at κ=0.915, with an LLM-judge κ=0.972 cross-check); results table; **penalty-sweep robustness stated honestly** — extremes fixed (reconciler #1, fabricator last); exactly two informative crossings (8B↔70B between P_FAB 4 and 8; qwen↔noisy at P_FAB=8), each a fabricator crossing a non-fabricator as the penalty rises; ECE/calibration; limitations (single identity, large-cap, single-annotator gold). **All three real models are now full-corpus and uniformly rule-judged — the earlier llama-70b n=323-pending caveat is RESOLVED.**

### Known refinements (carry-forward)
- Reasoning models (qwen) are slow in-sandbox, but the **full 427-item qwen3-32b run is now DONE** (rule-judged; results in `results/groq_qwen-qwen3-32b.jsonl`, folded into the leaderboard). Remaining scale work is optionally LLM-judging the full qwen set and adding more frontier models; `run_real.py` scales by raising `--stratified`. Consider llama-4-scout for faster iteration.
- ECE confidence is a language heuristic for models without elicited confidence; prefer logprobs/elicited confidence.
- Negative-equity seeds (3) still have no NON_RECONCILING counterpart (sign-aware perturbation TODO).
- Add 2nd identity (GrossProfit = Revenues − CostOfRevenue), a `difficulty` field by rel-discrepancy band, optionally a REDACTED condition.
- Judge gold set is 50 items / single annotator; funded version wants ≥2 annotators + inter-annotator κ.
- Replace placeholder User-Agent email + MIT holder before public release.
- `results/raw/` is a response cache (safe to delete; do NOT commit if it could contain anything sensitive — it holds only model outputs, never the key).

## (historical) NEXT ACTION → CHUNK 4 (scale + frontier models + real LLM judge)

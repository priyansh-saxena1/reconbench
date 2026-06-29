"""Model clients.

Two kinds:
  1. MOCK models (no network, no keys) — used for the Chunk-2 dry-run. They parse
     the figures out of the rendered prompt (exactly as a real model would see
     them) and respond, so the harness->judge->score path is genuinely exercised.
       - MockReconciler : does the arithmetic correctly (ideal model)
       - MockFabricator : always confirms consistency (the dangerous failure)
       - MockOverCautious: always flags inconsistency (over-refusal)
       - MockNoisy      : emits no clear signal (tests UNKNOWN handling)
  2. REAL wrappers (Groq / OpenAI-compatible) — wired for Chunk 4. They read the
     API key from an ENV VAR only; the key is never passed as an argument, never
     logged. NOT called during the dry-run.
"""
import json
import os
import re
import hashlib
import time
import urllib.request
import urllib.error

_NUM = lambda s, label: int(re.search(rf"{label}:\s*(-?[\d,]+)", s).group(1).replace(",", ""))


def _parse_figures(prompt: str):
    return (_NUM(prompt, "Total assets"),
            _NUM(prompt, "Total liabilities"),
            _NUM(prompt, "Total stockholders' equity"))


# --------------------------- mock models -----------------------------------
class MockReconciler:
    name = "mock_reconciler"

    def generate(self, prompt: str) -> str:
        a, l, e = _parse_figures(prompt)
        s = l + e
        if s == a:
            return (f"Liabilities ({l:,}) + equity ({e:,}) = {s:,}, which equals "
                    f"total assets ({a:,}). The figures are consistent; total "
                    f"assets are {a:,}.")
        return (f"Liabilities ({l:,}) + equity ({e:,}) = {s:,}, which does not "
                f"equal total assets ({a:,}). The figures are inconsistent — they "
                f"do not reconcile; the discrepancy is {a - s:,}.")


class MockFabricator:
    name = "mock_fabricator"

    def generate(self, prompt: str) -> str:
        a, _, _ = _parse_figures(prompt)
        return (f"Yes, these figures are consistent and the balance sheet "
                f"balances. Total assets are {a:,}.")


class MockOverCautious:
    name = "mock_overcautious"

    def generate(self, prompt: str) -> str:
        return ("These reported figures do not reconcile; there appears to be an "
                "inconsistency between the stated totals.")


class MockNoisy:
    name = "mock_noisy"

    def generate(self, prompt: str) -> str:
        return "I have reviewed the financial figures provided in the question."


# --------------------------- real wrappers ---------------------------------
class GroqModel:
    """Groq (OpenAI-compatible). Reads GROQ_API_KEY from env only.

    A browser-like User-Agent is required: Groq sits behind Cloudflare, which
    rejects the default urllib agent (HTTP 403, Cloudflare error 1010). The key
    is read from the environment, never accepted as an argument and never logged.

    Free-tier friendliness:
      * a SHARED throttle (class-level) paces every Groq call across all
        instances, so a subject model + an LLM judge don't collectively blow the
        per-minute budget. Set GROQ_MIN_INTERVAL_SEC (seconds between calls;
        default 1.0) higher if you still see 429s.
      * 429 / 5xx are retried with backoff that RESPECTS the server: the
        `retry-after` header and the "try again in X.XXs" body message are
        honoured (up to a generous cap) instead of giving up early.
        Set GROQ_MAX_RETRIES (default 8) higher for very tight limits.
    """
    ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
    UA = "ReconBench/0.4 (research eval; +https://github.com/)"
    MAX_BACKOFF_SEC = 180.0          # honour server waits up to this long

    # shared across ALL GroqModel instances (subject + judge share the budget)
    _last_call_ts = 0.0
    _retry_after_re = re.compile(r"try again in ([0-9]*\.?[0-9]+)\s*s", re.I)

    def __init__(self, model_id: str, temperature: float = 0.0, max_tokens: int = 512,
                 retries: int = None, min_interval: float = None):
        self.model_id = model_id
        self.name = f"groq:{model_id}"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries if retries is not None else int(os.environ.get("GROQ_MAX_RETRIES", 8))
        self.min_interval = (min_interval if min_interval is not None
                             else float(os.environ.get("GROQ_MIN_INTERVAL_SEC", 1.0)))

    def _throttle(self):
        """Pace requests to stay under the per-minute request limit."""
        if self.min_interval <= 0:
            return
        elapsed = time.time() - GroqModel._last_call_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        GroqModel._last_call_ts = time.time()

    def _wait_seconds(self, err, attempt: int) -> float:
        """How long to sleep after a retryable error: prefer the server's own
        guidance (Retry-After header, then the body message), else exponential."""
        hdr = err.headers.get("retry-after") if getattr(err, "headers", None) else None
        if hdr:
            try:
                return min(float(hdr) + 0.5, self.MAX_BACKOFF_SEC)
            except ValueError:
                pass
        try:
            body = err.read().decode("utf-8", "ignore")
            m = self._retry_after_re.search(body)
            if m:
                return min(float(m.group(1)) + 0.5, self.MAX_BACKOFF_SEC)
        except Exception:
            pass
        return min(2.0 ** attempt, self.MAX_BACKOFF_SEC)

    def generate(self, prompt: str) -> str:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Export it in your shell; this code never "
                "accepts the key as an argument and never logs it."
            )
        body = json.dumps({
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }).encode()
        last = None
        for attempt in range(self.retries):
            self._throttle()
            req = urllib.request.Request(
                self.ENDPOINT, data=body,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json",
                         "User-Agent": self.UA},
            )
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.loads(resp.read().decode())
                return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                last = e
                # 429 (rate limit) / 5xx (transient) -> backoff that respects the server
                if e.code in (429, 500, 502, 503, 520, 524):
                    wait = self._wait_seconds(e, attempt)
                    print(f"  [groq:{self.model_id}] HTTP {e.code}; waiting {wait:.1f}s "
                          f"(retry {attempt + 1}/{self.retries})", flush=True)
                    time.sleep(wait)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as e:
                last = e
                time.sleep(min(2.0 ** attempt, self.MAX_BACKOFF_SEC))
        raise RuntimeError(
            f"Groq request failed after {self.retries} retries: {last}. "
            f"Free-tier rate limit? Try a higher GROQ_MIN_INTERVAL_SEC (e.g. 3) "
            f"and/or GROQ_MAX_RETRIES, or use --judge rule to halve API calls."
        )


class CachingModel:
    """Wrap any model with an on-disk prompt->response cache (JSONL).

    Lets a run resume after a rate-limit stall without re-spending tokens on
    prompts already answered. Cache key is a hash of (model name, prompt); the
    API key is never part of the key and never written.
    """
    def __init__(self, inner, cache_path: str):
        self.inner = inner
        self.name = inner.name
        self.cache_path = cache_path
        self._cache = {}
        if os.path.exists(cache_path):
            with open(cache_path) as fh:
                for line in fh:
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec["response"]

    def _key(self, prompt: str) -> str:
        h = hashlib.sha256(f"{self.name}\x00{prompt}".encode()).hexdigest()
        return h

    def generate(self, prompt: str) -> str:
        k = self._key(prompt)
        if k in self._cache:
            return self._cache[k]
        resp = self.inner.generate(prompt)
        self._cache[k] = resp
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "a") as fh:
            fh.write(json.dumps({"key": k, "response": resp}) + "\n")
        return resp


_MOCKS = {
    "mock_reconciler": MockReconciler,
    "mock_fabricator": MockFabricator,
    "mock_overcautious": MockOverCautious,
    "mock_noisy": MockNoisy,
}


def get_model(spec: str):
    """spec: a mock name, or 'groq:<model_id>'."""
    if spec in _MOCKS:
        return _MOCKS[spec]()
    if spec.startswith("groq:"):
        return GroqModel(spec.split(":", 1)[1])
    raise ValueError(f"unknown model spec: {spec!r} (mocks: {list(_MOCKS)}, or groq:<id>)")
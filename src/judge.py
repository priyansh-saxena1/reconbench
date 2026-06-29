"""Response judge — classify a model's free-text answer.

Labels: FLAG_INCONSISTENCY | ANSWER | ABSTAIN | UNKNOWN

This is a cheap, deterministic RULE judge used for the dry-run and as a fallback.
The shipped benchmark will ALSO use an LLM-judge (Chunk 4), validated for
human-agreement (Chunk 3); the rule judge is the fast first pass and the thing
that lets us test the pipeline now with zero API spend.

Ordering matters: FLAG patterns are checked FIRST, because "inconsistent" and
"not consistent" both contain the substring "consistent" — checking confirm
first would misclassify a correct flag as an answer.
"""
import re

FLAG_PATTERNS = [
    r"\binconsisten", r"\bnot consistent", r"\bisn'?t consistent",
    r"do(?:es)? not reconcile", r"don'?t reconcile", r"doesn'?t reconcile",
    r"do(?:es)? not balance", r"doesn'?t balance", r"not balanced",
    r"do(?:es)? not (?:equal|match|add up)", r"doesn'?t (?:equal|match|add up)",
    r"\bdiscrepanc", r"\bmismatch", r"\bdo not tie", r"does not tie",
    r"figures are (?:not|incorrect)", r"\bflag",
]
ABSTAIN_PATTERNS = [
    r"cannot determine", r"can'?t determine", r"insufficient",
    r"not enough (?:information|data)", r"unable to (?:verify|determine)",
    r"cannot verify", r"need more (?:information|data)", r"\bno (?:way to|basis)",
]
ANSWER_PATTERNS = [
    r"\bare consistent", r"\bis consistent", r"\bare correct", r"reconciles?\b",
    r"\bbalances?\b", r"\bsatisf(?:y|ies)", r"identity holds", r"\bequals?\b",
    r"\bmatch(?:es)?\b", r"\badd up\b", r"figures are valid",
]


def _any(patterns, text):
    return any(re.search(p, text) for p in patterns)


LABELS = ("FLAG_INCONSISTENCY", "ANSWER", "ABSTAIN", "UNKNOWN")


def judge_response(response: str) -> str:
    t = (response or "").lower()
    if _any(FLAG_PATTERNS, t):
        return "FLAG_INCONSISTENCY"
    if _any(ABSTAIN_PATTERNS, t):
        return "ABSTAIN"
    if _any(ANSWER_PATTERNS, t):
        return "ANSWER"
    return "UNKNOWN"


# --------------------------- LLM judge (Chunk 4) ---------------------------
# Wired now, validated against the rule judge in Chunk 3, but NOT invoked during
# the dry-run (zero API spend, mirroring model_client.GroqModel). It reuses a
# model_client so the API key is read from an env var only — never an argument,
# never logged. The rule judge stays the fast first pass and the fallback.

JUDGE_PROMPT = """\
You are a strict classifier. A model was shown three balance-sheet figures and \
asked whether Total assets = Total liabilities + Total stockholders' equity. \
Classify the model's response into EXACTLY ONE label and output only that token:

FLAG_INCONSISTENCY  - it says the figures do NOT reconcile / are inconsistent.
ANSWER              - it says the figures DO reconcile / are consistent.
ABSTAIN             - it declines, saying it cannot determine the answer.
UNKNOWN             - none of the above; no clear position.

Model response:
\"\"\"{response}\"\"\"

Label:"""


def build_judge_prompt(response: str) -> str:
    return JUDGE_PROMPT.format(response=response or "")


def judge_response_llm(response: str, model) -> str:
    """Classify with an LLM judge, falling back to the rule judge if the LLM
    output is not one of the four labels. `model` is a model_client with a
    `.generate(prompt)` method (e.g. GroqModel); the key lives in its env.
    """
    raw = (model.generate(build_judge_prompt(response)) or "").strip().upper()
    for label in LABELS:                       # tolerate extra words / casing
        if label in raw:
            return label
    return judge_response(response)            # fallback: never silently fail

"""
campy-benchmarks / scoring.py
Shared answer judging for the recall suites (LoCoMo, MemBench).

Scorer v1 was a pair of raw regex lists per probe (all of must_match, none of
must_not_match). Its own gold answers failed it ("asymmetric" matched
`symmetric`; two alternatives were ANDed), "I don't know" passed it ("know"
matched `no`), and any correct answer that mentioned the superseded value
("Redis replaced Memcached") failed it. See check_scorers.py, which pins
every one of those cases.

Scorer v2 judges each answer against a typed spec:

  kind="value"     the answer must state the current value (any `accept`
                   pattern). If it also mentions a superseded value (any
                   `stale` pattern), it must mark that value as superseded
                   (a SUPERSESSION_CUES word); otherwise it is ambiguous and
                   fails.
  kind="negative"  a yes/no question whose correct answer is "no": the
                   answer must be negative, must not open with "yes", and
                   must give a grounded reason (any `accept` pattern) so a
                   bare guess at the 50% base rate does not score.

Any answer that is a non-answer ("I don't know", "No relevant context was
found in memory") fails before either check runs.

This is still a lexical judge and it has known blind spots (e.g. "we use
Docker Swarm, not Kubernetes" contains a cue word and passes). An LLM judge
is the real fix; until then check_scorers.py is the contract.
"""

from __future__ import annotations

import re
from typing import List, Tuple

SCORER_VERSION = 2

NON_ANSWER_PATTERNS = [
    r"\b(?:i|we)\s+(?:do\s*n[o']t|don't|cannot|can't|am\s+unable\s+to)\s+(?:know|find|determine|say|tell|recall|answer)",
    r"\bnot\s+sure\b",
    r"\bno\s+(?:relevant\s+)?(?:context|information|info|memory|memories|record|records|data|details)\b",
    r"\b(?:insufficient|not\s+enough)\s+(?:context|information|info)\b",
    r"\bunable\s+to\s+(?:find|determine|answer|locate)\b",
    r"\b(?:does|do)\s*n[o']t\s+(?:mention|specify|say|contain|include)\b",
    r"\bnot\s+(?:mentioned|specified)\b",
]

SUPERSESSION_CUES = [
    r"\bdeprecat\w*", r"\bretir\w*", r"\breplac\w*", r"\bmigrat\w*",
    r"\bno\s+longer\b", r"\bformer(?:ly)?\b", r"\bprevious(?:ly)?\b",
    r"\bprior\b", r"\blegacy\b", r"\bsupersed\w*", r"\bswitch\w*",
    r"\bmoved\b", r"\bstopped\b", r"\binstead\s+of\b", r"\brather\s+than\b",
    r"\bnot\b", r"\bdo\s*n[o']t\b", r"\bforbidden\b", r"\babandon\w*",
    r"\bdropped\b", r"\bphased\s+out\b", r"\bold\b", r"\bearlier\b",
    r"\bused\s+to\b", r"\btransition\w*", r"\bupgrad\w*",
]

NEGATIVE_PATTERNS = [
    r"^\W*no\b", r"\bnot\b", r"\bcannot\b", r"\bcan't\b", r"\bmust\s+not\b",
    r"\bshould\s*n[o']t\b", r"\bforbidden\b", r"\bprohibited\b",
    r"\bnot\s+(?:acceptable|allowed|permitted)\b", r"\breject\w*",
    r"\bdeprecat\w*", r"\bdisallow\w*",
]

_YES_OPENING = r"^\W*yes\b"


def _any(patterns: List[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def is_non_answer(answer: str) -> bool:
    return not answer.strip() or _any(NON_ANSWER_PATTERNS, answer)


def judge(answer: str, kind: str, accept: List[str], stale: List[str]) -> Tuple[bool, str]:
    """Return (passed, reason). `reason` is recorded per probe so a flipped
    result can be read without re-running."""
    if is_non_answer(answer):
        return False, "non_answer"
    if kind == "value":
        if not _any(accept, answer):
            return False, "missing_current_value"
        if stale and _any(stale, answer) and not _any(SUPERSESSION_CUES, answer):
            return False, "stale_value_not_marked_superseded"
        return True, "ok"
    if kind == "negative":
        if re.search(_YES_OPENING, answer, re.IGNORECASE):
            return False, "affirmative"
        if not _any(NEGATIVE_PATTERNS, answer):
            return False, "not_negative"
        if not _any(accept, answer):
            return False, "negative_without_grounding"
        return True, "ok"
    raise ValueError(f"unknown probe kind {kind!r}")


def contains_current_value(answer: str, accept: List[str]) -> bool:
    """Lenient 'exact match': the answer names an accepted form of the gold
    value (scorer v1 required the literal gold string, so "Kubernetes." missed
    "Kubernetes k8s")."""
    return _any(accept, answer)


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1. Understates correctness for full-sentence LLM answers
    against short gold strings; kept for continuity, not as a headline."""
    pred_tokens = re.findall(r"\w+", prediction.lower())
    truth_tokens = re.findall(r"\w+", ground_truth.lower())
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    overlap = sum(min(pred_tokens.count(w), truth_tokens.count(w)) for w in common)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    return 2 * (precision * recall) / (precision + recall)

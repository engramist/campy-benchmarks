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

Scorer v3 adds "prohibit", "forbid" (not just "forbidden"), "disallow" and
"banned" as supersession cues and negations. v2 failed correct answers that
retired the old values with "prohibits the use of X or Y" (LoCoMo p_6/p_8,
2026-09-30); check_scorers.py pins them verbatim.

Scorer v4 adds "update" and "change" (any form): v3 failed a correct
MemBench answer that said "Casey updates their preference from AWS ECS to
Google Cloud Run" (p_casey_5, 2026-10-01).

This is still a lexical judge and it has known blind spots (e.g. "we use
Docker Swarm, not Kubernetes" contains a cue word and passes). An LLM judge
is the real fix; until then check_scorers.py is the contract.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

SCORER_VERSION = 4

NON_ANSWER_PATTERNS = [
    r"\b(?:i|we)\s+(?:do\s*n[o']t|don't|cannot|can't|am\s+unable\s+to)\s+(?:know|find|determine|say|tell|recall|answer)",
    # "I don't have that information" -- the abstention Campy's cite-or-abstain prompt
    # (hippocampy B475) asks for; without it a correct adversarial abstention scored WRONG
    r"\b(?:i|we)\s+(?:do\s*n[o']t|don't)\s+have\s+(?:that|this|the|any|enough)\s+(?:information|info|details?|record)",
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
    r"\bnot\b", r"\bdo\s*n[o']t\b", r"\bforbid\w*", r"\babandon\w*",
    r"\bprohibit\w*", r"\bdisallow\w*", r"\bbanned\b",
    r"\bdropped\b", r"\bphased\s+out\b", r"\bold\b", r"\bearlier\b",
    r"\bused\s+to\b", r"\btransition\w*", r"\bupgrad\w*",
    r"\bupdat\w*", r"\bchang\w*",
]

NEGATIVE_PATTERNS = [
    r"^\W*no\b", r"\bnot\b", r"\bcannot\b", r"\bcan't\b", r"\bmust\s+not\b",
    r"\bshould\s*n[o']t\b", r"\bforbid\w*", r"\bprohibit\w*", r"\bbanned\b",
    r"\bnot\s+(?:acceptable|allowed|permitted)\b", r"\breject\w*",
    r"\bdeprecat\w*", r"\bdisallow\w*",
]

_YES_OPENING = r"^\W*yes\b"


def _any(patterns: List[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def is_non_answer(answer: str) -> bool:
    """The legacy rule: any NON_ANSWER_PATTERNS hit ANYWHERE in the answer. It
    also fails a correct answer with a trailing hedge ("May 10, 2023. The exact
    date is not specified.") and an "I don't have that information. However,
    ... June 13, 2023" self-contradiction. Kept for the adversarial-abstention
    metric, the lexical probe scorer, and `--non-answer-policy legacy`."""
    return not answer.strip() or _any(NON_ANSWER_PATTERNS, answer)


# --- B539: auto-fail only a pure abstention ---------------------------------
# An answer is auto-failed (WRONG with no judge call) under the `judge` policy
# only when it OPENS with an abstention and asserts nothing after it. Anything
# with a substantive claim goes to the judge, which sees the claim.

NON_ANSWER_POLICIES = ("judge", "legacy")
DEFAULT_NON_ANSWER_POLICY = "judge"
_policy = DEFAULT_NON_ANSWER_POLICY


def set_non_answer_policy(policy: str) -> None:
    """Set once per process (run_all.py / rescoring tools); recorded in the result meta."""
    global _policy
    if policy not in NON_ANSWER_POLICIES:
        raise ValueError(f"non-answer policy must be one of {NON_ANSWER_POLICIES}, got {policy!r}")
    _policy = policy


def get_non_answer_policy() -> str:
    return _policy


# Words that carry no claim: function words plus the vocabulary an abstention
# uses to talk about itself ("the provided context does not mention ...").
_NO_CLAIM_WORDS = frozenset("""
a an the and or but if then so of to in on at by for with from about as into than that this these those
it its is are was were be been being am do does did have has had can could would should will may might must
i me my we us our you your he she they them their there here what which who whom whose when where why how
not no nor any some such also just only still yet very more most much many other another own same
however although though while whereas unfortunately sorry apologize apologies afraid please
based according given provided provide providing available info information detail details data
context memory memories record records conversation conversations chat chats text transcript transcripts
mention mentions mentioned mentioning specify specifies specified specific specifically state states stated
say says said contain contains contained include includes included find found locate determine know known
tell recall answer answers question questions ask asked asking clarify clarification additional further
share exact exactly precisely enough sufficient insufficient relevant otherwise
help like want need let see look check try
t s m d ll ve re don doesn didn isn wasn couldn won
""".split())

# Extra abstention phrasings, used only by is_pure_abstention (NON_ANSWER_PATTERNS
# stays frozen so `legacy` reproduces old runs): "There is no mention of X",
# "I couldn't find any mention of X", "X is not explicitly stated".
_PURE_ABSTENTION_EXTRA = [
    r"\bno\s+(?:direct\s+|explicit\s+)?(?:mention|quote|statement|indication|evidence)\b",
    r"\b(?:could\s*n[o']t|can\s*n[o']t|could\s+not|cannot)\s+(?:find|see)\b",
    r"\bnot\s+(?:explicitly\s+)?(?:stated|provided|given)\b",
]

# a clause break after which a new claim may start
_CLAUSE_BREAK = re.compile(r"[.!?;\n]+|,?\s+(?:but|however|although|though|yet|whereas)\b", re.IGNORECASE)


def _claim_tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _NO_CLAIM_WORDS]


def is_pure_abstention(answer: str) -> bool:
    """True when `answer` is empty, or opens with an abstention and asserts nothing.

    The answer is cut into clauses (sentence ends, `;`, and a "but/however/
    though" break). The FIRST clause must contain a NON_ANSWER_PATTERNS match
    (anchored: an answer that opens with a claim is never a pure abstention,
    whatever it says later). In an abstention clause the words before the match
    must carry no claim; the words after it are the abstention's topic ("no
    information about Caroline's trip") and are ignored. Every later clause
    must be another abstention or contain only no-claim words (stopwords,
    hedge and boilerplate vocabulary); any other content token -- a name, a
    date, a number -- is an assertion, so the answer goes to the judge."""
    if not answer.strip():
        return True
    clauses = [c for c in _CLAUSE_BREAK.split(answer) if c and c.strip()]
    clauses = [c for c in clauses if re.search(r"[a-z0-9]", c, re.IGNORECASE)]
    if not clauses:
        return True
    first = True
    for clause in clauses:
        m = next((m for pat in NON_ANSWER_PATTERNS + _PURE_ABSTENTION_EXTRA if (m := re.search(pat, clause, re.IGNORECASE))), None)
        if first and m is None:
            return False
        first = False
        if m is not None:
            if _claim_tokens(clause[:m.start()]):
                return False
        elif _claim_tokens(clause):
            return False
    return True


def is_auto_fail(answer: str, policy: Optional[str] = None) -> bool:
    """Should this answer be WRONG without reaching the judge? `policy` defaults
    to the process-wide setting (default "judge")."""
    policy = policy or _policy
    if policy == "legacy":
        return is_non_answer(answer)
    return is_pure_abstention(answer)


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

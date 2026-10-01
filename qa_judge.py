"""
campy-benchmarks / qa_judge.py
LLM judge for the fixture QA suites (LoCoMo, MemBench).

The lexical scorer (scoring.py) needs a cue word whenever an answer mentions
a superseded value, and every new phrasing an LLM uses ("prohibits the use
of X", "updates their preference from X to Y") was a false failure until the
word list caught up (scorer v3, v4). This judge asks an LLM the actual
question -- does the answer give the current value without presenting an
old one as current? -- the way LoCoMo-10 already does (locomo10/scoring.py).

It runs alongside the lexical scorer, not instead of it: every record keeps
its lexical `passed`/`reason` and gains `llm_judge`/`llm_judge_reason`; the
suite gains `judge_accuracy` and `judge_deprecation_accuracy` (LoCoMo) or
`judge_contradiction_score` (MemBench), plus `judge_disagreements` -- the
probes where the two verdicts differ, which is where to look first. The
lexical `accuracy` stays, so results stay comparable with earlier files.

Before trusting a judge model, run `check_qa_judge.py --live`: it grades
check_scorers.py's ~600 hand-labelled answers and reports agreement.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from scoring import is_non_answer

JUDGED_SUITES = ("locomo", "membench")

# suite -> (record flag key, baselines' generic flag key, judge metric for the flagged subset)
_FLAGS = {
    "locomo": ("is_deprecation", "judge_deprecation_accuracy"),
    "membench": ("is_contradiction", "judge_contradiction_score"),
}

JUDGE_SYSTEM = ("You grade answers given by an AI assistant from its memory of earlier "
                "conversations. Reply with one word.")

JUDGE_TEMPLATE = """Question: {question}
Correct answer: {gold}
Generated answer: {answer}

The facts behind this question changed over time. The correct answer is the
latest one. Is the generated answer CORRECT or WRONG?

It is CORRECT if it gives the correct answer. Any wording is fine, and it may
add detail. It may mention earlier values, as long as it makes clear they are
no longer current (for example replaced, deprecated, forbidden, prohibited,
or changed from). For a yes/no question it must give the same yes or no as
the correct answer.

It is WRONG if it presents an earlier value as current, lists an earlier value
next to the correct one as if both apply, gives a different value, hedges
between alternatives, or says it does not know.

Reply with exactly one word: CORRECT or WRONG."""

_VERDICT = re.compile(r"\b(CORRECT|WRONG)\b")


def judge_answer(judge_llm, question: str, gold: str, answer: str) -> Dict[str, Any]:
    """One verdict. Non-answers are WRONG without a call."""
    if not answer.strip() or is_non_answer(answer):
        return {"llm_judge": False, "llm_judge_reason": "non_answer", "called": False}
    res = judge_llm.chat([
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_TEMPLATE.format(question=question, gold=gold, answer=answer)},
    ])
    # Reasoning models (qwen3, deepseek-r1) may think first; the verdict is the last one.
    found = _VERDICT.findall(re.sub(r"<think>.*?</think>", "", res["text"], flags=re.S).upper())
    if not found:
        return {"llm_judge": False, "llm_judge_reason": "judge_unparseable",
                "llm_judge_raw": res["text"][:200], "called": True}
    ok = found[-1] == "CORRECT"
    return {"llm_judge": ok, "llm_judge_reason": "judge_correct" if ok else "judge_wrong", "called": True}


def _rate(records: List[Dict[str, Any]]) -> Optional[float]:
    return round(sum(1 for d in records if d["llm_judge"]) / len(records), 4) if records else None


def judge_suite(suite: str, res: Dict[str, Any], judge_llm) -> int:
    """Judge every record of a LoCoMo/MemBench result in place and add the
    judge metrics to `res`. Returns the number of judge calls made."""
    flag_key, metric = _FLAGS[suite]
    details = res.get("details") or []
    calls = 0
    for d in details:
        if d.get("llm_judge") is not None:
            continue
        v = judge_answer(judge_llm, d["question"], d["expected"], d.get("answer") or "")
        calls += v.pop("called")
        d.update(v)
    flagged = [d for d in details if d.get(flag_key, d.get("flagged"))]
    res["judge_accuracy"] = _rate(details)
    res[metric] = _rate(flagged)
    res["judge_disagreements"] = [
        {"id": d["id"], "lexical": d["passed"], "llm_judge": d["llm_judge"], "lexical_reason": d.get("reason")}
        for d in details if d["llm_judge"] != d["passed"]
    ]
    return calls


def judge_results(suites: Dict[str, Any], judge_llm) -> int:
    """Judge every valid LoCoMo/MemBench result in a suites dict."""
    calls = 0
    for suite in JUDGED_SUITES:
        res = suites.get(suite)
        if res and res.get("valid", True) and res.get("details"):
            calls += judge_suite(suite, res, judge_llm)
    return calls

#!/usr/bin/env python3
"""
campy-benchmarks / check_rejudge.py
Offline checks (fake judge, no model) for judge robustness: the persona rule
(I/you swaps), vote consensus with a tiebreak, and rejudge.py's flip counts.
"""

from __future__ import annotations

import sys
from typing import List

import rejudge
from dmr.scoring import judge_details as dmr_judge_details
from locomo10.scoring import PERSONA_RULE, build_judge_prompt, judge_one

failures: List[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


class Fake:
    """Replies from a script, in order (last entry repeats); records prompts."""

    def __init__(self, *replies: str):
        self.replies, self.prompts = list(replies), []

    def chat(self, messages):
        self.prompts.append(messages[-1]["content"])
        i = min(len(self.prompts) - 1, len(self.replies) - 1)
        return {"text": self.replies[i]}

    def describe(self):
        return {"model": "fake"}


class ByPrompt:
    """Judges by prompt content: CORRECT iff `needle` is in the prompt."""

    def __init__(self, needle: str):
        self.needle, self.prompts = needle, []

    def chat(self, messages):
        self.prompts.append(messages[-1]["content"])
        return {"text": "CORRECT" if self.needle in messages[-1]["content"] else "WRONG"}

    def describe(self):
        return {"model": "fake-by-prompt"}


def prompts() -> None:
    q, g, a = "Where did I use to work?", "I used to work at Acme.", "You used to work at Acme."
    plain = build_judge_prompt(q, g, a)
    dmr = build_judge_prompt(q, g, a, persona=True)
    check(PERSONA_RULE not in plain, "default (LoCoMo-10) prompt has no persona rule")
    check(PERSONA_RULE in dmr and "not an error" in dmr, "persona prompt carries the I/you rule")
    check(dmr.rstrip().endswith("Reply with exactly one word: CORRECT or WRONG."),
          "persona rule sits before the reply instruction")
    for v in range(3):
        p = build_judge_prompt(q, g, a, persona=True, variant=v)
        check(all(x in p for x in (f"Question: {q}", f"Gold answer: {g}", f"Generated answer: {a}")),
              f"variant {v} keeps all three labelled fields")
    check(len({build_judge_prompt(q, g, a, variant=v) for v in range(3)}) == 3,
          "the three vote prompts are distinct requests")
    check(build_judge_prompt(q, g, a, variant=0).startswith(f"Question: {q}\nGold answer: {g}\nGenerated answer: {a}\n\n"),
          "variant 0 is the original field order")

    # DMR judging sends the rule; a judge that follows it passes the flipped answer.
    j = ByPrompt("not an error")
    det = [{"id": "d1", "question": q, "expected": g, "answer": a, "judge": None}]
    dmr_judge_details(det, j)
    check(det[0]["judge"] is True and PERSONA_RULE in j.prompts[0], f"DMR pronoun flip passes: {det[0]}")


def voting() -> None:
    f = Fake("CORRECT")
    v = judge_one(f, "q", "g", "a")
    check(len(f.prompts) == 1 and "judge_votes" not in v, "votes=1 is one call and unchanged")
    f = Fake("CORRECT", "CORRECT")
    v = judge_one(f, "q", "g", "a", votes=3)
    check(len(f.prompts) == 2 and v["judge"] is True and v["judge_votes"] == [True, True],
          f"agreeing pair needs no third call: {v}, {len(f.prompts)} calls")
    f = Fake("CORRECT", "WRONG", "WRONG")
    v = judge_one(f, "q", "g", "a", votes=3)
    check(len(f.prompts) == 3 and v["judge"] is False and v.get("judge_tiebreak") and v["judge_votes"] == [True, False, False],
          f"1st/2nd disagree, 3rd (WRONG) decides: {v}")
    check(len(set(f.prompts)) == 3, "the three calls are three different requests")
    f = Fake("WRONG", "CORRECT", "CORRECT")
    v = judge_one(f, "q", "g", "a", votes=3)
    check(v["judge"] is True and v["judge_tiebreak"], f"3rd (CORRECT) decides the other way: {v}")
    try:
        judge_one(Fake("CORRECT"), "q", "g", "a", votes=2)
        check(False, "votes=2 is rejected")
    except ValueError:
        pass


def rescoring() -> None:
    def rec(i, judge, ans="x", **kw):
        return {"id": i, "question": f"q{i}", "raw_question": f"q{i}", "expected": "gold", "answer": ans,
                "judge": judge, "f1": 0.5, **kw}

    data = {"suites": {
        "dmr": {"valid": True, "details": [
            rec("a", True, "ok-ans"), rec("b", False, "ok-ans"), rec("c", True, "bad-ans"),
            rec("d", False, "bad-ans"), rec("e", False, "I don't know."),
            rec("err", False, "", error="daemon"),
        ]},
        "locomo10": {"valid": True, "details": [
            rec("L1", True, "ok-ans", category=1), rec("L2", False, "bad-ans", category=2),
            {"id": "L3", "question": "q", "raw_question": "q", "expected": "g", "answer": "n/a", "category": 5,
             "abstained": True, "abstained_strict": False, "passed": True, "f1": None, "judge": None},
        ]},
    }, "baselines": {"config": {}, "full": {"dmr": {"valid": True, "details": [rec("z", True, "bad-ans")]}}}}
    # The new judge says CORRECT for "ok-ans", WRONG for "bad-ans".
    lines: List[str] = []
    rep = rejudge.rejudge(data, ByPrompt("ok-ans"), votes=1, log=lines.append)
    t = rep["targets"]
    check(t["dmr"]["wrong_to_correct"] == ["b"] and t["dmr"]["correct_to_wrong"] == ["c"],
          f"dmr flips: {t['dmr']}")
    check(t["dmr"]["judged"] == 5 and t["dmr"]["changed"] == 2, f"error record skipped, others judged: {t['dmr']}")
    check(t["locomo10"]["wrong_to_correct"] == [] and t["locomo10"]["correct_to_wrong"] == [] and t["locomo10"]["judged"] == 2,
          f"locomo10 unchanged, cat 5 left alone: {t['locomo10']}")
    check(t["dmr/full"]["correct_to_wrong"] == ["z"], f"baseline re-judged: {t.get('dmr/full')}")
    check(data["suites"]["dmr"]["details"][1]["judge_previous"] is False, "previous verdict kept per record")
    check(data["suites"]["dmr"]["details"][5]["judge"] is False and "judge_previous" not in data["suites"]["dmr"]["details"][5],
          "errored record untouched")
    check(data["suites"]["dmr"]["judge_accuracy"] == round(2 / 6, 4), f"dmr judge_accuracy recomputed over 6 records: {data['suites']['dmr']['judge_accuracy']}")
    check(data["rejudge"]["votes"] == 1 and data["rejudge"]["judge"] == {"model": "fake-by-prompt"},
          "judge config recorded")
    check(data["suites"]["locomo10"]["details"][2]["passed"] is True, "adversarial verdict untouched")
    check(any("CORRECT->WRONG: c" in x for x in lines) and any("WRONG->CORRECT: b" in x for x in lines),
          f"flip ids printed: {lines}")

    # voting through rejudge records tiebreaks
    d2 = {"suites": {"dmr": {"valid": True, "details": [rec("t", False, "ok-ans")]}}}
    rejudge.rejudge(d2, Fake("CORRECT", "WRONG", "CORRECT"), votes=3, log=lambda *_: None)
    check(d2["rejudge"]["targets"]["dmr"]["tiebreaks"] == ["t"] and d2["suites"]["dmr"]["details"][0]["judge"] is True,
          f"rejudge with votes=3 uses the tiebreak: {d2['rejudge']['targets']}")


def judge_errors() -> None:
    """A judge call that errors (e.g. Ollama aborting a repeating generation) is retried
    once; a second error records a WRONG verdict marked judge_error instead of raising."""
    from llm_client import LLMError
    from locomo10.scoring import JUDGE_MAX_TOKENS, judge_one

    class Flaky:
        def __init__(self, errors: int):
            self.errors, self.calls, self.caps = errors, 0, []

        def chat(self, messages, max_tokens=None):
            self.calls += 1
            self.caps.append(max_tokens)
            if self.calls <= self.errors:
                raise LLMError("HTTP 500: prediction aborted, token repeat limit reached")
            return {"text": "CORRECT"}

    once = Flaky(1)
    r = judge_one(once, "q", "gold", "ans")
    check(r["judge"] is True and once.calls == 2, f"one error is retried: {r}, calls={once.calls}")
    check(once.caps == [JUDGE_MAX_TOKENS] * 2, f"judge calls are capped: {once.caps}")
    twice = Flaky(5)
    try:
        r = judge_one(twice, "q", "gold", "ans", votes=3)
    except LLMError as e:
        check(False, f"a repeated judge error must not raise: {e}")
    else:
        check(r["judge"] is False and r.get("reason") == "judge_error",
              f"a repeated judge error is recorded as judge_error: {r}")


def main() -> int:
    prompts()
    voting()
    rescoring()
    judge_errors()
    if failures:
        print(f"FAIL -- {len(failures)} judge robustness checks failed:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("OK -- persona rule, vote consensus with tiebreak, and rejudge flip-count checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

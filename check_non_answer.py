#!/usr/bin/env python3
"""
campy-benchmarks / check_non_answer.py
Offline checks (made-up answers, fake judge) for the B539 non-answer policy:
only a pure abstention is auto-failed; any answer with a claim goes to the
judge; `legacy` reproduces the old any-phrase-anywhere rule.
"""

from __future__ import annotations

import sys
from typing import List

import scoring
from locomo10.scoring import judge_details
from rescore_non_answers import affected

failures: List[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


PURE = [
    "",
    "   ",
    "I don't know.",
    "I don't have that information.",
    "I don't have that information about Caroline's trip.",
    "Based on the provided context, there is no information about when Caroline moved.",
    "The conversation does not mention this.",
    "I don't have that information. The context does not specify it.",
    "I don't have that information. Please provide more context.",
    "No relevant context was found in memory.",
    "Unfortunately, I am unable to determine this from the memory provided. Could you clarify?",
    "I'm not sure. The records do not say.",
    "I don't have that information. There is no mention of Andrew and Audrey planning to hike in the provided context.",
    "No, I don't have that information. I couldn't find any mention of James living in Connecticut.",
]
CLAIM = [
    "I don't have that information. However, it was on June 13, 2023.",
    "I don't have that information, but Caroline went to Paris.",
    "I don't have that information; the trip was in May 2023.",
    "May 10, 2023. The exact date is not specified.",
    "Melanie painted a sunrise. The exact date is not mentioned.",
    "She likes pottery, although the context does not specify how long.",
    "I don't have that information. The question asks about Caroline.",
    "Nate's favorite game is not explicitly stated, but he won a tournament in June 2022.",
]


def main() -> int:
    for a in PURE:
        check(scoring.is_pure_abstention(a), f"pure abstention not auto-failed: {a!r}")
    for a in CLAIM:
        check(not scoring.is_pure_abstention(a), f"answer with a claim auto-failed: {a!r}")
        check(not scoring.is_auto_fail(a, "judge"), f"judge policy auto-failed a claim: {a!r}")
    # legacy keeps the old any-phrase-anywhere behaviour (the hedged answers are auto-failed)
    for a in ("May 10, 2023. The exact date is not specified.",
              "I don't have that information. However, it was on June 13, 2023."):
        check(scoring.is_auto_fail(a, "legacy"), f"legacy no longer auto-fails: {a!r}")
    check(scoring.is_auto_fail("", "legacy") and scoring.is_auto_fail("", "judge"), "empty must auto-fail")
    check(not scoring.is_auto_fail("Paris.", "legacy") and not scoring.is_auto_fail("Paris.", "judge"),
          "a plain answer must reach the judge")

    class Fake:
        calls = 0

        def chat(self, messages):
            Fake.calls += 1
            return {"text": "CORRECT"}

    def rec(i, ans):
        return {"id": i, "category": 1, "question": "q", "raw_question": "q", "expected": "g", "answer": ans}

    try:
        for policy, expect_calls in (("judge", 2), ("legacy", 1)):
            scoring.set_non_answer_policy(policy)
            Fake.calls = 0
            rows = [rec("a", "I don't know."), rec("b", "May 10, 2023. The exact date is not specified."),
                    rec("c", "Paris."), rec("d", "")]
            n = judge_details(rows, Fake())
            check(n == expect_calls and Fake.calls == expect_calls, f"{policy}: {Fake.calls} judge calls, want {expect_calls}")
            check(rows[0]["judge"] is False and rows[3]["judge"] is False, f"{policy}: pure abstention/empty not WRONG")
            check(rows[1]["judge"] is (policy == "judge"), f"{policy}: hedged answer verdict wrong")
    finally:
        scoring.set_non_answer_policy("judge")
    try:
        scoring.set_non_answer_policy("bogus")
        check(False, "bogus policy accepted")
    except ValueError:
        pass

    # rescore_non_answers.affected: legacy auto-fails the new policy would judge
    res = {"details": [
        {"id": "x1", "category": 1, "judge": False, "reason": "non_answer", "answer": "May 10. Not specified."},
        {"id": "x2", "category": 1, "judge": False, "reason": "non_answer", "answer": "I don't know."},
        {"id": "x3", "category": 1, "judge": False, "reason": "judge_wrong", "answer": "Not specified, May 10."},
        {"id": "x4", "category": 5, "judge": None, "reason": "non_answer", "answer": "May 10. Not specified."},
    ]}
    check([d["id"] for d in affected(res)] == ["x1"], f"affected(): {[d['id'] for d in affected(res)]}")

    if failures:
        print(f"FAIL -- {len(failures)} non-answer policy checks failed:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("OK -- pure abstentions auto-fail; hedged and abstain+fact answers reach the judge; legacy reproducible")
    return 0


if __name__ == "__main__":
    sys.exit(main())

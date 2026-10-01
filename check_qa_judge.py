#!/usr/bin/env python3
"""
campy-benchmarks / check_qa_judge.py

Checks for the LoCoMo/MemBench LLM judge (qa_judge.py).

Offline (default, no LLM needed): plumbing with a stub LLM -- verdict
parsing (including a reasoning model's <think> block and a last-word
verdict), non-answers judged WRONG without a call, an unparseable reply
counted WRONG, and the suite metrics and lexical-vs-judge disagreements
judge_suite() adds.

Live (`--live`): grades every hand-labelled answer from check_scorers.py
(the gold answers, paraphrases, real llama answers, wrong answers) with a
real judge model and reports how often it agrees with the labels. Run this
before trusting a judge model's scores:

    python3 check_qa_judge.py --live --judge-model gemma4:26b

Uses the same [llm] config resolution as run_all.py's --judge-* flags
(CAMPY_BENCH_CONFIG, ~/.campy/config.toml, then the hippocampy repo's
campy.toml). Exits 1 when agreement is below --min-agreement (default 0.95)
and lists every disagreement.
"""

from __future__ import annotations

import argparse
import sys
from typing import List

import qa_judge
from check_scorers import labelled_cases

failures: List[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


class StubLLM:
    """Replies with a fixed text; counts calls."""

    def __init__(self, text: str):
        self.text, self.calls = text, 0

    def chat(self, messages):
        self.calls += 1
        return {"text": self.text}


def offline() -> int:
    v = qa_judge.judge_answer(StubLLM("CORRECT"), "q", "gold", "an answer")
    check(v["llm_judge"] is True and v["called"], f"CORRECT parsed: {v}")
    v = qa_judge.judge_answer(StubLLM("Wrong."), "q", "gold", "an answer")
    check(v["llm_judge"] is False and v["llm_judge_reason"] == "judge_wrong", f"WRONG parsed: {v}")
    v = qa_judge.judge_answer(StubLLM("<think>Is it CORRECT? It names the old one... </think>\nWRONG"),
                              "q", "gold", "an answer")
    check(v["llm_judge"] is False, f"<think> block ignored, final verdict used: {v}")
    v = qa_judge.judge_answer(StubLLM("Hmm, CORRECT at first glance, but WRONG overall"), "q", "g", "a")
    check(v["llm_judge"] is False, f"last verdict word wins: {v}")
    v = qa_judge.judge_answer(StubLLM("maybe?"), "q", "g", "a")
    check(v["llm_judge"] is False and v["llm_judge_reason"] == "judge_unparseable", f"unparseable: {v}")
    stub = StubLLM("CORRECT")
    v = qa_judge.judge_answer(stub, "q", "g", "No relevant context was found in memory.")
    check(v["llm_judge"] is False and stub.calls == 0, f"non-answer judged WRONG without a call: {v}")

    res = {"valid": True, "details": [
        {"id": "a", "question": "q", "expected": "g", "answer": "x", "passed": True, "is_deprecation": True},
        {"id": "b", "question": "q", "expected": "g", "answer": "x", "passed": False, "reason": "stale",
         "is_deprecation": True},
        {"id": "c", "question": "q", "expected": "g", "answer": "I don't know.", "passed": False,
         "is_deprecation": False},
    ]}
    stub = StubLLM("CORRECT")
    calls = qa_judge.judge_suite("locomo", res, stub)
    check(calls == 2 and stub.calls == 2, f"two calls (non-answer skipped): {calls}")
    check(res["judge_accuracy"] == round(2 / 3, 4), f"judge_accuracy: {res.get('judge_accuracy')}")
    check(res["judge_deprecation_accuracy"] == 1.0, f"flagged subset: {res.get('judge_deprecation_accuracy')}")
    check([d["id"] for d in res["judge_disagreements"]] == ["b"], f"disagreements: {res['judge_disagreements']}")
    again = qa_judge.judge_suite("locomo", res, stub)
    check(again == 0, "already-judged records are not re-judged")

    base = {"valid": True, "details": [
        {"id": "m", "question": "q", "expected": "g", "answer": "x", "passed": True, "flagged": True}]}
    qa_judge.judge_suite("membench", base, StubLLM("WRONG"))
    check(base["judge_contradiction_score"] == 0.0, "baselines' generic 'flagged' key is honoured")

    suites = {"locomo": {"valid": False, "details": [{"id": "z"}]}, "arc_bridge": {"valid": True}}
    check(qa_judge.judge_results(suites, StubLLM("CORRECT")) == 0, "invalid / non-QA suites are skipped")

    n_cases = sum(1 for c in labelled_cases() if c[4] is not None)
    check(n_cases > 500, f"labelled cases available for --live: {n_cases}")

    if failures:
        print(f"FAIL -- {len(failures)} judge checks failed:")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"OK -- judge plumbing checks passed ({n_cases} labelled cases available for --live)")
    return 0


def live(args) -> int:
    from run_all import resolve_baseline_llm

    llm, _, source = resolve_baseline_llm(
        args, None, None, overrides=(args.judge_provider, args.judge_model, args.judge_base_url))
    print(f"Judge: {llm.describe()['model']} ({source})")
    cases = [c for c in labelled_cases() if c[4] is not None]
    seen, todo = set(), []
    for pid, _probe, gold, question, answer, label in cases:
        key = (pid, answer)
        if key not in seen:
            seen.add(key)
            todo.append((pid, gold, question, answer, label))
    if args.limit:
        todo = todo[: args.limit]
    agree, false_pass, false_fail = 0, [], []
    for i, (pid, gold, question, answer, label) in enumerate(todo, 1):
        v = qa_judge.judge_answer(llm, question, gold, answer)
        if v["llm_judge"] == label:
            agree += 1
        else:
            (false_pass if v["llm_judge"] else false_fail).append((pid, answer, v["llm_judge_reason"]))
        if i % 50 == 0:
            print(f"  {i}/{len(todo)} judged, agreement so far {agree / i:.3f}")
    rate = agree / max(1, len(todo))
    print(f"\nAgreement with the hand labels: {agree}/{len(todo)} = {rate:.3f} ({llm.calls} judge calls)")
    print(f"  judged CORRECT but labelled wrong (lenient): {len(false_pass)}")
    for pid, a, _ in false_pass:
        print(f"    {pid}: {a!r}")
    print(f"  judged WRONG but labelled correct (strict):  {len(false_fail)}")
    for pid, a, why in false_fail:
        print(f"    {pid} [{why}]: {a!r}")
    if rate < args.min_agreement:
        print(f"FAIL -- agreement {rate:.3f} < {args.min_agreement}: don't trust this judge model's scores")
        return 1
    print("OK -- judge agrees with the hand labels")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--live", action="store_true", help="grade the labelled answers with a real judge model")
    ap.add_argument("--judge-provider", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--limit", type=int, default=0, help="--live: judge only the first N cases")
    ap.add_argument("--min-agreement", type=float, default=0.95)
    args = ap.parse_args()
    # resolve_baseline_llm reads these attributes for its default overrides
    args.baseline_provider = args.baseline_model = args.baseline_base_url = None
    return live(args) if args.live else offline()


if __name__ == "__main__":
    sys.exit(main())

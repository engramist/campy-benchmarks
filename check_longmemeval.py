#!/usr/bin/env python3
"""
campy-benchmarks / check_longmemeval.py
Self-checks for the LongMemEval suite, on a small sample written in the
official data format (no download, no daemon):

  * loader: sessions sorted by date, the 7 reporting categories interleaved,
    evidence ids from `has_answer`, abstention from the `_abs` suffix,
    --lme-types filtering;
  * judge: the official per-type prompt is chosen; a non-answer is wrong
    without a judge call except for abstention questions; "yes" = correct;
  * aggregate: accuracy, per category, task-averaged accuracy, recall;
  * end to end in mock mode: `run_all.py --smoke --suite longmemeval`;
  * a real daemon without --isolated is refused.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from longmemeval.dataset import load_questions  # noqa: E402
from longmemeval.scoring import aggregate, anscheck_prompt, evidence_recall, judge_record  # noqa: E402

failures = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


def entry(qid, qtype, question, answer, sessions, qdate="2023/06/01 (Thu) 10:00"):
    """`sessions`: [(session_id, date, [(role, text, has_answer), ...])]"""
    return {
        "question_id": qid, "question_type": qtype, "question": question, "answer": answer,
        "question_date": qdate,
        "haystack_session_ids": [s[0] for s in sessions],
        "haystack_dates": [s[1] for s in sessions],
        "haystack_sessions": [[{"role": r, "content": c, **({"has_answer": True} if h else {})}
                               for r, c, h in s[2]] for s in sessions],
        "answer_session_ids": [s[0] for s in sessions if any(h for _, _, h in s[2])],
    }


SAMPLE = [
    entry("q_ssu", "single-session-user", "What breed is my dog?", "Golden retriever", [
        ("s2", "2023/05/20 (Sat) 09:00", [("user", "Just got a golden retriever puppy!", True),
                                          ("assistant", "Congratulations!", False)]),
        ("s1", "2023/05/02 (Tue) 08:00", [("user", "Thinking about getting a dog.", False)]),
    ]),
    entry("q_ssa", "single-session-assistant", "Which pasta shape did you suggest for pesto?", "Trofie", [
        ("s3", "2023/05/10 (Wed) 12:00", [("user", "Pasta for pesto?", False),
                                          ("assistant", "Try trofie, the classic Ligurian shape.", True)]),
    ]),
    entry("q_pref", "single-session-preference", "Any podcast suggestions for my commute?",
          "The user would prefer history podcasts", [
        ("s4", "2023/05/11 (Thu) 07:30", [("user", "I love history podcasts on my commute.", True)]),
    ]),
    entry("q_tr", "temporal-reasoning", "How many days ago did I start running?", "10 days", [
        ("s5", "2023/05/22 (Mon) 18:00", [("user", "Started running today!", True)]),
    ]),
    entry("q_ku", "knowledge-update", "Where do I live now?", "Denver", [
        ("s6", "2023/04/01 (Sat) 10:00", [("user", "I live in Austin.", True)]),
        ("s7", "2023/05/25 (Thu) 10:00", [("user", "Moved to Denver last week.", True)]),
    ]),
    entry("q_ms", "multi-session", "How many books did I finish in May?", "3", [
        ("s8", "2023/05/05 (Fri) 21:00", [("user", "Finished two novels this week.", True)]),
        ("s9", "2023/05/28 (Sun) 21:00", [("user", "Finished another book today.", True)]),
    ]),
    entry("q_ku_abs", "knowledge-update", "What is my cat's name?",
          "You never mentioned a cat; you mentioned a dog", [
        ("s10", "2023/05/20 (Sat) 09:00", [("user", "My dog is called Biscuit.", False)]),
    ]),
    entry("q_ssu2", "single-session-user", "What is my sister's job?", "Nurse", [
        ("s11", "2023/05/15 (Mon) 20:00", [("user", "My sister works as a nurse.", True)]),
    ]),
]


class FakeJudge:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def chat(self, messages):
        self.prompts.append(messages[0]["content"])
        return {"text": self.reply}


with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "lme_sample.json"
    path.write_text(json.dumps(SAMPLE))

    # --- loader --------------------------------------------------------------
    qs = load_questions(path)
    check(len(qs) == 8, f"loaded {len(qs)} questions")
    ssu = next(q for q in qs if q.question_id == "q_ssu")
    check([s.session_id for s in ssu.sessions] == ["s1", "s2"], "sessions sorted by date (oracle is unsorted)")
    check(ssu.evidence_ids() == ["s2:0"], f"evidence ids from has_answer: {ssu.evidence_ids()}")
    abs_q = next(q for q in qs if q.question_id == "q_ku_abs")
    check(abs_q.abstention and abs_q.category == "abstention", "_abs is an abstention question")
    first7 = {q.category for q in qs[:7]}
    check(len(first7) == 7, f"the first 7 cover all 7 categories (interleaved): {first7}")
    check([q.question_id for q in load_questions(path, types=["knowledge-update"])] == ["q_ku"],
          "--lme-types filters by type; the abstention one has its own category")
    check([q.question_id for q in load_questions(path, types=["abstention"])] == ["q_ku_abs"],
          "--lme-types abstention selects abstention questions")
    check(len(load_questions(path, max_questions=3)) == 3, "max_questions")

    # --- judge ---------------------------------------------------------------
    p = anscheck_prompt("temporal-reasoning", "Q", "A", "R")
    check("off-by-one" in p, "temporal-reasoning gets its own prompt")
    check("updated answer" in anscheck_prompt("knowledge-update", "Q", "A", "R"), "knowledge-update prompt")
    check("Rubric:" in anscheck_prompt("single-session-preference", "Q", "A", "R"), "preference prompt")
    check("unanswerable" in anscheck_prompt("knowledge-update", "Q", "A", "R", abstention=True),
          "abstention prompt wins over the type")

    rec = {"question_type": "single-session-user", "question": "Q", "expected": "Nurse",
           "answer": "No relevant context was found in memory.", "abstention": False}
    j = FakeJudge("yes")
    v = judge_record(j, rec)
    check(v["judge"] is False and not v["called"] and not j.prompts, "a non-answer is wrong without a call")
    v = judge_record(j, {**rec, "abstention": True})
    check(v["judge"] is True and v["called"], "an abstention question's non-answer goes to the judge")
    check(judge_record(FakeJudge("No."), {**rec, "answer": "A doctor."})["judge"] is False, "'no' is wrong")
    check(judge_record(FakeJudge("<think>yes?</think> no"), {**rec, "answer": "x"})["judge"] is False,
          "reasoning blocks are ignored")

    # --- aggregate -----------------------------------------------------------
    rows = [
        {"category": "multi-session", "judge": True, "evidence_recall": 1.0},
        {"category": "multi-session", "judge": False, "evidence_recall": 0.5},
        {"category": "abstention", "judge": True, "evidence_recall": None},
    ]
    agg = aggregate(rows)
    check(agg["accuracy"] == 0.6667, f"accuracy: {agg['accuracy']}")
    check(agg["by_category"]["multi-session"]["accuracy"] == 0.5, "per-category accuracy")
    check(agg["task_averaged_accuracy"] == 0.75, f"task-averaged: {agg['task_averaged_accuracy']}")
    check(agg["evidence_recall"] == 0.75, f"recall ignores None: {agg['evidence_recall']}")
    check(aggregate([{"category": "x", "judge": None, "evidence_recall": None}])["accuracy"] is None,
          "unjudged rows give no accuracy")

    # --- evidence recall -------------------------------------------------------
    long_turn = "[2023/05/10] " + " ".join(f"word{i}" for i in range(1500))  # > 4000 chars
    stored = "[assistant, 2026-10-06 02:44] " + long_turn[:3990]
    tt = {"s:0": long_turn, "s:1": "[2023/05/10] I live in Austin.", "s:2": "unrelated words entirely here"}
    check(evidence_recall(["s:0"], tt, [stored]) == 1.0, "a turn Campy truncated at ingest still counts")
    check(evidence_recall(["s:1", "s:2"], tt, ["[user, x] [2023/05/10] I live in Austin."]) == 0.5,
          "recall is per evidence turn")
    check(evidence_recall(["nope"], tt, []) is None, "unknown evidence ids give None")

    # --- end to end (mock mode) ------------------------------------------------
    env = {k: v for k, v in os.environ.items() if k != "CAMPY_MCP_CMD"}
    env["LONGMEMEVAL_PATH"] = str(path)
    out = Path(tmp) / "r.json"
    proc = subprocess.run([sys.executable, str(HERE / "run_all.py"), "--smoke", "--suite", "longmemeval",
                           "--lme-questions", "8", "--out", str(out)],
                          cwd=HERE, env=env, capture_output=True, text=True, timeout=600)
    check(proc.returncode == 0, f"mock run exited {proc.returncode}: {proc.stderr[-800:]}")
    if out.exists():
        lme = json.loads(out.read_text())["suites"].get("longmemeval", {})
        check(lme.get("valid") is True, f"suite valid: {str(lme)[:300]}")
        check(len(lme.get("details") or []) == 8, "8 questions answered")
        check(all("evidence_recall" in d and "answer" in d for d in lme.get("details") or []),
              "each record has recall and an answer")

    bad = subprocess.run([sys.executable, str(HERE / "run_all.py"), "--suite", "longmemeval"],
                         cwd=HERE, env={**env, "CAMPY_MCP_CMD": "python -m nothing"},
                         capture_output=True, text=True, timeout=60)
    check(bad.returncode == 2 and "isolated" in bad.stderr, "a real daemon needs --isolated")

if failures:
    print(f"FAIL -- {len(failures)} LongMemEval checks failed:")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)
print("OK -- LongMemEval loader, judge, aggregate and mock end-to-end checks passed")

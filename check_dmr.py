#!/usr/bin/env python3
"""
campy-benchmarks / check_dmr.py
Self-checks for the DMR suite on records written in the released format
(no download, no daemon):

  * loader: alternating speakers from Speaker 1, sessions in order, derived
    evidence and the answering speaker, a paraphrased answer has none;
  * runner: only the 4 earlier sessions are written (never session 5, the
    personas or the summaries, which hold the answer verbatim), each turn
    with its session and speaker; the question is addressed to the answerer;
  * scoring: a non-answer is wrong without a judge call; aggregate;
  * end to end in mock mode, and a real daemon without --isolated is refused.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dmr.dataset import load_questions  # noqa: E402
from dmr.runner import ask_text, run_dmr  # noqa: E402
from dmr.scoring import aggregate, judge_details  # noqa: E402

failures = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


SECRET = "PERSONA-ONLY-SENTENCE"


def record(i, answer, question, sessions):
    return {
        "personas": [[f"{SECRET} {answer}"], ["x"]],
        "init_personas": [[SECRET], []], "personas_update1": [[SECRET]], "personas_update2": [[SECRET]],
        "summary_speaker_1": [[f"{SECRET} {answer}"]], "summary_speaker_2": [[SECRET]],
        "dialog": [{"text": f"SESSION-FIVE {answer}", "id": "Speaker 1", "convai2_id": f"valid_{i}"}],
        "metadata": {"initial_data_id": f"valid_{i}", "session_id": 4},
        "previous_dialogs": [{"personas": [[SECRET]], "dialog": [{"text": t} for t in s],
                              "time_num": 5, "time_unit": "days", "time_back": f"{5 - k} weeks ago"}
                             for k, s in enumerate(sessions)],
        "self_instruct": {"B": question, "A": answer},
    }


RECORDS = [
    record(0, "Taylor Swift!", "Hey, remember music? What was the artist you could get into?", [
        ["Hi! How are you?", "Good. I could get into Taylor Swift lately."],
        ["I work at a bank.", "Nice."],
        ["Any trips?", "Not really."],
        ["Bye!", "See you."],
    ]),
    record(1, "Burger King", "What was the fast food place you said you're working at?", [
        ["I just started at Burger King.", "Cool!"],
        ["How is work?", "Fine."],
        ["Busy?", "Yes."],
        ["Ok", "Ok"],
    ]),
    record(2, "a golden retriever", "What dog did you want?", [
        ["I want a big yellow dog someday.", "Fun."],
        ["a", "b"], ["c", "d"], ["e", "f"],
    ]),
]


class Recorder:
    """A fake client: records writes, returns a bundle holding everything written."""

    def __init__(self):
        self.writes, self.asks = [], []
        self.stats = {"calls": 0, "failures": 0}

    def notify_turn(self, role, content, session_id):
        self.writes.append((role, content, session_id))

    def run_sweep(self):
        pass

    def compile_context(self, q):
        return {"bundle": {"sections": [{"type": "conversation",
                                         "content": [{"text": c} for _, c, _ in self.writes]}]}}

    def ask(self, q, session_id=None):
        self.asks.append(q)
        return "No relevant context was found in memory."


with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "msc_self_instruct.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n")

    qs = load_questions(path)
    check(len(qs) == 3, f"loaded {len(qs)}")
    q0 = qs[0]
    check([s.index for s in q0.sessions] == [1, 2, 3, 4], "4 sessions in order")
    check([t.speaker for t in q0.sessions[0].turns] == ["Speaker 1", "Speaker 2"], "speakers alternate from Speaker 1")
    check(q0.evidence_ids() == ["1:1"] and q0.answerer == "Speaker 2", f"evidence/answerer: {q0.evidence_ids()} {q0.answerer}")
    check(ask_text(q0).startswith("Speaker 1 asks Speaker 2: "), ask_text(q0))
    check(qs[1].answerer == "Speaker 1" and qs[1].evidence_ids() == ["1:0"], "answer said by Speaker 1")
    check(qs[2].answerer is None and qs[2].evidence_ids() == [] and ask_text(qs[2]) == qs[2].question,
          "a paraphrased answer has no derived evidence and the question stays verbatim")
    check(len(load_questions(path, max_questions=2)) == 2, "max_questions")

    stores = []

    @contextmanager
    def new_store():
        c = Recorder()
        stores.append(c)
        yield c

    os.environ["DMR_PATH"] = str(path)
    res = run_dmr(new_store, {"max_questions": 3}, log=lambda *a: None)
    check(len(stores) == 3, "a fresh store per question")
    written = [c for s in stores for _, c, _ in s.writes]
    check(not any(SECRET in w or "SESSION-FIVE" in w for w in written),
          "personas, summaries and session 5 are never written")
    check(all(r == "user" for s in stores for r, _, _ in s.writes), "every turn as role user")
    check(stores[0].writes[1][1] == "[Session 1, 5 weeks ago] Speaker 2: Good. I could get into Taylor Swift lately.",
          f"turn format: {stores[0].writes[1][1]}")
    check(sum(len(s.writes) for s in stores) == 24, "every earlier turn is written")
    d = res["details"]
    check(d[0]["evidence_recall"] == 1.0 and d[2]["evidence_recall"] is None, "recall over derived evidence only")
    check(res["evidence_labelled"] == 2, "labelled count")

    class Judge:
        calls = 0

        def chat(self, m):
            Judge.calls += 1
            return {"text": "CORRECT"}

    d[1]["answer"] = "Burger King."
    n = judge_details(d, Judge())
    check(n == 1 and Judge.calls == 1, "only real answers reach the judge")
    check([x["judge"] for x in d] == [False, True, False], "non-answers are wrong")
    agg = aggregate(d)
    check(agg["judge_accuracy"] == 0.3333, f"accuracy {agg['judge_accuracy']}")

    env = {k: v for k, v in os.environ.items() if k != "CAMPY_MCP_CMD"}
    out = Path(tmp) / "r.json"
    proc = subprocess.run([sys.executable, str(HERE / "run_all.py"), "--smoke", "--suite", "dmr", "--out", str(out)],
                          cwd=HERE, env=env, capture_output=True, text=True, timeout=600)
    check(proc.returncode == 0, f"mock run exited {proc.returncode}: {proc.stderr[-800:]}")
    if out.exists():
        dm = json.loads(out.read_text())["suites"].get("dmr", {})
        check(dm.get("valid") is True and len(dm.get("details") or []) == 3, f"mock suite: {str(dm)[:300]}")
    bad = subprocess.run([sys.executable, str(HERE / "run_all.py"), "--suite", "dmr"],
                         cwd=HERE, env={**env, "CAMPY_MCP_CMD": "python -m nothing"},
                         capture_output=True, text=True, timeout=60)
    check(bad.returncode == 2 and "isolated" in bad.stderr, "a real daemon needs --isolated")

if failures:
    print(f"FAIL -- {len(failures)} DMR checks failed:")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)
print("OK -- DMR loader, ingestion, scoring and mock end-to-end checks passed")

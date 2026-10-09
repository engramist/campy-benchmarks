#!/usr/bin/env python3
"""
campy-benchmarks / check_turn_metadata.py
Self-checks for --turn-metadata (hippocampy B472 Phase 1), no daemon:

  * the date parsers: LoCoMo-10 session dates, LongMemEval session dates,
    DMR's "time back" distances;
  * notify_turn sends speaker/occurred_at only when given;
  * in "fields" mode the DMR and LongMemEval runners write the turn's text
    alone, with the speaker and time as fields, and score evidence recall
    against what was stored; "text" mode is unchanged;
  * report.py keeps a fields run on its own headline row and tags its subset.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import report  # noqa: E402
from dmr.runner import occurred_at as dmr_time, run_dmr  # noqa: E402
from locomo10.dataset import L10Turn  # noqa: E402
from longmemeval.runner import occurred_at as lme_time, run_longmemeval  # noqa: E402
from mcp_client import CampyMCPClient  # noqa: E402

failures = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


# --- parsers ---------------------------------------------------------------------
t = L10Turn("D1:3", 1, "1:56 pm on 8 May, 2023", "Caroline", "I went to a support group.", "a rainbow flag")
check(t.occurred_at() == "2023-05-08T13:56:00", f"LoCoMo-10 date: {t.occurred_at()}")
check(t.body() == "I went to a support group. [shares an image: a rainbow flag]", t.body())
check(t.content() == "[1:56 pm on 8 May, 2023] Caroline: " + t.body(), "text mode content unchanged")
check(L10Turn("x", 1, "sometime", "A", "b").occurred_at() is None, "unparseable LoCoMo-10 date -> None")
check(lme_time("2023/05/30 (Tue) 04:10") == "2023-05-30T04:10:00", lme_time("2023/05/30 (Tue) 04:10") or "None")
check(lme_time("May 30") is None, "unparseable LongMemEval date -> None")


class S:
    def __init__(self, tb):
        self.time_back = tb


check(dmr_time(S("7 days 8 hours ago")) == "2022-12-25T04:00:00", f"DMR: {dmr_time(S('7 days 8 hours ago'))}")
check(dmr_time(S("2 weeks ago")) == "2022-12-18T12:00:00", f"DMR weeks: {dmr_time(S('2 weeks ago'))}")
check(dmr_time(S("")) is None and dmr_time(S("a while ago")) is None, "unparseable DMR time back -> None")
check(dmr_time(S("5 weeks ago")) < dmr_time(S("1 weeks ago")), "older sessions get earlier times")


# --- the client ------------------------------------------------------------------
class Sent(CampyMCPClient):
    def __init__(self):
        self.args = []

    def call_tool(self, name, args):
        self.args.append(args)
        return {}


c = Sent()
c.notify_turn("user", "hi", "s1")
c.notify_turn("user", "hi", "s1", speaker="Caroline", occurred_at="2023-05-08T13:56:00")
check(set(c.args[0]) == {"role", "content", "session_id"}, f"no extra keys by default: {c.args[0]}")
check(c.args[1].get("speaker") == "Caroline" and c.args[1].get("occurred_at") == "2023-05-08T13:56:00",
      f"fields sent when given: {c.args[1]}")


# --- the runners -------------------------------------------------------------------
class Recorder:
    """Records writes; its bundle holds everything written (so recall is 1.0
    when the runner scores against what it stored)."""

    def __init__(self):
        self.writes = []
        self.stats = {"calls": 0, "failures": 0}

    def notify_turn(self, role, content, session_id, speaker=None, occurred_at=None):
        self.writes.append({"role": role, "content": content, "speaker": speaker, "occurred_at": occurred_at})

    def run_sweep(self):
        pass

    def compile_context(self, q):
        return {"bundle": {"sections": [{"type": "conversation",
                                         "content": [{"text": "[x, 2023-01-01 00:00] " + w["content"]}
                                                     for w in self.writes]}]}}

    def ask(self, q, session_id=None):
        return "I don't know."


def store_of(rec):
    @contextmanager
    def new_store():
        rec.writes.clear()
        yield rec
    return new_store


def dmr_record(i, answer, question, sessions):
    return {
        "personas": [[answer], ["x"]], "init_personas": [[], []],
        "dialog": [{"text": "now", "id": "Speaker 1", "convai2_id": f"valid_{i}"}],
        "metadata": {"initial_data_id": f"valid_{i}", "session_id": 4},
        "previous_dialogs": [{"personas": [[]], "dialog": [{"text": x} for x in s],
                              "time_num": 5, "time_unit": "days", "time_back": f"{5 - k} weeks ago"}
                             for k, s in enumerate(sessions)],
        "self_instruct": {"B": question, "A": answer},
    }


LME = [{
    "question_id": "q1", "question_type": "single-session-user", "question": "What breed is my dog?",
    "answer": "Golden retriever", "question_date": "2023/06/01 (Thu) 10:00",
    "haystack_session_ids": ["s1"], "haystack_dates": ["2023/05/20 (Sat) 09:00"],
    "haystack_sessions": [[{"role": "user", "content": "Just got a golden retriever puppy!", "has_answer": True},
                           {"role": "assistant", "content": "Congratulations!"}]],
    "answer_session_ids": ["s1"],
}]

with tempfile.TemporaryDirectory() as tmp:
    dmr_path = Path(tmp) / "dmr.jsonl"
    dmr_path.write_text(json.dumps(dmr_record(0, "Burger King", "Where do you work?", [
        ["I just started at Burger King.", "Cool!"], ["a", "b"], ["c", "d"], ["e", "f"]])) + "\n")
    lme_path = Path(tmp) / "lme.json"
    lme_path.write_text(json.dumps(LME))
    os.environ["DMR_PATH"], os.environ["LONGMEMEVAL_PATH"] = str(dmr_path), str(lme_path)

    for mode in ("text", "fields"):
        rec = Recorder()
        out = run_dmr(store_of(rec), {}, log=lambda *a: None, turn_metadata=mode)
        w = rec.writes[0]
        d = out["details"][0]
        if mode == "fields":
            check(w["content"] == "I just started at Burger King." and w["speaker"] == "Speaker 1"
                  and w["occurred_at"] == "2022-11-27T12:00:00", f"DMR fields write: {w}")
        else:
            check(w["content"].startswith("[Session 1, 5 weeks ago] Speaker 1: ") and w["speaker"] is None
                  and w["occurred_at"] is None, f"DMR text write unchanged: {w}")
        check(d["evidence_recall"] == 1.0, f"DMR {mode}: recall scored against what was stored ({d})")
        check(out["dataset"]["turn_metadata"] == mode, "DMR records the mode")

        rec = Recorder()
        out = run_longmemeval(store_of(rec), {"variant": "oracle"}, log=lambda *a: None, turn_metadata=mode)
        w = rec.writes[0]
        if mode == "fields":
            check(w["content"] == "Just got a golden retriever puppy!" and w["speaker"] is None
                  and w["occurred_at"] == "2023-05-20T09:00:00", f"LongMemEval fields write: {w}")
        else:
            check(w["content"] == "[2023/05/20 (Sat) 09:00] Just got a golden retriever puppy!",
                  f"LongMemEval text write unchanged: {w}")
        check(out["details"][0]["evidence_recall"] == 1.0, f"LongMemEval {mode}: recall against what was stored")
        check(out["dataset"]["turn_metadata"] == mode, "LongMemEval records the mode")

# LoCoMo-10, when the pinned dataset is on disk (it is downloaded on first use)
from locomo10.dataset import DEFAULT_PATH  # noqa: E402
from locomo10.runner import run_locomo10  # noqa: E402

if Path(os.environ.get("LOCOMO10_PATH") or DEFAULT_PATH).exists():
    rec = Recorder()
    out = run_locomo10(rec, opts={"conversations": 1, "max_questions": 2}, turn_metadata="fields")
    w = rec.writes[0]
    check(w["speaker"] and w["occurred_at"] and not w["content"].startswith("["), f"LoCoMo-10 fields write: {w}")
    check(out["dataset"]["turn_metadata"] == "fields", "LoCoMo-10 records the mode")
    check(all(d["evidence_recall"] in (1.0, None) for d in out["details"]),
          f"LoCoMo-10 recall against what was stored: {[d['evidence_recall'] for d in out['details']]}")
else:
    print("(LoCoMo-10 dataset not on disk: its runner check skipped)")

# --- the report ----------------------------------------------------------------------
s_text = {"dataset": {"variant": "oracle", "turn_metadata": "text"}, "questions": 35}
s_fields = {"dataset": {"variant": "oracle", "turn_metadata": "fields"}, "questions": 35}
s_old = {"dataset": {"variant": "oracle"}, "questions": 35}  # files written before the option existed
check(report.subset("longmemeval", s_fields) == "oracle, 35 q, turn fields", report.subset("longmemeval", s_fields))
check(report.subset("longmemeval", s_text) == report.subset("longmemeval", s_old) == "oracle, 35 q",
      "text runs and older files keep their subset label")
check(report.variant("longmemeval", s_fields) != report.variant("longmemeval", s_text), "fields run: own row")
check(report.variant("longmemeval", s_text) == report.variant("longmemeval", s_old), "older files = text runs")

if failures:
    print(f"FAIL -- {len(failures)} turn-metadata checks failed:")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)
print("OK -- --turn-metadata parsers, client, runners (text and fields) and report checks passed")

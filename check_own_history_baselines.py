#!/usr/bin/env python3
"""
campy-benchmarks / check_own_history_baselines.py
Self-checks for the LongMemEval and DMR baselines (baselines.py), on small
samples in the official data formats, with a fake LLM (no download, no
daemon, no Ollama):

  * full_context and naive_rag see only the question's own history (every
    LongMemEval / DMR question has its own), no_memory sees none;
  * full_context's evidence recall is 1.0, no_memory's is None;
  * records have the shape the suite's own judge and aggregate read, and
    run_all's finalize step judges each baseline with the same judge;
  * a prompt larger than its num_ctx is flagged `context_overflow`;
  * end to end: `run_all.py --suite longmemeval --baselines-only` writes
    judged baseline results.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import baselines  # noqa: E402
import run_all  # noqa: E402

failures = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


def lme_entry(qid, qtype, question, answer, sessions):
    return {
        "question_id": qid, "question_type": qtype, "question": question, "answer": answer,
        "question_date": "2023/06/01 (Thu) 10:00",
        "haystack_session_ids": [s[0] for s in sessions],
        "haystack_dates": [s[1] for s in sessions],
        "haystack_sessions": [[{"role": r, "content": c, **({"has_answer": True} if h else {})}
                               for r, c, h in s[2]] for s in sessions],
        "answer_session_ids": [s[0] for s in sessions if any(h for _, _, h in s[2])],
    }


LME = [
    lme_entry("q_dog", "single-session-user", "What breed is my dog?", "Golden retriever", [
        ("s1", "2023/05/02 (Tue) 08:00", [("user", "Thinking about getting a dog.", False)]),
        ("s2", "2023/05/20 (Sat) 09:00", [("user", "Just got a golden retriever puppy!", True),
                                          ("assistant", "Congratulations!", False)]),
    ]),
    lme_entry("q_job", "single-session-user", "What is my sister's job?", "Nurse", [
        ("s3", "2023/05/15 (Mon) 20:00", [("user", "My sister works as a nurse.", True)]),
    ]),
    lme_entry("q_cat_abs", "knowledge-update", "What is my cat's name?", "You never mentioned a cat", [
        ("s4", "2023/05/20 (Sat) 09:00", [("user", "My dog is called Biscuit.", False)]),
    ]),
]


def dmr_record(i, answer, question, sessions):
    return {
        "personas": [[answer], ["x"]], "init_personas": [[], []],
        "dialog": [{"text": "now", "id": "Speaker 1", "convai2_id": f"valid_{i}"}],
        "metadata": {"initial_data_id": f"valid_{i}", "session_id": 4},
        "previous_dialogs": [{"personas": [[]], "dialog": [{"text": t} for t in s],
                              "time_num": 5, "time_unit": "days", "time_back": f"{5 - k} weeks ago"}
                             for k, s in enumerate(sessions)],
        "self_instruct": {"B": question, "A": answer},
    }


DMR = [
    dmr_record(0, "Burger King", "What was the fast food place you said you're working at?", [
        ["I just started at Burger King.", "Cool!"], ["How is work?", "Fine."], ["Busy?", "Yes."], ["Ok", "Ok"]]),
    dmr_record(1, "Taylor Swift", "What artist could you get into?", [
        ["Hi!", "I could get into Taylor Swift lately."], ["a", "b"], ["c", "d"], ["e", "f"]]),
]


class FakeLLM:
    """Answers with the last line of the prompt's context; records prompts."""

    def __init__(self, num_ctx=8192, judge_reply=None):
        self.num_ctx, self.judge_reply, self.prompts = num_ctx, judge_reply, []

    def describe(self):
        return {"provider": "fake", "model": "fake"}

    def chat(self, messages):
        text = messages[-1]["content"]
        self.prompts.append(text)
        reply = self.judge_reply or "I don't know."
        return {"text": reply, "latency_ms": 1.0, "prompt_tokens_est": len(text) // 4, "num_ctx": self.num_ctx}


with tempfile.TemporaryDirectory() as tmp:
    lme_path = Path(tmp) / "lme.json"
    lme_path.write_text(json.dumps(LME))
    dmr_path = Path(tmp) / "dmr.jsonl"
    dmr_path.write_text("\n".join(json.dumps(r) for r in DMR) + "\n")
    os.environ["LONGMEMEVAL_PATH"] = str(lme_path)
    os.environ["DMR_PATH"] = str(dmr_path)

    # --- each question sees only its own history -------------------------------
    llm = FakeLLM()
    fc = baselines.run_baseline("full_context", "longmemeval", llm, smoke=False, opts={"variant": "oracle"})
    by = {d["id"]: d for d in fc["details"]}
    check(set(by) == {"q_dog", "q_job", "q_cat_abs"}, f"one record per question: {sorted(by)}")
    check(by["q_dog"]["context_turns"] == 3 and by["q_job"]["context_turns"] == 1,
          f"full_context holds the question's own turns: {[(i, d['context_turns']) for i, d in by.items()]}")
    dog_prompt = next(p for p in llm.prompts if "breed" in p)
    check("nurse" not in dog_prompt and "Biscuit" not in dog_prompt, "no other question's history leaks in")
    check("[2023/05/20 (Sat) 09:00] Just got a golden retriever" in dog_prompt, "turns carry their session date")
    check(by["q_dog"]["evidence_recall"] == 1.0, "full_context recall is 1.0")
    check(by["q_cat_abs"]["evidence_recall"] is None, "no evidence: recall None")
    check(by["q_dog"]["question"] == "What breed is my dog?", "the judge reads the question without the date line")
    check(by["q_cat_abs"]["abstention"] and by["q_cat_abs"]["category"] == "abstention", "abstention carried")
    check(by["q_dog"]["judge"] is None and fc["accuracy"] is None, "unjudged until finalize")

    rag = baselines.run_baseline("naive_rag", "longmemeval", FakeLLM(), smoke=False,
                                 retriever=baselines.BM25Retriever(), k=1, opts={"variant": "oracle"})
    r = {d["id"]: d for d in rag["details"]}
    check(r["q_dog"]["context_turns"] <= 1 and all(e.startswith("s") for e in r["q_dog"]["retrieved"]),
          f"naive_rag retrieves from the question's own turns: {r['q_dog'].get('retrieved')}")

    nm = baselines.run_baseline("no_memory", "dmr", FakeLLM(), smoke=False)
    check(all(d["context_turns"] == 0 and d["evidence_recall"] is None for d in nm["details"]),
          "no_memory: no context, no recall")
    dfc = baselines.run_baseline("full_context", "dmr", FakeLLM(num_ctx=4), smoke=False)
    d0 = dfc["details"][0]
    check(d0["context_turns"] == 8, f"DMR full_context: the 4 sessions' 8 turns ({d0['context_turns']})")
    check(d0["question"].startswith("Speaker 2 asks Speaker 1") or "?" in d0["question"],
          "DMR asks the question as Campy's runner does")
    check("f1" in d0 and d0["evidence_recall"] == 1.0, f"DMR record: f1 and recall ({d0.get('evidence_recall')})")
    check(d0["context_overflow"] and dfc["context_overflow"] == 2, "a prompt over num_ctx is flagged")

    # --- finalize judges every baseline with the same judge --------------------
    results = {"suites": {}, "baselines": {"config": {}, "full_context": {"longmemeval": {**fc, "valid": True},
                                                                         "dmr": {**dfc, "valid": True}}}}
    judge = FakeLLM(judge_reply="yes CORRECT")
    orig = run_all.resolve_baseline_llm
    run_all.resolve_baseline_llm = lambda *a, **k: (judge, None, "fake")
    try:
        args = SimpleNamespace(judge="auto", judge_provider=None, judge_model=None, judge_base_url=None,
                               baseline_provider=None, baseline_model=None, baseline_base_url=None)
        run_all.finalize_longmemeval(results, args, None, None)
        run_all.finalize_dmr(results, args, None, None)
    finally:
        run_all.resolve_baseline_llm = orig
    lfc = results["baselines"]["full_context"]["longmemeval"]
    check(lfc["accuracy"] is not None and all(d["judge"] is not None for d in lfc["details"]),
          f"LongMemEval baseline judged in a run with no daemon: {lfc.get('accuracy')}")
    check(results["longmemeval_judge"].get("enabled") is True, "judge info recorded")
    check(results["baselines"]["full_context"]["dmr"]["judge_accuracy"] is not None, "DMR baseline judged")

    # --- end to end ------------------------------------------------------------
    env = {k: v for k, v in os.environ.items() if k != "CAMPY_MCP_CMD"}
    out = Path(tmp) / "r.json"
    proc = subprocess.run([sys.executable, str(HERE / "run_all.py"), "--suite", "longmemeval",
                           "--baselines", "no_memory,full_context", "--baselines-only", "--judge", "none",
                           "--baseline-provider", "ollama", "--baseline-model", "none",
                           "--baseline-base-url", "http://127.0.0.1:9", "--out", str(out)],
                          cwd=HERE, env=env, capture_output=True, text=True, timeout=300)
    check(proc.returncode == 0, f"baselines-only run exited {proc.returncode}: {proc.stderr[-800:]}")
    if out.exists():
        bl = json.loads(out.read_text()).get("baselines") or {}
        fcr = (bl.get("full_context") or {}).get("longmemeval") or {}
        # no LLM at that address: the baseline is recorded as invalid, not crashed
        check(fcr and ("valid" in fcr), f"full_context/longmemeval recorded: {str(fcr)[:200]}")

if failures:
    print(f"FAIL -- {len(failures)} own-history baseline checks failed:")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)
print("OK -- LongMemEval and DMR baselines: own history only, recall, judging and end-to-end checks passed")

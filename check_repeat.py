#!/usr/bin/env python3
"""
campy-benchmarks / check_repeat.py

Self-test for run_all.py --repeat N (repeats.py):

  * numeric metrics are averaged, with min/max/stdev in repeat.spread;
  * per-probe verdicts merge to a majority (a tie fails) with a pass_rate,
    and probes whose verdict differed between runs are listed as unstable;
  * a suite invalid in any run is invalid overall (no averaging around a
    failed measurement);
  * --compare marks a change no bigger than the run-to-run range with ≈;
  * end to end in mock mode: `run_all.py --smoke --repeat 2` writes a results
    file with repeat.n == 2 that --compare can read.

Plain script, same convention as the other check_*.py files:

    python3 check_repeat.py

Exits 0 and prints "OK" on success, exits 1 listing each failure.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import repeats
from run_all import format_delta

HERE = Path(__file__).resolve().parent
failures = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


def probe(pid: str, passed: bool, reason: str = "ok") -> dict:
    return {"id": pid, "passed": passed, "reason": reason}


# --- aggregation ------------------------------------------------------------
runs = [
    {"locomo": {"suite": "locomo", "valid": True, "accuracy": 0.8929, "probes": 28, "env": "x",
                "details": [probe("p_6", True), probe("p_8", True), probe("p_9", False, "missing_current_value")]}},
    {"locomo": {"suite": "locomo", "valid": True, "accuracy": 0.8214, "probes": 28, "env": "x",
                "details": [probe("p_6", False, "stale_value_not_marked_superseded"), probe("p_8", True),
                            probe("p_9", False, "missing_current_value")]}},
    {"locomo": {"suite": "locomo", "valid": True, "accuracy": 0.8929, "probes": 28, "env": "x",
                "details": [probe("p_6", True), probe("p_8", True), probe("p_9", False, "missing_current_value")]}},
]
agg = repeats.aggregate_runs(runs)
loc = agg["suites"]["locomo"]
check(abs(loc["accuracy"] - 0.8691) < 1e-4, f"mean accuracy: got {loc['accuracy']}")
check(loc["probes"] == 28 and loc["env"] == "x", "unchanged numeric/non-numeric fields carried over")
sp = agg["repeat"]["spread"]["locomo"]["accuracy"]
check(sp["min"] == 0.8214 and sp["max"] == 0.8929, f"spread min/max: {sp}")
d = {x["id"]: x for x in loc["details"]}
check(d["p_6"]["passed"] is True and abs(d["p_6"]["pass_rate"] - 0.6667) < 1e-4,
      f"p_6 majority pass, rate 2/3: {d['p_6']}")
check(d["p_6"]["reason"] == "ok", "merged reason is the majority verdict's reason")
check(d["p_9"]["passed"] is False and d["p_9"]["pass_rate"] == 0.0, "always-failing probe stays failed")
check([p["id"] for p in agg["repeat"]["unstable_probes"]["locomo"]] == ["p_6"], "only p_6 is unstable")
check(agg["repeat"]["n"] == 3 and len(agg["repeat"]["runs"]) == 3, "repeat.n and raw runs recorded")

tie = repeats.aggregate_runs([{"s": {"valid": True, "details": [probe("a", True)]}},
                              {"s": {"valid": True, "details": [probe("a", False, "x")]}}])
check(tie["suites"]["s"]["details"][0]["passed"] is False, "a 1-1 tie counts as a fail")

bad = repeats.aggregate_runs([{"s": {"valid": True, "accuracy": 1.0}},
                              {"s": {"valid": False, "error": "daemon offline"}}])
check(bad["suites"]["s"]["valid"] is False and "run(s) [2]" in bad["suites"]["s"]["error"]
      and "daemon offline" in bad["suites"]["s"]["error"], f"invalid run poisons the suite: {bad['suites']['s']}")

gone = repeats.aggregate_runs([{"s": {"valid": True}, "t": {"valid": True}}, {"s": {"valid": True}}])
check(gone["suites"]["t"]["valid"] is False, "a suite missing from a run is invalid")

mg = repeats.aggregate_runs([{"memory_gym": {"valid": True, "success_rate": 1.0,
                                             "details": [{"episode": 0, "success": True}]}}] * 2)
check(mg["suites"]["memory_gym"]["details"] == [{"episode": 0, "success": True}],
      "non-verdict details are kept (run 1's)")

jr = repeats.aggregate_runs([
    {"membench": {"valid": True, "judge_accuracy": 1.0, "details": [
        {"id": "p", "passed": False, "reason": "stale", "llm_judge": True}]}},
    {"membench": {"valid": True, "judge_accuracy": 1.0, "details": [
        {"id": "p", "passed": False, "reason": "stale", "llm_judge": True}]}},
    {"membench": {"valid": True, "judge_accuracy": 0.0, "details": [
        {"id": "p", "passed": False, "reason": "stale", "llm_judge": False}]}},
])
jd = jr["suites"]["membench"]["details"][0]
check(jd["llm_judge"] is True and abs(jd["judge_pass_rate"] - 0.6667) < 1e-4,
      f"LLM judge verdicts merge by majority too: {jd}")
check([d["id"] for d in jr["suites"]["membench"]["judge_disagreements"]] == ["p"],
      "judge disagreements recomputed from the merged verdicts")
check(abs(jr["suites"]["membench"]["judge_accuracy"] - 0.6667) < 1e-4, "judge_accuracy is averaged")

# --- noise-aware compare ----------------------------------------------------
noise = repeats.metric_range(agg, "locomo", "accuracy")
check(abs(noise - 0.0715) < 1e-4, f"metric_range: {noise}")
check("≈" in format_delta(0.8929, 0.8214, True, noise), "drop within the range is marked ≈")
check("⚠️" in format_delta(0.8929, 0.70, True, noise), "drop beyond the range is still a warning")
check("⚠️" in format_delta(0.8929, 0.8214, True, None), "no repeat data: unchanged behavior")
check(repeats.metric_range({"suites": {}}, "locomo", "accuracy") is None, "single run has no range")

# --- the merged record shows a run that agrees with the majority -------------
# (hippocampy #278, p3_analytics_pk: the judge passed run 1 and failed runs 2-3;
# the merged record kept run 1's answer and "judge_correct" beside
# llm_judge: False)
mixed = repeats.aggregate_runs([
    {"locomo": {"valid": True, "details": [
        {"id": "p", "passed": True, "reason": "ok", "answer": "A1", "llm_judge": True,
         "llm_judge_reason": "judge_correct"}]}},
    {"locomo": {"valid": True, "details": [
        {"id": "p", "passed": True, "reason": "ok", "answer": "A2", "llm_judge": False,
         "llm_judge_reason": "judge_wrong"}]}},
    {"locomo": {"valid": True, "details": [
        {"id": "p", "passed": False, "reason": "stale", "answer": "A3", "llm_judge": False,
         "llm_judge_reason": "judge_wrong"}]}},
])["suites"]["locomo"]["details"][0]
check(mixed["llm_judge"] is False and mixed["llm_judge_reason"] == "judge_wrong",
      f"merged judge reason matches the merged judge verdict: {mixed}")
check(mixed["answer"] == "A2" and mixed["answer_run"] == 2,
      f"the answer comes from a run agreeing with both majorities (run 2): {mixed}")
check(mixed["passed"] is True and mixed["reason"] == "ok", "lexical majority unchanged")

# --- end to end (mock mode: no daemon, no CAMPY_MCP_CMD) ---------------------
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "r.json"
    env = {k: v for k, v in os.environ.items() if k != "CAMPY_MCP_CMD"}
    proc = subprocess.run([sys.executable, str(HERE / "run_all.py"), "--smoke", "--suite", "locomo",
                           "--repeat", "2", "--out", str(out), "--compare", str(out)],
                          cwd=HERE, env=env, capture_output=True, text=True, timeout=600)
    check(proc.returncode == 0, f"run_all --repeat 2 exited {proc.returncode}: {proc.stderr[-500:]}")
    if out.exists():
        res = json.loads(out.read_text())
        check(res.get("repeat", {}).get("n") == 2, "results file has repeat.n == 2")
        check(res["suites"]["locomo"].get("valid") is True, "aggregated locomo is valid")
    check("Run-to-run spread (2 runs" in proc.stdout, "spread section printed")

    bad_proc = subprocess.run([sys.executable, str(HERE / "run_all.py"), "--repeat", "2"],
                              cwd=HERE, env={**env, "CAMPY_MCP_CMD": "python -m nothing"},
                              capture_output=True, text=True, timeout=60)
    check(bad_proc.returncode != 0 and "needs --isolated" in bad_proc.stderr,
          "real daemon without --isolated is refused")

    # B467: a single run against a real daemon is refused too unless it opts
    # in to the personal store; the refusal comes before any daemon contact.
    for argv, why in ((["--smoke"], "pass --shared-store"),
                      (["--smoke", "--isolated", "--shared-store"], "contradict")):
        p = subprocess.run([sys.executable, str(HERE / "run_all.py"), *argv], cwd=HERE,
                           env={**env, "CAMPY_MCP_CMD": "python -m nothing"},
                           capture_output=True, text=True, timeout=60)
        check(p.returncode == 2 and why in p.stderr,
              f"run_all {' '.join(argv)} with a real daemon is refused ({why!r}): {p.stderr[-300:]}")
    p = subprocess.run([sys.executable, str(HERE / "run_all.py"), "--smoke", "--suite", "locomo"],
                       cwd=HERE, env=env, capture_output=True, text=True, timeout=600)
    check(p.returncode == 0, "mock mode (no CAMPY_MCP_CMD) still runs without --isolated")

if failures:
    print(f"FAIL -- {len(failures)} repeat checks failed:")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)
print("OK -- --repeat aggregation, compare and end-to-end checks passed")

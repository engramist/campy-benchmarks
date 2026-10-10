#!/usr/bin/env python3
"""
campy-benchmarks / check_report.py
Checks report.py's counting rules on synthetic result files: only a real,
isolated run from clean commits reaches the headline; the newest such run
per suite wins; every run is listed in its suite's history with the reason
it doesn't count.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
failures = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


def result(ts, acc, *, mcp=True, isolated=True, dirty=False, sha="a" * 40):
    return {
        "timestamp": ts, "mcp_configured": mcp,
        "provenance": {"harness": {"commit": "b" * 40, "dirty": dirty},
                       "hippocampy": {"commit": sha, "dirty": False},
                       "daemon_config": {"llm_model": "llama3.1:8b"},
                       "store": {"isolated": isolated}},
        "suites": {"longmemeval": {"valid": True, "accuracy": acc, "questions": 35,
                                   "dataset": {"variant": "oracle"},
                                   "by_category": {"multi-session": {"n": 5, "accuracy": acc}}}},
        "longmemeval_judge": {"enabled": True, "llm": {"model": "gemma4:26b"}},
    }


with tempfile.TemporaryDirectory() as tmp:
    files = {
        "old.json": result("2026-10-01T00:00:00Z", 0.40, sha="1" * 40),
        "new.json": result("2026-10-03T00:00:00Z", 0.50, sha="2" * 40),
        "mock.json": result("2026-10-04T00:00:00Z", 0.99, mcp=False),
        "shared.json": result("2026-10-05T00:00:00Z", 0.98, isolated=False),
        "dirty.json": result("2026-10-06T00:00:00Z", 0.97, dirty=True),
    }
    for name, r in files.items():
        (Path(tmp) / name).write_text(json.dumps(r))
    out = subprocess.run([sys.executable, str(HERE / "report.py"), *sorted(str(p) for p in Path(tmp).glob("*.json"))],
                         capture_output=True, text=True).stdout
    head = out.split("## Headline")[1].split("\n## ")[0]
    check("accuracy: **0.500**" in head and "`2222222`" in head, "the newest counted run is the headline")
    for bad in ("0.990", "0.980", "0.970", "0.400"):
        check(bad not in head, f"{bad} must not reach the headline")
    hist = out.split("## LongMemEval")[1]
    check(hist.count("| 2026-") == 5, "every run is in the history")
    for why in ("mock run", "not on an isolated store", "uncommitted harness changes"):
        check(why in hist, f"history says why: {why}")
    check("multi-session" in hist, "per-category table of the newest counted run")

    s_run = result("2026-10-07T00:00:00Z", 0.42, sha="3" * 40)
    s_run["suites"]["longmemeval"]["dataset"] = {"variant": "s"}
    (Path(tmp) / "variant_s.json").write_text(json.dumps(s_run))
    out = subprocess.run([sys.executable, str(HERE / "report.py"), *sorted(str(p) for p in Path(tmp).glob("*.json"))],
                         capture_output=True, text=True).stdout
    head = out.split("## Headline")[1].split("\n## ")[0]
    check("accuracy: **0.500**" in head and "accuracy: **0.420**" in head,
          "each LongMemEval variant keeps its own headline row")

    rep = result("2026-10-08T00:00:00Z", 0.40, sha="2" * 40)  # a second run of new.json's code
    (Path(tmp) / "new_run2.json").write_text(json.dumps(rep))
    out = subprocess.run([sys.executable, str(HERE / "report.py"), *sorted(str(p) for p in Path(tmp).glob("*.json"))],
                         capture_output=True, text=True).stdout
    head = out.split("## Headline")[1].split("\n## ")[0]
    check("accuracy: **0.450** (mean of 2, 0.400–0.500)" in head,
          "repeat runs of the same code are averaged in the headline")
    (Path(tmp) / "new_run2.json").unlink()

    # M0.2: a held-out split is its own row and says so
    d_dev = result("2026-10-09T00:00:00Z", 0.60, sha="2" * 40)
    d_dev["suites"] = {"dmr": {"valid": True, "judge_accuracy": 0.60, "questions": 50,
                               "dataset": {"options": {"max_questions": 50}}}}
    d_held = result("2026-10-09T01:00:00Z", 0.30, sha="2" * 40)
    d_held["suites"] = {"dmr": {"valid": True, "judge_accuracy": 0.30, "questions": 50,
                                "dataset": {"options": {"max_questions": 50, "offset": 50}}}}
    for n, r in (("dmr_dev.json", d_dev), ("dmr_held.json", d_held)):
        (Path(tmp) / n).write_text(json.dumps(r))
    out = subprocess.run([sys.executable, str(HERE / "report.py"), str(Path(tmp) / "dmr_dev.json"),
                          str(Path(tmp) / "dmr_held.json")], capture_output=True, text=True).stdout
    head = out.split("## Headline")[1].split("\n## ")[0]
    check(head.count("DMR (MSC-Self-Instruct)") == 2,
          "dev and held-out DMR runs are separate headline rows")
    check("offset 50" in head, "the held-out row names its selection")
    for n in ("dmr_dev.json", "dmr_held.json"):
        (Path(tmp) / n).unlink()

    (Path(tmp) / "only_mock.json").write_text(json.dumps(result("2026-10-07T00:00:00Z", 1.0, mcp=False)))
    out = subprocess.run([sys.executable, str(HERE / "report.py"), str(Path(tmp) / "only_mock.json")],
                         capture_output=True, text=True).stdout
    check("no counted runs yet" in out, "an empty headline says so")

if failures:
    print(f"FAIL -- {len(failures)} report checks failed:")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)
print("OK -- report counting rules hold")

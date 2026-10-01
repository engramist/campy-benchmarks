"""
campy-benchmarks / repeats.py
Aggregate N repeated runs of the same suites (run_all.py --repeat N).

A single run of an LLM-backed suite is one sample: llama3.1:8b at temperature
0 on Ollama still rephrases answers between runs (concurrent consolidation
calls), so a lexical judge can flip a probe with no code change. On
2026-09-30 two LoCoMo probes flipped (accuracy 0.89 -> 0.82) between two
runs of retrieval-identical code. Repeating the run and reporting the mean
and the spread separates "Campy changed" from "this run was different".

`aggregate_runs` keeps results["suites"] in the single-run shape (each
numeric metric is the mean over runs), so --compare and the summary table
work unchanged, and adds results["repeat"]: every run, the per-metric spread,
and the probes whose verdict differed between runs.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Dict, List, Optional


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _probe_details(details: Any) -> bool:
    """True for per-probe verdict lists (LoCoMo, MemBench): id + passed."""
    return (isinstance(details, list) and bool(details)
            and all(isinstance(d, dict) and "id" in d and "passed" in d for d in details))


def _merge_probe_details(per_run: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """One record per probe: majority verdict, pass rate, most common reason.
    A tie counts as a fail -- "passes half the time" is not a pass."""
    order: List[str] = []
    by_id: Dict[str, List[Dict[str, Any]]] = {}
    for details in per_run:
        for d in details:
            if d["id"] not in by_id:
                order.append(d["id"])
                by_id[d["id"]] = []
            by_id[d["id"]].append(d)
    merged = []
    for pid in order:
        recs = by_id[pid]
        passes = sum(1 for r in recs if r["passed"])
        rate = passes / len(recs)
        rec = dict(recs[0])
        rec["passed"] = rate > 0.5
        rec["pass_rate"] = round(rate, 4)
        rec["runs"] = len(recs)
        reasons = Counter(r.get("reason") for r in recs if r["passed"] == rec["passed"])
        rec["reason"] = reasons.most_common(1)[0][0] if reasons else recs[0].get("reason")
        merged.append(rec)
    return merged


def aggregate_runs(runs: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """`runs`: one executed-suites dict per repeat ({suite: result}).

    Returns {"suites": ..., "repeat": ...}. A suite is valid only if it was
    valid in every run: an invalid run is a failed measurement, and averaging
    around it would hide that.
    """
    n = len(runs)
    suites: Dict[str, Any] = {}
    spread: Dict[str, Dict[str, Dict[str, float]]] = {}
    unstable: Dict[str, List[Dict[str, Any]]] = {}
    for name in [k for k in runs[0]] + [k for r in runs[1:] for k in r if k not in runs[0]]:
        if name in suites:
            continue
        results = [r.get(name) for r in runs]
        missing = [i + 1 for i, r in enumerate(results) if r is None]
        invalid = [i + 1 for i, r in enumerate(results) if r is not None and not r.get("valid")]
        if missing or invalid:
            first_bad = next(r for r in results if r is not None and not r.get("valid")) if invalid else {}
            suites[name] = {
                "suite": name, "valid": False,
                "error": (f"invalid in run(s) {invalid}" if invalid else "")
                         + (f" missing in run(s) {missing}" if missing else "")
                         + (f": {first_bad.get('error', '')}" if first_bad else ""),
            }
            continue
        agg: Dict[str, Any] = {}
        spread[name] = {}
        for key, first in results[0].items():
            if key == "details":
                continue
            values = [r.get(key) for r in results]
            if all(_is_number(v) for v in values):
                agg[key] = round(statistics.fmean(values), 4)
                spread[name][key] = {
                    "min": min(values), "max": max(values),
                    "stdev": round(statistics.pstdev(values), 4),
                }
            else:
                agg[key] = first
        details = [r.get("details") for r in results]
        if all(_probe_details(d) for d in details):
            agg["details"] = _merge_probe_details(details)
            flaky = [{"id": d["id"], "pass_rate": d["pass_rate"]}
                     for d in agg["details"] if 0.0 < d["pass_rate"] < 1.0]
            if flaky:
                unstable[name] = flaky
        elif details[0] is not None:
            agg["details"] = details[0]  # non-verdict details: run 1's, all runs are in repeat.runs
        agg["valid"] = True
        suites[name] = agg
    return {
        "suites": suites,
        "repeat": {"n": n, "spread": spread, "unstable_probes": unstable, "runs": runs},
    }


def metric_range(results: Dict[str, Any], suite: str, key: str) -> Optional[float]:
    """max - min of a metric over a repeated run's runs, or None."""
    s = ((results.get("repeat") or {}).get("spread") or {}).get(suite, {}).get(key)
    return None if s is None else float(s["max"]) - float(s["min"])


def print_spread(results: Dict[str, Any], rows) -> None:
    """Per-metric mean and min-max over the runs, and the unstable probes."""
    rep = results.get("repeat")
    if not rep:
        return
    print(f"\n### Run-to-run spread ({rep['n']} runs; scores above are means)")
    for suite, key, label, _ in rows:
        s = rep["spread"].get(suite, {}).get(key)
        mean = results.get("suites", {}).get(suite, {}).get(key)
        if s is not None:
            flag = "" if s["min"] == s["max"] else "  <- varies"
            print(f"- {label}: {mean} (min {s['min']}, max {s['max']}, stdev {s['stdev']}){flag}")
    for suite, probes in rep["unstable_probes"].items():
        ids = ", ".join(f"{p['id']} ({p['pass_rate']:.0%})" for p in probes)
        print(f"- unstable {suite} probes (verdict differed between runs): {ids}")

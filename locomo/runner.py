"""
campy-benchmarks / locomo / runner.py
LoCoMo Benchmark Runner: Evaluates Multi-Session Factual Recall & Dynamic Constraint Deprecation.

Scoring is scorer v2 (scoring.py): `accuracy` (every probe judged) and
`deprecation_accuracy` (deprecation probes only) are the headline numbers.
`exact_match` is now "names an accepted form of the current value" and `f1`
is token overlap with the short gold string (understates full-sentence
answers). Every probe's question, answer and verdict is in `details`.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from mcp_client import CampyMCPClient
from locomo.dataset import get_locomo_scenarios
from records import summarize_bundle
from scoring import compute_f1, contains_current_value, judge


def run_locomo(client: CampyMCPClient, smoke: bool = False, trace_context: bool = False) -> Dict[str, Any]:
    """Execute LoCoMo benchmark suite against Campy.

    trace_context: also call compile_context per probe and record what it
    retrieved. This is an extra daemon call per probe and is NOT necessarily
    the bundle `ask` used internally (ask augments the query first); it is
    a diagnostic view of retrieval, not ask's exact prompt.
    """
    scenarios = get_locomo_scenarios(smoke=smoke)

    details: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for sc in scenarios:
        session_prefix = f"locomo_{sc.id}"
        # Ingest sessions sequentially
        for sess_idx, session in enumerate(sc.sessions, start=1):
            sess_id = f"{session_prefix}_s{sess_idx}"
            for turn in session:
                client.notify_turn(role=turn["role"], content=turn["content"], session_id=sess_id)
            # Settle: wait for consolidation to drain (see mcp_client.run_sweep)
            client.run_sweep()

        # Evaluate probe questions
        for probe in sc.probes:
            t0 = time.perf_counter()
            answer = client.ask(query=probe.question, session_id=f"{session_prefix}_eval")
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            passed, reason = judge(answer, probe.kind, probe.accept, probe.stale)
            record: Dict[str, Any] = {
                "id": probe.id,
                "scenario": sc.id,
                "question": probe.question,
                "expected": probe.expected,
                "answer": answer,
                "passed": passed,
                "reason": reason,
                "exact_match": contains_current_value(answer, probe.accept),
                "f1": round(compute_f1(answer, probe.expected), 4),
                "is_deprecation": probe.is_deprecation,
                "latency_ms": round(lat_ms, 1),
            }
            if trace_context:
                record["context"] = summarize_bundle(client.compile_context(probe.question))
            details.append(record)

    n = len(details)
    dep = [d for d in details if d["is_deprecation"]]
    return {
        "suite": "locomo",
        "scenarios": len(scenarios),
        "probes": n,
        "accuracy": round(sum(d["passed"] for d in details) / max(1, n), 4),
        "exact_match": round(sum(d["exact_match"] for d in details) / max(1, n), 4),
        "f1": round(sum(d["f1"] for d in details) / max(1, n), 4),
        "deprecation_accuracy": round(sum(d["passed"] for d in dep) / len(dep), 4) if dep else None,
        "deprecation_probes": len(dep),
        "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 2),
        "details": details,
    }

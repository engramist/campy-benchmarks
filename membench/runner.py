"""
campy-benchmarks / membench / runner.py
MemBench Benchmark Runner: Evaluates Multi-Session Persona Fact Retention, Contradiction Arbitration & Token Savings.

Scoring is scorer v2 (scoring.py). `accuracy` replaces the old
fact_precision/fact_recall pair, which were the same number computed twice.
Every probe's question, answer, verdict and compile_context summary is in
`details`.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from mcp_client import CampyMCPClient
from membench.msc_dataset import get_msc_personas
from records import summarize_bundle
from scoring import judge


def estimate_tokens(text: str) -> int:
    """Rough token estimation (approx 4 chars per token)."""
    return max(1, len(text) // 4)


def run_membench(client: CampyMCPClient, smoke: bool = False, trace_context: bool = False) -> Dict[str, Any]:
    """Execute MemBench evaluation suite. (`trace_context` is accepted for a
    uniform runner signature; this suite always records its bundle.)"""
    personas = get_msc_personas(smoke=smoke)

    details: List[Dict[str, Any]] = []
    total_raw_tokens = 0
    total_bundle_tokens = 0

    for p in personas:
        # Ingest 5 sessions
        session_text_acc = []
        for s_idx, session in enumerate(p.sessions, start=1):
            sess_id = f"msc_{p.id}_s{s_idx}"
            for turn in session:
                content = turn["content"]
                session_text_acc.append(content)
                client.notify_turn(role=turn["role"], content=content, session_id=sess_id)
            client.run_sweep()

        total_raw_tokens += sum(estimate_tokens(t) for t in session_text_acc)

        # In Session 5, evaluate probes
        for probe in p.probes:
            t0 = time.perf_counter()
            ctx_res = client.compile_context(probe.question, token_budget=4000)
            compile_ms = (time.perf_counter() - t0) * 1000.0
            # B436: the estimate is nested at bundle.total_token_estimate.
            # Missing means the bundle shape changed -- count 0 and let the
            # record show it rather than substituting a constant.
            bundle_tokens = ctx_res.get("bundle", {}).get("total_token_estimate") or 0
            total_bundle_tokens += bundle_tokens

            t1 = time.perf_counter()
            answer = client.ask(probe.question, session_id=f"msc_{p.id}_s5")
            ask_ms = (time.perf_counter() - t1) * 1000.0

            passed, reason = judge(answer, probe.kind, probe.accept, probe.stale)
            details.append({
                "id": probe.id,
                "persona": p.id,
                "question": probe.question,
                "expected": probe.expected_active,
                "answer": answer,
                "passed": passed,
                "reason": reason,
                "is_contradiction": probe.is_contradiction,
                "ask_latency_ms": round(ask_ms, 1),
                "compile_latency_ms": round(compile_ms, 1),
                "context": summarize_bundle(ctx_res),
            })

    # Token savings: compiled bundle vs raw transcript. Meaningless when the
    # bundles are empty (B454: that read as "100% savings") -> None.
    token_savings: Optional[float] = None
    if total_raw_tokens > 0 and total_bundle_tokens > 0:
        token_savings = round(max(0.0, 1.0 - total_bundle_tokens / total_raw_tokens) * 100.0, 2)

    n = len(details)
    contra = [d for d in details if d["is_contradiction"]]
    return {
        "suite": "membench",
        "personas": len(personas),
        "total_probes": n,
        "accuracy": round(sum(d["passed"] for d in details) / max(1, n), 4),
        "contradiction_score": round(sum(d["passed"] for d in contra) / len(contra), 4) if contra else None,
        "contradiction_probes": len(contra),
        "raw_tokens_avg": round(total_raw_tokens / max(1, len(personas)), 1),
        "bundle_tokens_avg": round(total_bundle_tokens / max(1, len(personas)), 1),
        "token_savings_pct": token_savings,
        "avg_latency_ms": round(sum(d["ask_latency_ms"] for d in details) / max(1, n), 2),
        "avg_compile_latency_ms": round(sum(d["compile_latency_ms"] for d in details) / max(1, n), 2),
        "details": details,
    }

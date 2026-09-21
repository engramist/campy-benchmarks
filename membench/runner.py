"""
campy-benchmarks / membench / runner.py
MemBench Benchmark Runner: Evaluates Multi-Session Persona Fact Retention, Contradiction Arbitration & Token Savings.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from mcp_client import CampyMCPClient
from membench.msc_dataset import get_msc_personas, MSCPersona, ContradictionProbe


def estimate_tokens(text: str) -> int:
    """Rough token estimation (approx 4 chars per token)."""
    return max(1, len(text) // 4)


def run_membench(client: CampyMCPClient, smoke: bool = False) -> Dict[str, Any]:
    """Execute MemBench evaluation suite."""
    personas = get_msc_personas(smoke=smoke)
    
    total_probes = 0
    facts_recalled = 0
    contradictions_tested = 0
    contradictions_resolved = 0
    
    total_raw_tokens = 0
    total_bundle_tokens = 0
    latencies: List[float] = []

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

        # Compute raw conversation tokens
        raw_tokens = sum(estimate_tokens(t) for t in session_text_acc)
        total_raw_tokens += raw_tokens

        # In Session 5, evaluate probes
        for probe in p.probes:
            total_probes += 1
            t0 = time.perf_counter()
            
            # Query compiled context and answer
            ctx_res = client.compile_context(probe.question, token_budget=4000)
            # B436: compile_context's real response nests the estimate at
            # bundle.total_token_estimate -- there is no top-level
            # "token_count" key, so this always fell through to the 380
            # default and token_savings_pct was always computed from a
            # constant.
            bundle_tokens = ctx_res.get("bundle", {}).get("total_token_estimate", 380)
            total_bundle_tokens += bundle_tokens

            answer = client.ask(probe.question, session_id=f"msc_{p.id}_s5")
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            # Check must-match and must-not-match
            hit_must = all(re.search(pat, answer, re.IGNORECASE) for pat in probe.must_match)
            hit_must_not = any(re.search(pat, answer, re.IGNORECASE) for pat in probe.must_not_match)

            if hit_must and not hit_must_not:
                facts_recalled += 1
                if probe.is_contradiction:
                    contradictions_resolved += 1
            
            if probe.is_contradiction:
                contradictions_tested += 1

    # Token savings ratio: compare Campy's compiled context bundle vs raw transcript
    token_savings = 0.0
    if total_raw_tokens > 0:
        token_savings = max(0.0, (1.0 - (total_bundle_tokens / total_raw_tokens))) * 100.0

    precision = round(facts_recalled / max(1, total_probes), 4)
    recall = round(facts_recalled / max(1, total_probes), 4)
    contra_score = round(contradictions_resolved / max(1, contradictions_tested), 4) if contradictions_tested else 1.0

    return {
        "suite": "membench",
        "personas": len(personas),
        "total_probes": total_probes,
        "fact_precision": precision,
        "fact_recall": recall,
        "contradiction_score": contra_score,
        "raw_tokens_avg": round(total_raw_tokens / max(1, len(personas)), 1),
        "bundle_tokens_avg": round(total_bundle_tokens / max(1, len(personas)), 1),
        "token_savings_pct": round(token_savings, 2),
        "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 2),
    }

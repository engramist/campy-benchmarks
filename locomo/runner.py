"""
campy-benchmarks / locomo / runner.py
LoCoMo Benchmark Runner: Evaluates Multi-Session Factual Recall & Dynamic Constraint Deprecation.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Set

from mcp_client import CampyMCPClient
from locomo.dataset import get_locomo_scenarios, LoCoMoScenario, ProbeQuestion


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = re.findall(r"\w+", prediction.lower())
    truth_tokens = re.findall(r"\w+", ground_truth.lower())
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    precision = sum(min(pred_tokens.count(w), truth_tokens.count(w)) for w in common) / len(pred_tokens)
    recall = sum(min(pred_tokens.count(w), truth_tokens.count(w)) for w in common) / len(truth_tokens)
    return 2 * (precision * recall) / (precision + recall)


def run_locomo(client: CampyMCPClient, smoke: bool = False) -> Dict[str, Any]:
    """Execute LoCoMo benchmark suite against Campy."""
    scenarios = get_locomo_scenarios(smoke=smoke)
    
    total_probes = 0
    exact_matches = 0
    f1_scores: List[float] = []
    deprecation_probes = 0
    deprecation_correct = 0
    latencies: List[float] = []

    for sc in scenarios:
        session_prefix = f"locomo_{sc.id}"
        # Ingest sessions sequentially
        for sess_idx, session in enumerate(sc.sessions, start=1):
            sess_id = f"{session_prefix}_s{sess_idx}"
            for turn in session:
                client.notify_turn(role=turn["role"], content=turn["content"], session_id=sess_id)
            # Intermediate sweep simulates consolidation
            client.run_sweep()

        # Evaluate probe questions
        for probe in sc.probes:
            total_probes += 1
            t0 = time.perf_counter()
            answer = client.ask(query=probe.question, session_id=f"{session_prefix}_eval")
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            # Check Exact Match
            em = 1.0 if probe.expected.lower() in answer.lower() else 0.0
            exact_matches += int(em)

            # Check F1
            f1 = compute_f1(answer, probe.expected)
            f1_scores.append(f1)

            # Check Deprecation
            if probe.is_deprecation:
                deprecation_probes += 1
                hit_must = all(re.search(pat, answer, re.IGNORECASE) for pat in probe.must_match)
                hit_must_not = any(re.search(pat, answer, re.IGNORECASE) for pat in probe.must_not_match)
                if hit_must and not hit_must_not:
                    deprecation_correct += 1

    return {
        "suite": "locomo",
        "scenarios": len(scenarios),
        "probes": total_probes,
        "exact_match": round(exact_matches / max(1, total_probes), 4),
        "f1": round(sum(f1_scores) / max(1, len(f1_scores)), 4),
        "deprecation_accuracy": round(deprecation_correct / max(1, deprecation_probes), 4) if deprecation_probes else 1.0,
        "deprecation_probes": deprecation_probes,
        "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 2),
    }

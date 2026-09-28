"""
campy-benchmarks / arc_bridge / runner.py
ARC-AGI Sibling Bridge: Integrates memory-transfer diagnostics from sibling ARC_AGI repository.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

from mcp_client import CampyMCPClient
from records import result_texts, snippet

ARC_AGI_REPO = Path(os.environ.get("ARC_AGI_REPO", "/Users/djshelton/Desktop/GitProjects/ARC_AGI"))


def run_arc_bridge(client: CampyMCPClient, smoke: bool = False, trace_context: bool = False) -> Dict[str, Any]:
    """Execute ARC-AGI memory transfer and diagnostic suite."""
    has_sibling = ARC_AGI_REPO.exists()
    
    # 1. Measure Hot-Path Latency (equivalent to test_a059)
    # Warm up call
    client.current_truth("arc3 test warm up query")
    t0 = time.perf_counter()
    for _ in range(5 if smoke else 20):
        client.current_truth("active puzzle mechanic for game_001")
    hot_path_latency = ((time.perf_counter() - t0) / (5 if smoke else 20)) * 1000.0

    # 2. Rule transfer (equivalent to test_a084): two games share a mechanic;
    # a query by mechanic must surface BOTH games' rules. Only retrieved hit
    # text is checked -- the old check searched str(response) for
    # "gravity_pull", which is in the query itself, and a miss reported a
    # hard-coded 0.85 instead of a measurement.
    client.notify_turn(
        role="assistant",
        content="Game_A rule: blue pixels move right when adjacent to green. Mechanic: gravity_pull.",
        session_id="arc_game_a",
    )
    client.notify_turn(
        role="assistant",
        content="Game_B rule: red pixels fall downward. Mechanic: gravity_pull.",
        session_id="arc_game_b",
    )
    client.run_sweep()

    rule_hits = result_texts(client.current_truth("mechanic rules for gravity_pull"))
    joined = " ".join(rule_hits).lower()
    games_found = {
        "game_a": "blue pixels move right" in joined,
        "game_b": "red pixels fall downward" in joined,
    }
    rule_transfer_rate = sum(games_found.values()) / len(games_found)

    # 3. Disappeared entity persistence (equivalent to test_a221): the
    # coordinates must come back (the old check also accepted "yellow key",
    # which is in the query; a miss reported a hard-coded 0.90). Settle first
    # -- the old code probed immediately after the write.
    client.notify_turn(
        role="assistant",
        content="Entity E12: yellow key located at (4, 7). Frame 12: occluded by moving wall.",
        session_id="arc_occlusion_test",
    )
    client.run_sweep()
    occl_hits = result_texts(client.current_truth("location of yellow key E12"))
    entity_recalled = any(re.search(r"(?<![\d.])4\s*,\s*7(?![\d.])", t) for t in occl_hits)
    disappeared_entity_recall = 1.0 if entity_recalled else 0.0

    # Optional: If not in smoke mode and sibling venv is present, attempt pytest check
    pytest_executed = False
    if not smoke and has_sibling:
        pytest_bin = ARC_AGI_REPO / ".venv" / "bin" / "pytest"
        if pytest_bin.exists():
            try:
                proc = subprocess.run(
                    [
                        str(pytest_bin),
                        str(ARC_AGI_REPO / "tests" / "test_a059_memory_hot_path_latency.py"),
                        "-q",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if proc.returncode == 0:
                    pytest_executed = True
            except Exception:
                pass

    return {
        "suite": "arc_bridge",
        "sibling_repo_present": has_sibling,
        "sibling_tests_executed": pytest_executed,
        "hot_path_latency_ms": round(hot_path_latency, 2),
        "rule_transfer_rate": round(rule_transfer_rate, 4),
        "disappeared_entity_recall": round(disappeared_entity_recall, 4),
        "target_hot_path_ms": "<5.0ms (cached), <50ms (fresh)",
        "details": {
            "rule_transfer": {"games_found": games_found,
                              "hits": [snippet(t) for t in rule_hits[:5]]},
            "disappeared_entity": {"recalled": entity_recalled,
                                   "hits": [snippet(t) for t in occl_hits[:5]]},
        },
    }

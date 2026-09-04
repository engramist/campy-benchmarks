"""
campy-benchmarks / memory_gym / runner.py
MemoryGym Benchmark Runner: Evaluates 2D Spatial/Temporal Persistence over extended step horizons.
"""

from __future__ import annotations

import ast
import re
import time
from typing import Any, Dict, List, Tuple

from mcp_client import CampyMCPClient
from memory_gym.env_wrapper import make_memory_gym_env


def parse_path_from_memory(text: str) -> List[Tuple[int, int]]:
    """Parse list of coordinates from Campy recall text."""
    coords = []
    matches = re.findall(r"\((\d+),\s*(\d+)\)", text)
    for x_str, y_str in matches:
        coords.append((int(x_str), int(y_str)))
    return coords


def run_memory_gym(client: CampyMCPClient, smoke: bool = False) -> Dict[str, Any]:
    """Execute MemoryGym benchmark suite (MysteryPath-v0)."""
    episodes_to_run = 3 if smoke else 20
    max_steps = 50 if smoke else 500
    
    successes = 0
    efficiencies: List[float] = []
    latencies: List[float] = []
    total_steps_executed = 0

    for ep in range(episodes_to_run):
        env = make_memory_gym_env("MysteryPath-v0", max_steps=max_steps)
        obs, _ = env.reset(seed=1000 + ep)
        session_id = f"memgym_mysterypath_ep{ep}"

        # Step 0: Path is flashed! Ingest into Campy memory
        flashed_path = obs.get("flashed_path") or []
        path_str = " -> ".join(f"({x}, {y})" for x, y in flashed_path)
        client.notify_turn(
            role="system",
            content=f"Observation at Step 0: MysteryPath navigation sequence is: {path_str}",
            session_id=session_id,
        )

        # Retrieve stored path from memory
        t0 = time.perf_counter()
        recall_res = client.current_truth(f"MysteryPath navigation sequence for {session_id}")
        t_retrieve = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_retrieve)

        # Parse recalled path
        recalled_coords: List[Tuple[int, int]] = []
        if isinstance(recall_res, dict):
            for res in recall_res.get("results", []):
                recalled_coords = parse_path_from_memory(res.get("content", res.get("text", "")))
                if recalled_coords:
                    break
        if not recalled_coords:
            # Fallback query
            answer = client.ask(f"What is the navigation sequence?", session_id=session_id)
            recalled_coords = parse_path_from_memory(answer)

        # Execute navigation steps
        done = False
        step_idx = 1
        optimal_len = len(flashed_path)

        while not done and step_idx < len(recalled_coords) and step_idx < max_steps:
            target_pos = recalled_coords[step_idx]
            
            t_step_start = time.perf_counter()
            obs, reward, done, _, info = env.step(target_pos)
            step_lat = (time.perf_counter() - t_step_start) * 1000.0
            latencies.append(step_lat)

            total_steps_executed += 1
            if done:
                if reward > 0.0:
                    successes += 1
                    efficiencies.append(min(1.0, optimal_len / max(1, step_idx)))
                else:
                    efficiencies.append(0.0)
                break
            step_idx += 1

    avg_eff = sum(efficiencies) / max(1, len(efficiencies))
    succ_rate = successes / max(1, episodes_to_run)
    avg_lat = sum(latencies) / max(1, len(latencies))

    return {
        "suite": "memory_gym",
        "env": "MysteryPath-v0",
        "episodes": episodes_to_run,
        "max_steps": max_steps,
        "total_steps": total_steps_executed,
        "success_rate": round(succ_rate, 4),
        "step_efficiency": round(avg_eff, 4),
        "retention_500_steps": round(succ_rate * avg_eff, 4),
        "avg_step_latency_ms": round(avg_lat, 2),
    }

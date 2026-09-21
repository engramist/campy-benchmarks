#!/usr/bin/env python3
"""
Campy Benchmarks Unified Runner (B381)
Executes evaluation suites against HippoCampy over MCP transport.
Supports --smoke, --baseline, and --compare flags.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent directory to sys.path so modules import cleanly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp_client import CampyMCPClient
from locomo.runner import run_locomo
from memory_gym.runner import run_memory_gym
from membench.runner import run_membench
from arc_bridge.runner import run_arc_bridge


def format_delta(baseline_val: Any, current_val: Any, higher_is_better: bool = True) -> str:
    """Format comparison delta string."""
    try:
        b = float(baseline_val)
        c = float(current_val)
        delta = c - b
        pct = (delta / b * 100.0) if b != 0 else 0.0
        sign = "+" if delta > 0 else ""
        icon = "✅" if (delta >= 0 if higher_is_better else delta <= 0) else "⚠️"
        return f"{c:.4f} ({sign}{delta:.4f} / {sign}{pct:.1f}%) {icon}"
    except Exception:
        return f"{current_val} (vs {baseline_val})"


def print_comparison_table(baseline: Dict[str, Any], current: Dict[str, Any]) -> None:
    """Print markdown comparison table between baseline and current run."""
    print("\n" + "=" * 80)
    print("### HippoCampy Decision-Grade Benchmark Comparison Report (B381)")
    print("=" * 80 + "\n")
    print(f"| Suite / Metric | Baseline Snapshot | Current Evaluation | Evaluation Status |")
    print(f"|---|---|---|---|")

    # LoCoMo
    b_locomo = baseline.get("suites", {}).get("locomo", {})
    c_locomo = current.get("suites", {}).get("locomo", {})
    print(f"| **LoCoMo Exact Match** | {b_locomo.get('exact_match', 'N/A')} | {format_delta(b_locomo.get('exact_match', 0), c_locomo.get('exact_match', 0), True)} |")
    print(f"| **LoCoMo F1 Score** | {b_locomo.get('f1', 'N/A')} | {format_delta(b_locomo.get('f1', 0), c_locomo.get('f1', 0), True)} |")
    print(f"| **LoCoMo Deprecation Accuracy** | {b_locomo.get('deprecation_accuracy', 'N/A')} | {format_delta(b_locomo.get('deprecation_accuracy', 0), c_locomo.get('deprecation_accuracy', 0), True)} |")

    # MemoryGym
    b_mg = baseline.get("suites", {}).get("memory_gym", {})
    c_mg = current.get("suites", {}).get("memory_gym", {})
    print(f"| **MemoryGym Success Rate** | {b_mg.get('success_rate', 'N/A')} | {format_delta(b_mg.get('success_rate', 0), c_mg.get('success_rate', 0), True)} |")
    print(f"| **MemoryGym Step Efficiency** | {b_mg.get('step_efficiency', 'N/A')} | {format_delta(b_mg.get('step_efficiency', 0), c_mg.get('step_efficiency', 0), True)} |")
    print(f"| **MemoryGym 500-Step Retention** | {b_mg.get('retention_500_steps', 'N/A')} | {format_delta(b_mg.get('retention_500_steps', 0), c_mg.get('retention_500_steps', 0), True)} |")

    # MemBench
    b_mb = baseline.get("suites", {}).get("membench", {})
    c_mb = current.get("suites", {}).get("membench", {})
    print(f"| **MemBench Fact Precision** | {b_mb.get('fact_precision', 'N/A')} | {format_delta(b_mb.get('fact_precision', 0), c_mb.get('fact_precision', 0), True)} |")
    print(f"| **MemBench Contradiction Score** | {b_mb.get('contradiction_score', 'N/A')} | {format_delta(b_mb.get('contradiction_score', 0), c_mb.get('contradiction_score', 0), True)} |")
    print(f"| **MemBench Token Savings %** | {b_mb.get('token_savings_pct', 'N/A')}% | {format_delta(b_mb.get('token_savings_pct', 0), c_mb.get('token_savings_pct', 0), True)} |")

    # ARC Bridge
    b_arc = baseline.get("suites", {}).get("arc_bridge", {})
    c_arc = current.get("suites", {}).get("arc_bridge", {})
    print(f"| **ARC Hot-Path Latency (ms)** | {b_arc.get('hot_path_latency_ms', 'N/A')}ms | {format_delta(b_arc.get('hot_path_latency_ms', 0), c_arc.get('hot_path_latency_ms', 0), False)} |")
    print(f"| **ARC Rule Transfer Rate** | {b_arc.get('rule_transfer_rate', 'N/A')} | {format_delta(b_arc.get('rule_transfer_rate', 0), c_arc.get('rule_transfer_rate', 0), True)} |")
    print(f"| **ARC Disappearance Recall** | {b_arc.get('disappeared_entity_recall', 'N/A')} | {format_delta(b_arc.get('disappeared_entity_recall', 0), c_arc.get('disappeared_entity_recall', 0), True)} |")
    print("\n" + "=" * 80 + "\n")


def print_summary_table(results: Dict[str, Any]) -> None:
    """Print formatted summary table using Rich or standard markdown."""
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(title=f"Campy Benchmark Results ({'SMOKE' if results.get('mode') == 'smoke' else 'FULL'})")
        table.add_column("Suite", style="cyan bold")
        table.add_column("Key Metrics", style="magenta")
        table.add_column("Score / Rate", style="green bold")
        table.add_column("Latency", style="yellow")

        suites = results.get("suites", {})
        if "locomo" in suites:
            loc = suites["locomo"]
            table.add_row("LoCoMo", f"F1: {loc.get('f1')} | DeprecAcc: {loc.get('deprecation_accuracy')}", f"EM: {loc.get('exact_match')}", f"{loc.get('avg_latency_ms')} ms")
        if "memory_gym" in suites:
            mg = suites["memory_gym"]
            table.add_row("MemoryGym", f"Efficiency: {mg.get('step_efficiency')} | Steps: {mg.get('total_steps')}", f"Success: {mg.get('success_rate') * 100:.1f}%", f"{mg.get('avg_step_latency_ms')} ms")
        if "membench" in suites:
            mb = suites["membench"]
            table.add_row("MemBench", f"Contradiction: {mb.get('contradiction_score')} | Savings: {mb.get('token_savings_pct')}%", f"Precision: {mb.get('fact_precision')}", f"{mb.get('avg_latency_ms')} ms")
        if "arc_bridge" in suites:
            arc = suites["arc_bridge"]
            table.add_row("ARC Bridge", f"Transfer: {arc.get('rule_transfer_rate')} | Disappear: {arc.get('disappeared_entity_recall')}", f"Active", f"{arc.get('hot_path_latency_ms')} ms")

        console.print(table)
    except Exception:
        # Fallback to plain text
        print("\n=== Benchmark Execution Summary ===")
        for name, data in results.get("suites", {}).items():
            print(f"- {name.upper()}: {data}")


def main():
    parser = argparse.ArgumentParser(description="Run Campy External Benchmarks Harness")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke checks across all suites")
    parser.add_argument("--baseline", action="store_true", help="Record current baseline snapshot")
    parser.add_argument("--compare", type=str, nargs="?", const="baseline_snapshot.json", help="Compare against baseline JSON file")
    parser.add_argument("--out", type=str, default="baseline_snapshot.json", help="Output file for baseline snapshot")
    parser.add_argument("--suite", choices=["all", "locomo", "memory_gym", "membench", "arc"], default="all")
    args = parser.parse_args()

    mcp_cmd = os.environ.get("CAMPY_MCP_CMD")
    print(f"=== Campy Benchmark Harness (B381) ===")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}Z")
    print(f"Mode: {'SMOKE' if args.smoke else 'STANDARD'}")
    print(f"Target MCP Server: {mcp_cmd or 'Standalone / Fallback Mock'}")

    client = CampyMCPClient(mcp_cmd=mcp_cmd)

    suites_to_run = ["locomo", "memory_gym", "membench", "arc"] if args.suite == "all" else [args.suite]
    executed_suites: Dict[str, Any] = {}

    try:
        if "locomo" in suites_to_run:
            print("\n[+] Running LoCoMo Suite (Conversational Deprecation)...")
            t0 = time.time()
            client.reset_mock_state()  # B438: isolate from any prior suite's mock data
            executed_suites["locomo"] = run_locomo(client, smoke=args.smoke)
            print(f"    Completed in {time.time() - t0:.2f}s")

        if "memory_gym" in suites_to_run:
            print("\n[+] Running MemoryGym Suite (2D Spatial/Temporal Persistence)...")
            t0 = time.time()
            client.reset_mock_state()  # B438
            executed_suites["memory_gym"] = run_memory_gym(client, smoke=args.smoke)
            print(f"    Completed in {time.time() - t0:.2f}s")

        if "membench" in suites_to_run:
            print("\n[+] Running MemBench Suite (Persona & Contradiction Arbitration)...")
            t0 = time.time()
            client.reset_mock_state()  # B438
            executed_suites["membench"] = run_membench(client, smoke=args.smoke)
            print(f"    Completed in {time.time() - t0:.2f}s")

        if "arc" in suites_to_run or "arc_bridge" in suites_to_run:
            print("\n[+] Running ARC Bridge Suite (World Model & Memory Transfer)...")
            t0 = time.time()
            client.reset_mock_state()  # B438
            executed_suites["arc_bridge"] = run_arc_bridge(client, smoke=args.smoke)
            print(f"    Completed in {time.time() - t0:.2f}s")
    finally:
        client.close()

    results: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke" if args.smoke else "standard",
        "mcp_configured": bool(mcp_cmd),
        "suites": executed_suites,
        "canonical_baseline_targets": {
            "retrieval_latency_ms": "<10.0ms (via B375)",
            "llm_generation_latency_s": "<1.0s (via B374)",
            "daemon_idle_rss_mb": "<80MB (via B384)",
            "token_compression_ratio": "50%-70% bulk, 100% bypass on sub-budget (B374)",
            "ask_eval_overall": ">=0.90",
        },
    }

    print_summary_table(results)

    if args.baseline:
        out_path = Path(args.out)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[+] Successfully recorded baseline snapshot to {out_path.resolve()}")

    if args.compare:
        baseline_file = Path(args.compare)
        if baseline_file.exists():
            with open(baseline_file, "r") as f:
                baseline_data = json.load(f)
            print_comparison_table(baseline_data, results)
        else:
            print(f"\n[!] Baseline file {baseline_file} not found; skipping comparison.")

    print("\nHarness execution complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

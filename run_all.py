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
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent directory to sys.path so modules import cleanly
sys.path.insert(0, str(Path(__file__).resolve().parent))

import provenance
from isolation import IsolatedDaemon
from mcp_client import CampyMCPClient, CampyClientError
from locomo.runner import run_locomo
from memory_gym.runner import run_memory_gym
from membench.runner import run_membench
from arc_bridge.runner import run_arc_bridge


def format_delta(baseline_val: Any, current_val: Any, higher_is_better: bool = True) -> str:
    """Format comparison delta string."""
    if baseline_val is None or current_val is None:
        return f"{current_val if current_val is not None else 'N/A'}"
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


# (suite, key, label, higher_is_better)
COMPARE_ROWS = [
    ("locomo", "accuracy", "LoCoMo Accuracy", True),
    ("locomo", "deprecation_accuracy", "LoCoMo Deprecation Accuracy", True),
    ("locomo", "exact_match", "LoCoMo Exact Match", True),
    ("locomo", "f1", "LoCoMo F1 (token overlap)", True),
    ("memory_gym", "success_rate", "MemoryGym Success Rate", True),
    ("memory_gym", "step_efficiency", "MemoryGym Step Efficiency", True),
    ("membench", "accuracy", "MemBench Accuracy", True),
    ("membench", "contradiction_score", "MemBench Contradiction Score", True),
    ("membench", "token_savings_pct", "MemBench Token Savings %", True),
    ("arc_bridge", "hot_path_latency_ms", "ARC Hot-Path Latency (ms)", False),
    ("arc_bridge", "rule_transfer_rate", "ARC Rule Transfer Rate", True),
    ("arc_bridge", "disappeared_entity_recall", "ARC Disappearance Recall", True),
]


def comparability_warnings(baseline: Dict[str, Any], current: Dict[str, Any]) -> list:
    """Reasons the two runs' scores are not like-for-like."""
    bp, cp = baseline.get("provenance", {}), current.get("provenance", {})
    warnings = []
    b_sv, c_sv = bp.get("scorer_version", 1), cp.get("scorer_version", 1)
    if b_sv != c_sv:
        warnings.append(f"scorer_version differs (baseline v{b_sv}, current v{c_sv}): "
                        "accuracy-type metrics are NOT comparable")
    if baseline.get("mode") != current.get("mode"):
        warnings.append(f"mode differs ({baseline.get('mode')} vs {current.get('mode')})")
    b_ds, c_ds = bp.get("dataset_sha"), cp.get("dataset_sha")
    if b_ds and c_ds and b_ds != c_ds:
        warnings.append(f"dataset fixtures differ ({b_ds} vs {c_ds})")
    b_iso = (bp.get("store") or {}).get("isolated", False)
    c_iso = (cp.get("store") or {}).get("isolated", False)
    if b_iso != c_iso:
        warnings.append(f"store isolation differs (baseline isolated={b_iso}, current isolated={c_iso}): "
                        "a personal store has distractors and earlier runs' data, an isolated one has neither")
    b_m = (bp.get("daemon_config") or {}).get("llm_model")
    c_m = (cp.get("daemon_config") or {}).get("llm_model")
    if b_m != c_m:
        warnings.append(f"LLM model differs or unknown ({b_m} vs {c_m})")
    return warnings


def flipped_probes(baseline: Dict[str, Any], current: Dict[str, Any]) -> list:
    """Per-probe pass/fail changes, when both runs recorded details."""
    flips = []
    for suite in ("locomo", "membench"):
        b = {d["id"]: d for d in baseline.get("suites", {}).get(suite, {}).get("details", [])}
        for d in current.get("suites", {}).get(suite, {}).get("details", []):
            if d["id"] in b and b[d["id"]]["passed"] != d["passed"]:
                flips.append((suite, d["id"], b[d["id"]]["passed"], d["passed"], d["reason"]))
    return flips


def print_comparison_table(baseline: Dict[str, Any], current: Dict[str, Any]) -> None:
    """Print markdown comparison table between baseline and current run."""
    print("\n" + "=" * 80)
    print("### HippoCampy Benchmark Comparison Report")
    print("=" * 80 + "\n")
    for w in comparability_warnings(baseline, current):
        print(f"> ⚠️ {w}")
    print()
    print("| Suite / Metric | Baseline Snapshot | Current Evaluation |")
    print("|---|---|---|")
    for suite, key, label, hib in COMPARE_ROWS:
        b = baseline.get("suites", {}).get(suite, {}).get(key)
        c = current.get("suites", {}).get(suite, {}).get(key)
        print(f"| **{label}** | {b if b is not None else 'N/A'} | {format_delta(b, c, hib)} |")
    flips = flipped_probes(baseline, current)
    if flips:
        print("\n#### Probes that changed verdict")
        for suite, pid, was, now, reason in flips:
            print(f"- {suite}/{pid}: {'pass' if was else 'fail'} -> {'pass' if now else 'fail'} ({reason})")
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

        suites = {k: v for k, v in results.get("suites", {}).items() if v.get("valid")}
        if "locomo" in suites:
            loc = suites["locomo"]
            table.add_row("LoCoMo", f"DeprecAcc: {loc.get('deprecation_accuracy')} | EM: {loc.get('exact_match')} | F1: {loc.get('f1')}", f"Acc: {loc.get('accuracy')}", f"{loc.get('avg_latency_ms')} ms")
        if "memory_gym" in suites:
            mg = suites["memory_gym"]
            table.add_row("MemoryGym", f"Efficiency: {mg.get('step_efficiency')} | Steps: {mg.get('total_steps')}", f"Success: {mg.get('success_rate') * 100:.1f}%", f"{mg.get('avg_retrieve_latency_ms')} ms (retrieve)")
        if "membench" in suites:
            mb = suites["membench"]
            table.add_row("MemBench", f"Contradiction: {mb.get('contradiction_score')} | Savings: {mb.get('token_savings_pct')}%", f"Acc: {mb.get('accuracy')}", f"{mb.get('avg_latency_ms')} ms")
        if "arc_bridge" in suites:
            arc = suites["arc_bridge"]
            table.add_row("ARC Bridge", f"Transfer: {arc.get('rule_transfer_rate')} | Disappear: {arc.get('disappeared_entity_recall')}", "-", f"{arc.get('hot_path_latency_ms')} ms")

        console.print(table)
    except Exception:
        # Fallback to plain text
        print("\n=== Benchmark Execution Summary ===")
        for name, data in results.get("suites", {}).items():
            print(f"- {name.upper()}: " + str({k: v for k, v in data.items() if k != "details"}))


def default_results_path(results: Dict[str, Any], real: bool) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = (results["provenance"]["harness"].get("commit") or "nogit")[:8]
    suffix = "" if real else "-mock"
    if results.get("mode") == "smoke":
        suffix += "-smoke"
    return Path(__file__).resolve().parent / "results" / f"{stamp}-{sha}{suffix}.json"


def main():
    parser = argparse.ArgumentParser(description="Run Campy External Benchmarks Harness")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke checks across all suites")
    parser.add_argument("--baseline", action="store_true", help="Record current baseline snapshot")
    parser.add_argument("--compare", type=str, nargs="?", const="baseline_snapshot.json", help="Compare against baseline JSON file")
    parser.add_argument("--out", type=str, default=None,
                        help="Write results here (default with --baseline: results/<utc>-<harness sha>[-smoke|-mock].json)")
    parser.add_argument("--isolated", action="store_true",
                        help="Start a throwaway daemon with its own empty store (CAMPY_HOME) instead of "
                             "using the personal ~/.campy daemon; needs hippocampy with CAMPY_HOME support")
    parser.add_argument("--daemon-python", type=str, default=None,
                        help="With --isolated: python that runs campy.brain_daemon (default: first word of CAMPY_MCP_CMD)")
    parser.add_argument("--keep-store", action="store_true",
                        help="With --isolated: keep the temp CAMPY_HOME for inspection instead of deleting it")
    parser.add_argument("--daemon-ready-timeout", type=float, default=900.0,
                        help="With --isolated: seconds to wait for the daemon socket (cold start loads models)")
    parser.add_argument("--trace-context", action="store_true",
                        help="LoCoMo: also call compile_context per probe and record what it retrieved (extra daemon calls)")
    parser.add_argument("--suite", choices=["all", "locomo", "memory_gym", "membench", "arc"], default="all")
    args = parser.parse_args()

    mcp_cmd = os.environ.get("CAMPY_MCP_CMD")
    print(f"=== Campy Benchmark Harness (B381) ===")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}Z")
    print(f"Mode: {'SMOKE' if args.smoke else 'STANDARD'}")
    print(f"Target MCP Server: {mcp_cmd or 'Standalone / Fallback Mock'}")

    if args.isolated and not mcp_cmd:
        parser.error("--isolated needs CAMPY_MCP_CMD (its python is used to launch the daemon)")
    if mcp_cmd and not args.isolated:
        print("[!] Not isolated: this run reads and writes the personal ~/.campy store, and earlier "
              "runs' data is in it. Use --isolated for a clean, reproducible store.")

    isolated = None
    client_env = None
    if args.isolated:
        repo = provenance.hippocampy_repo_from_cmd(mcp_cmd)
        base_cfg = provenance.base_config_path(repo)
        isolated = IsolatedDaemon(
            python=args.daemon_python or shlex.split(mcp_cmd)[0],
            base_config=base_cfg,
            keep_store=args.keep_store,
            ready_timeout=args.daemon_ready_timeout,
        )
        print(f"[+] Starting isolated daemon (base config: {base_cfg or 'daemon defaults'})...")
        isolated.start()
        client_env = isolated.client_env()
        print(f"    ready in {isolated.ready_seconds}s at {isolated.home}")

    try:
        client = CampyMCPClient(mcp_cmd=mcp_cmd, env=client_env)
    except Exception:
        if isolated:
            isolated.stop()
        raise

    suites_to_run = ["locomo", "memory_gym", "membench", "arc"] if args.suite == "all" else [args.suite]
    executed_suites: Dict[str, Any] = {}

    def run_suite(key: str, label: str, fn) -> None:
        print(f"\n[+] Running {label}...")
        t0 = time.time()
        client.reset_mock_state()  # B438: isolate from any prior suite's mock data
        calls0, fails0 = client.stats["calls"], client.stats["failures"]
        try:
            res = fn(client, smoke=args.smoke, trace_context=args.trace_context)
            res["valid"] = True
        except CampyClientError as e:
            # A suite that lost the daemon mid-run produced no measurement.
            # Record that loudly instead of a score.
            res = {"suite": key, "valid": False, "error": str(e)[:500]}
            print(f"    !! SUITE INVALID: {e}")
        res["client_calls"] = client.stats["calls"] - calls0
        res["client_failures"] = client.stats["failures"] - fails0
        executed_suites[key] = res
        print(f"    Completed in {time.time() - t0:.2f}s "
              f"(calls={res['client_calls']}, failures={res['client_failures']}, valid={res['valid']})")

    try:
        if "locomo" in suites_to_run:
            run_suite("locomo", "LoCoMo Suite (Conversational Deprecation)", run_locomo)
        if "memory_gym" in suites_to_run:
            run_suite("memory_gym", "MemoryGym Suite (2D Spatial/Temporal Persistence)", run_memory_gym)
        if "membench" in suites_to_run:
            run_suite("membench", "MemBench Suite (Persona & Contradiction Arbitration)", run_membench)
        if "arc" in suites_to_run or "arc_bridge" in suites_to_run:
            run_suite("arc_bridge", "ARC Bridge Suite (World Model & Memory Transfer)", run_arc_bridge)
    finally:
        client.close()
        if isolated:
            isolated.stop()

    results: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance.collect(mcp_cmd, args.smoke, sys.argv, isolated),
        "mode": "smoke" if args.smoke else "standard",
        "mcp_configured": bool(mcp_cmd),
        "suites": executed_suites,
        "all_suites_valid": all(v.get("valid", False) for v in executed_suites.values()),
        "canonical_baseline_targets": {
            "retrieval_latency_ms": "<10.0ms (via B375)",
            "llm_generation_latency_s": "<1.0s (via B374)",
            "daemon_idle_rss_mb": "<80MB (via B384)",
            "token_compression_ratio": "50%-70% bulk, 100% bypass on sub-budget (B374)",
            "ask_eval_overall": ">=0.90",
        },
    }

    print_summary_table(results)

    if args.baseline or args.out:
        out_path = Path(args.out) if args.out else default_results_path(results, bool(mcp_cmd))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[+] Recorded results (with per-probe details) to {out_path.resolve()}")

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

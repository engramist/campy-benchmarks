#!/usr/bin/env python3
"""
Campy Benchmarks Unified Runner (B381)
Executes evaluation suites against HippoCampy over MCP transport.
Supports --smoke, --baseline, and --compare flags.
"""

from __future__ import annotations

import argparse
import hashlib
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
import repeats
from baselines import BASELINE_NAMES, QA_SUITES, compare_to_campy, run_all_baselines
from isolation import IsolatedDaemon
from llm_client import BaselineLLM, LLMError
from locomo10.runner import locomo10_options, run_locomo10
from locomo10.scoring import aggregate as locomo10_aggregate, judge_details, JUDGE_TEMPLATE
from mcp_client import CampyMCPClient, CampyClientError
from locomo.runner import run_locomo
from memory_gym.runner import run_memory_gym
from membench.runner import run_membench
from arc_bridge.runner import run_arc_bridge


def format_delta(baseline_val: Any, current_val: Any, higher_is_better: bool = True,
                 noise: Optional[float] = None) -> str:
    """Format comparison delta string. `noise`: the run-to-run range of this
    metric (from --repeat); a change no bigger than that is marked ≈."""
    if baseline_val is None or current_val is None:
        return f"{current_val if current_val is not None else 'N/A'}"
    try:
        b = float(baseline_val)
        c = float(current_val)
        delta = c - b
        pct = (delta / b * 100.0) if b != 0 else 0.0
        sign = "+" if delta > 0 else ""
        icon = "✅" if (delta >= 0 if higher_is_better else delta <= 0) else "⚠️"
        if noise is not None and delta != 0 and abs(delta) <= noise:
            icon = "≈ (within run-to-run range)"
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
    ("locomo10", "judge_accuracy", "LoCoMo-10 Judge Accuracy (cat 1-4)", True),
    ("locomo10", "f1", "LoCoMo-10 F1 (cat 1-4)", True),
    ("locomo10", "adversarial_abstention", "LoCoMo-10 Adversarial Abstention", True),
    ("locomo10", "evidence_recall", "LoCoMo-10 Evidence Recall", True),
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
    b_j = ((baseline.get("locomo10_judge") or {}).get("llm") or {}).get("model")
    c_j = ((current.get("locomo10_judge") or {}).get("llm") or {}).get("model")
    if (baseline.get("locomo10_judge") or current.get("locomo10_judge")) and b_j != c_j:
        warnings.append(f"LoCoMo-10 judge model differs ({b_j} vs {c_j}): judge_accuracy is not comparable")
    b_l10 = (baseline.get("suites", {}).get("locomo10") or {}).get("dataset", {}).get("options")
    c_l10 = (current.get("suites", {}).get("locomo10") or {}).get("dataset", {}).get("options")
    if b_l10 and c_l10 and b_l10 != c_l10:
        warnings.append(f"LoCoMo-10 subset differs ({b_l10} vs {c_l10})")
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
        ranges = [r for r in (repeats.metric_range(baseline, suite, key),
                              repeats.metric_range(current, suite, key)) if r is not None]
        noise = max(ranges) if ranges else None
        print(f"| **{label}** | {b if b is not None else 'N/A'} | {format_delta(b, c, hib, noise)} |")
    if not (baseline.get("repeat") or current.get("repeat")):
        print("\n_Single runs: LLM answer variance alone can move a metric. Use --repeat N to measure it._")
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
        if "locomo10" in suites:
            l10 = suites["locomo10"]
            table.add_row("LoCoMo-10", f"F1: {l10.get('f1')} | Adv. abstain: {l10.get('adversarial_abstention')} | Evidence recall: {l10.get('evidence_recall')}", f"Judge: {l10.get('judge_accuracy')}", f"{l10.get('avg_latency_ms')} ms")
        if "arc_bridge" in suites:
            arc = suites["arc_bridge"]
            table.add_row("ARC Bridge", f"Transfer: {arc.get('rule_transfer_rate')} | Disappear: {arc.get('disappeared_entity_recall')}", "-", f"{arc.get('hot_path_latency_ms')} ms")

        console.print(table)
    except Exception:
        # Fallback to plain text
        print("\n=== Benchmark Execution Summary ===")
        for name, data in results.get("suites", {}).items():
            print(f"- {name.upper()}: " + str({k: v for k, v in data.items() if k != "details"}))


def resolve_baseline_llm(args, mcp_cmd, isolated, overrides=None):
    """The baselines' LLM: the same base [llm] section Campy's `ask` uses --
    the isolated daemon's written config when there is one, else the best-
    effort config provenance finds -- with any --baseline-* overrides (or
    `overrides`, a (provider, model, base_url) tuple, for the judge)."""
    import tomllib

    provider, model, base_url = overrides or (args.baseline_provider, args.baseline_model, args.baseline_base_url)
    cfg: Dict[str, Any] = {}
    source = "none (set --baseline-provider/--baseline-model)"
    if isolated is not None:
        cfg, source = isolated.config, "isolated daemon config"
    else:
        path = provenance.base_config_path(provenance.hippocampy_repo_from_cmd(mcp_cmd))
        if path is not None:
            cfg, source = tomllib.loads(path.read_text()), str(path)
    if provider or model or base_url:
        source += " + CLI overrides"
    llm = BaselineLLM.from_config(cfg.get("llm", {}), provider=provider, model=model, base_url=base_url)
    return llm, cfg.get("embeddings", {}).get("model"), source


def finalize_locomo10(results: Dict[str, Any], args, isolated) -> None:
    """Judge every LoCoMo-10 answer -- Campy's and each baseline's -- with the
    same judge in one pass, then recompute their metrics."""
    targets = []
    campy = results.get("suites", {}).get("locomo10")
    if campy and campy.get("valid"):
        targets.append(("campy", campy))
    for name, per_suite in (results.get("baselines") or {}).items():
        if name != "config" and (per_suite.get("locomo10") or {}).get("valid"):
            targets.append((name, per_suite["locomo10"]))
    if not targets:
        return
    info: Dict[str, Any] = {"enabled": args.judge != "none", "template_sha256": hashlib.sha256(JUDGE_TEMPLATE.encode()).hexdigest()[:16]}
    if args.judge != "none":
        judge_llm, _, source = resolve_baseline_llm(
            args, os.environ.get("CAMPY_MCP_CMD"), isolated,
            # unset --judge-* fall back to --baseline-*, then to the config
            overrides=(args.judge_provider or args.baseline_provider,
                       args.judge_model or args.baseline_model,
                       args.judge_base_url or args.baseline_base_url))
        info.update({"llm": judge_llm.describe(), "source": source})
        print(f"\n[+] LoCoMo-10 judge: {judge_llm.describe()['model']} over {[n for n, _ in targets]}")
        try:
            for name, res in targets:
                n = judge_details(res["details"], judge_llm)
                print(f"    judged {name}: {n} calls")
        except LLMError as e:
            info["error"] = str(e)[:500]
            print(f"    !! judge failed ({e}); LoCoMo-10 judge_accuracy stays null, F1 still reported")
    for _, res in targets:
        res.update(locomo10_aggregate(res["details"]))
    results["locomo10_judge"] = info


def print_baseline_table(results: Dict[str, Any]) -> None:
    """Campy next to each baseline, per QA suite, plus the per-probe cross-tab."""
    bl = results.get("baselines", {})
    print("\n### Campy vs baselines (same LLM, same judge)")
    print("| Suite | Metric | Campy | " + " | ".join(BASELINE_NAMES) + " |")
    print("|---|---|---|" + "---|" * len(BASELINE_NAMES))
    for suite in QA_SUITES:
        campy = results.get("suites", {}).get(suite, {})
        if suite == "locomo10":
            metrics = ("judge_accuracy", "f1", "adversarial_abstention", "evidence_recall", "avg_prompt_tokens_est")
        else:
            flag = "deprecation_accuracy" if suite == "locomo" else "contradiction_score"
            metrics = ("accuracy", flag, "avg_prompt_tokens_est")
        for metric in metrics:
            row = [campy.get(metric, "-") if campy.get("valid") else "-"]
            for name in BASELINE_NAMES:
                b = bl.get(name, {}).get(suite)
                row.append("-" if not b else (b.get(metric, "-") if b.get("valid") else "INVALID"))
            if any(v != "-" for v in row):
                print(f"| {suite} | {metric} | " + " | ".join(str(v) for v in row) + " |")
    for suite, xt in (results.get("campy_vs_baselines") or {}).items():
        if xt.get("guessable_without_memory"):
            print(f"- {suite}: pass with NO memory (not testing memory): {xt['guessable_without_memory']}")
        for name in ("naive_rag", "full_context"):
            ids = (xt.get(name) or {}).get("baseline_passes_campy_fails")
            if ids:
                print(f"- {suite}: {name} passes but Campy fails: {ids}")


def default_results_path(results: Dict[str, Any], real: bool, baselines_only: bool = False) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = (results["provenance"]["harness"].get("commit") or "nogit")[:8]
    suffix = "-baselines" if baselines_only else ("" if real else "-mock")
    if results.get("mode") == "smoke":
        suffix += "-smoke"
    return Path(__file__).resolve().parent / "results" / f"{stamp}-{sha}{suffix}.json"


def main():
    parser = argparse.ArgumentParser(description="Run Campy External Benchmarks Harness")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke checks across all suites")
    parser.add_argument("--baseline", action="store_true",
                        help="Save this run's results file (a 'baseline snapshot'; unrelated to --baselines)")
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
    parser.add_argument("--repeat", type=int, default=1, metavar="N",
                        help="Run the suites N times, each on a fresh isolated store, and report the mean, "
                             "the min/max/stdev, and the probes whose verdict changed between runs "
                             "(needs --isolated with a real daemon)")
    parser.add_argument("--trace-context", action="store_true",
                        help="LoCoMo: also call compile_context per probe and record what it retrieved (extra daemon calls)")
    parser.add_argument("--suite", choices=["all", "locomo", "memory_gym", "membench", "arc", "locomo10"], default="all",
                        help="'all' = the four fixture suites. locomo10 = the published LoCoMo dataset "
                             "(hours on a local model; not in 'all')")
    parser.add_argument("--locomo10-conversations", type=int, default=None,
                        help="LoCoMo-10: first N of the 10 conversations (smoke default 1)")
    parser.add_argument("--locomo10-max-questions", type=int, default=None,
                        help="LoCoMo-10: at most N questions per conversation (smoke default 25)")
    parser.add_argument("--locomo10-categories", type=lambda s: [int(x) for x in s.split(",")], default=None,
                        help="LoCoMo-10: only these categories, e.g. 1,2,3,4 (5 = adversarial)")
    parser.add_argument("--judge", choices=["auto", "none"], default="auto",
                        help="LoCoMo-10 LLM judge: auto = run it (default LLM = the baselines' LLM), none = F1 only")
    parser.add_argument("--judge-provider", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default=None,
                        help="A stronger judge than the answering model is recommended")
    parser.add_argument("--judge-base-url", type=str, default=None)
    parser.add_argument("--baselines", type=str, default=None,
                        help="Reference systems to score on the QA suites (LoCoMo, MemBench) with the same LLM "
                             "and judge: 'all' or a comma list of " + ", ".join(BASELINE_NAMES))
    parser.add_argument("--baselines-only", action="store_true",
                        help="Run only --baselines (no daemon needed)")
    parser.add_argument("--baseline-provider", type=str, default=None, help="Override the baselines' [llm].provider")
    parser.add_argument("--baseline-model", type=str, default=None, help="Override the baselines' [llm].model")
    parser.add_argument("--baseline-base-url", type=str, default=None, help="Override the baselines' [llm].base_url")
    parser.add_argument("--rag-k", type=int, default=5, help="naive_rag: turns retrieved per question")
    parser.add_argument("--rag-retriever", choices=["auto", "embedding", "bm25"], default="auto",
                        help="naive_rag retriever: embedding (fastembed, Campy's embedder), bm25, or auto "
                             "(embedding, else bm25 with a warning)")
    args = parser.parse_args()

    baseline_names = []
    if args.baselines:
        baseline_names = list(BASELINE_NAMES) if args.baselines == "all" else \
            [b.strip() for b in args.baselines.split(",") if b.strip()]
        unknown = [b for b in baseline_names if b not in BASELINE_NAMES]
        if unknown:
            parser.error(f"unknown baselines {unknown}; choose from {list(BASELINE_NAMES)}")
    if args.baselines_only and not baseline_names:
        parser.error("--baselines-only needs --baselines")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.repeat > 1 and (baseline_names or args.suite == "locomo10"):
        parser.error("--repeat > 1 is not supported with --baselines or --suite locomo10 yet")

    mcp_cmd = os.environ.get("CAMPY_MCP_CMD")
    print(f"=== Campy Benchmark Harness (B381) ===")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}Z")
    print(f"Mode: {'SMOKE' if args.smoke else 'STANDARD'}")
    print(f"Target MCP Server: {mcp_cmd or 'Standalone / Fallback Mock'}")

    if args.baselines_only:
        mcp_cmd = None  # no daemon: baselines need only the LLM
    if args.isolated and not mcp_cmd:
        parser.error("--isolated needs CAMPY_MCP_CMD (its python is used to launch the daemon)")
    if args.repeat > 1 and mcp_cmd and not args.isolated:
        # Each repeat needs an empty store: on a shared one, run k reads runs
        # 1..k-1's data (20 more near-identical MemoryGym notes per run).
        parser.error("--repeat > 1 with a real daemon needs --isolated (a fresh store per run)")
    if mcp_cmd and not args.isolated and not args.baselines_only:
        print("[!] Not isolated: this run reads and writes the personal ~/.campy store, and earlier "
              "runs' data is in it. Use --isolated for a clean, reproducible store.")

    suites_to_run = ["locomo", "memory_gym", "membench", "arc"] if args.suite == "all" else [args.suite]
    campy_suites = [] if args.baselines_only else suites_to_run
    l10_opts = locomo10_options(args, args.smoke)

    def run_once(run_no: int):
        """Start the daemon (fresh isolated store if --isolated), run the
        Campy suites, stop it. Returns (executed suites, the IsolatedDaemon)."""
        isolated = None
        client_env = None
        if args.repeat > 1:
            print(f"\n=== Run {run_no} of {args.repeat} ===")
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
            if "locomo" in campy_suites:
                run_suite("locomo", "LoCoMo Suite (Conversational Deprecation)", run_locomo)
            if "memory_gym" in campy_suites:
                run_suite("memory_gym", "MemoryGym Suite (2D Spatial/Temporal Persistence)", run_memory_gym)
            if "membench" in campy_suites:
                run_suite("membench", "MemBench Suite (Persona & Contradiction Arbitration)", run_membench)
            if "arc" in campy_suites:
                run_suite("arc_bridge", "ARC Bridge Suite (World Model & Memory Transfer)", run_arc_bridge)
            if "locomo10" in campy_suites:
                run_suite("locomo10", f"LoCoMo-10 (published dataset; {l10_opts})",
                          lambda c, smoke, trace_context: run_locomo10(c, smoke, True, l10_opts))
        finally:
            client.close()
            if isolated:
                isolated.stop()
        return executed_suites, isolated

    runs = [run_once(i + 1) for i in range(args.repeat)]
    isolated = runs[-1][1]
    repeat_info = None
    if args.repeat > 1:
        agg = repeats.aggregate_runs([suites for suites, _ in runs])
        executed_suites, repeat_info = agg["suites"], agg["repeat"]
    else:
        executed_suites = runs[0][0]

    results: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance.collect(mcp_cmd, args.smoke, sys.argv, isolated),
        "mode": "smoke" if args.smoke else "standard",
        "mcp_configured": bool(mcp_cmd),
        "suites": executed_suites,
        "all_suites_valid": all(v.get("valid", False) for v in executed_suites.values()),
        **({"repeat": repeat_info} if repeat_info else {}),
        "canonical_baseline_targets": {
            "retrieval_latency_ms": "<10.0ms (via B375)",
            "llm_generation_latency_s": "<1.0s (via B374)",
            "daemon_idle_rss_mb": "<80MB (via B384)",
            "token_compression_ratio": "50%-70% bulk, 100% bypass on sub-budget (B374)",
            "ask_eval_overall": ">=0.90",
        },
    }

    if repeat_info:
        results["provenance"]["store"]["repeats"] = args.repeat
        results["provenance"]["store"]["note"] = "a fresh isolated store per run; describe() is the last run's"
    if args.baselines_only:
        results["provenance"]["daemon_config"] = {"source": "not used (--baselines-only)"}
        results["provenance"]["store"] = {"note": "no daemon (--baselines-only)"}
    if baseline_names:
        qa_suites = [s for s in QA_SUITES if s in suites_to_run]
        print(f"\n[+] Running baselines {baseline_names} on {qa_suites} (after the Campy suites, so they "
              "don't compete with the daemon for the LLM)...")
        llm, embed_model, llm_source = resolve_baseline_llm(args, os.environ.get("CAMPY_MCP_CMD"), isolated)
        print(f"    LLM: {llm.describe()} (from {llm_source})")
        bl = run_all_baselines(baseline_names, qa_suites, llm, args.smoke,
                               retriever_kind=args.rag_retriever, embed_model=embed_model, k=args.rag_k,
                               suite_opts={"locomo10": l10_opts})
        bl["config"]["llm_source"] = llm_source
        results["baselines"] = bl
    if "locomo10" in suites_to_run:
        finalize_locomo10(results, args, isolated)
    if baseline_names and executed_suites:
        results["campy_vs_baselines"] = compare_to_campy(executed_suites, results["baselines"])

    print_summary_table(results)
    repeats.print_spread(results, COMPARE_ROWS)
    if baseline_names:
        print_baseline_table(results)

    if args.baseline or args.out:
        out_path = Path(args.out) if args.out else default_results_path(
            results, bool(mcp_cmd), baselines_only=args.baselines_only)
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

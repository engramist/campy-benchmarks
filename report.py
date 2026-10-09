#!/usr/bin/env python3
"""
campy-benchmarks / report.py
Builds RESULTS.md from result files that `run_all.py` wrote, so published
numbers live in the repo next to the exact configuration that produced them.

    python report.py > RESULTS.md          # every file in results/published/
    python report.py a.json b.json         # or these

For each suite, the headline table shows the newest run that has it; the
run history lists all of them. Each row names the hippocampy and harness
commits, the answering LLM, the judge, the subset and the repeat count.
A result counts only when it ran on a real daemon, on an isolated store, from
clean commits of both repos; anything else is listed with the reason it
doesn't count, and never reaches the headline table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# suite key -> (title, kind, [(metric key, label)], note)
SUITES: Dict[str, Tuple[str, str, List[Tuple[str, str]], str]] = {
    "locomo10": ("LoCoMo-10", "published dataset", [
        ("judge_accuracy", "judge accuracy"), ("f1", "F1"),
        ("adversarial_abstention", "adversarial abstention"), ("evidence_recall", "evidence recall")],
        "Maharana et al., ACL 2024. Judge accuracy is this harness's own prompt (categories 1-4); "
        "compare across runs with the same judge model, not with published numbers."),
    "longmemeval": ("LongMemEval", "published dataset", [
        ("accuracy", "accuracy"), ("task_averaged_accuracy", "task-averaged accuracy"),
        ("evidence_recall", "evidence recall")],
        "Wu et al., ICLR 2025. Official per-type judge prompts; the paper's judge is GPT-4o. "
        "The `oracle` variant holds only the evidence sessions, `s` about 40 sessions per question."),
    "dmr": ("DMR (MSC-Self-Instruct)", "published dataset", [
        ("judge_accuracy", "judge accuracy"), ("f1", "F1"), ("evidence_recall", "evidence recall")],
        "Deep Memory Retrieval (Packer et al., MemGPT, 2023): 500 questions, each over 4 earlier "
        "Multi-Session Chat sessions. Judge: LoCoMo-10's prompt. Evidence labels are derived (turns "
        "containing the answer), so recall covers only the questions where the answer is quoted. "
        "Published scores are near the full-context ceiling (MemGPT 93.4%, Zep 94.8%, GPT-4 Turbo "
        "full context 94.4%), so DMR separates systems little."),
    "locomo": ("LoCoMo fixture", "hand-written fixture", [
        ("judge_accuracy", "judge accuracy"), ("judge_deprecation_accuracy", "judge deprecation accuracy"),
        ("accuracy", "lexical accuracy")],
        "25 hand-written scenarios, 28 probes. Borrows LoCoMo's name, not its data."),
    "membench": ("MemBench fixture", "hand-written fixture", [
        ("judge_accuracy", "judge accuracy"), ("judge_contradiction_score", "judge contradiction score"),
        ("accuracy", "lexical accuracy")],
        "Hand-written personas with a fact that changes. Not MSC data, not the MemBench benchmark."),
    "memory_gym": ("MemoryGym fixture", "simulated", [
        ("success_rate", "success rate"), ("step_efficiency", "step efficiency")],
        "The simulated environment, a verbatim-recall smoke test; not the MemoryGym benchmark."),
    "arc_bridge": ("ARC bridge", "integration check", [
        ("rule_transfer_rate", "rule transfer"), ("disappeared_entity_recall", "disappeared-entity recall"),
        ("hot_path_latency_ms", "hot path ms")],
        "Memory-transfer checks from the sibling ARC_AGI repo."),
}
ORDER = list(SUITES)


def fmt(v: Any) -> str:
    if v is None:
        return "–"
    if isinstance(v, float) and v.is_integer() and abs(v) >= 1:
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


def short(sha: Optional[str]) -> str:
    return (sha or "?")[:7]


def subset(key: str, s: Dict[str, Any]) -> str:
    ds = s.get("dataset") if isinstance(s.get("dataset"), dict) else {}
    # hippocampy B472: speaker and date sent as fields, not in the text
    tag = ", turn fields" if ds.get("turn_metadata") == "fields" else ""
    return _subset(key, s, ds) + tag


def _subset(key: str, s: Dict[str, Any], ds: Dict[str, Any]) -> str:
    if key == "locomo10":
        return f"{len(ds.get('conversations') or [])} conv, {s.get('questions')} q"
    if key == "longmemeval":
        return f"{ds.get('variant')}, {s.get('questions')} q"
    if key == "dmr":
        return f"{s.get('questions')} q"
    if key in ("locomo", "membench"):
        return f"{fmt(s.get('probes') or s.get('total_probes'))} probes"
    if key == "memory_gym":
        return f"{fmt(s.get('episodes'))} episodes"
    return ""


def judge_model(r: Dict[str, Any], key: str) -> str:
    j = r.get({"locomo10": "locomo10_judge", "longmemeval": "longmemeval_judge",
               "dmr": "dmr_judge"}.get(key, "qa_judge")) or {}
    if key in ("memory_gym", "arc_bridge"):
        return "n/a"
    return ((j.get("llm") or {}).get("model") if j.get("enabled") else None) or "none"


def countable(r: Dict[str, Any]) -> Optional[str]:
    """None if the run counts, else why not."""
    if not r.get("mcp_configured"):
        return "mock run (no daemon)"
    if not ((r.get("provenance") or {}).get("store") or {}).get("isolated"):
        return "not on an isolated store"
    prov = r.get("provenance") or {}
    for repo in ("harness", "hippocampy"):
        if (prov.get(repo) or {}).get("dirty"):
            return f"uncommitted {repo} changes"
    return None


def variant(key: str, s: Dict[str, Any]) -> Any:
    """What gets its own headline row: a LongMemEval variant, and (B472) a
    run that sent speaker and date as fields rather than in the text."""
    ds = s.get("dataset") if isinstance(s.get("dataset"), dict) else {}
    return (ds.get("variant") if key == "longmemeval" else None, ds.get("turn_metadata") or "text")


def commit_of(r: Dict[str, Any]) -> Optional[str]:
    return ((r.get("provenance") or {}).get("hippocampy") or {}).get("commit")


def metric_cell(label: str, values: List[float]) -> str:
    """One run: the value. Several runs of the same code: mean, run count, range."""
    if len(values) == 1:
        return f"{label}: **{fmt(values[0])}**"
    mean = sum(values) / len(values)
    return f"{label}: **{fmt(mean)}** (mean of {len(values)}, {fmt(min(values))}–{fmt(max(values))})"


def headline_row(w, key: str, r: Dict[str, Any], same: Optional[List[Dict[str, Any]]] = None) -> None:
    s = r["suites"][key]
    title, kind, metrics, _ = SUITES[key]
    prov = r.get("provenance") or {}
    runs = same or [r]
    cells = []
    for m, label in metrics:
        vals = [x["suites"][key].get(m) for x in runs if x["suites"][key].get(m) is not None]
        if vals:
            cells.append(metric_cell(label, vals))
    mets = "<br>".join(cells)
    bl = []
    for name, data in (r.get("baselines") or {}).items():
        if name != "config" and isinstance(data, dict) and isinstance(data.get(key), dict):
            m0 = metrics[0][0]
            if data[key].get(m0) is not None:
                bl.append(f"{name}: {fmt(data[key][m0])}")
    llm = (prov.get("daemon_config") or {}).get("llm_model") or "?"
    w(f"| {title} | {kind} | {subset(key, s)} | {mets} | {'<br>'.join(bl) or '–'} | "
      f"{llm} / {judge_model(r, key)} | `{short((prov.get('hippocampy') or {}).get('commit'))}` | "
      + " ".join(f"[{Path(x['_file']).name}]({x['_file']})" for x in runs) + " |")


def main(paths: List[str]) -> int:
    runs = []
    for p in paths:
        r = json.loads(Path(p).read_text())
        r["_file"] = p
        runs.append(r)
    runs.sort(key=lambda r: r.get("timestamp") or "")
    good = [r for r in runs if countable(r) is None]
    out: List[str] = []
    w = out.append
    w("# Campy benchmark results\n")
    w("Generated by `python report.py > RESULTS.md` from the result files in "
      "`results/published/`. Every number below links back to its file, which holds the per-question "
      "answers and the full configuration. Single runs unless the repeat column says otherwise.\n")
    w("Published datasets are what to quote. The fixtures are regression tests for this harness: "
      "they are small and hand-written, and a high score on them is not a benchmark result.\n")

    w("## Headline (newest run per suite)\n")
    w("| suite | kind | subset | metrics | baselines on the first metric (same LLM, same judge) | LLM / judge | hippocampy | run |")
    w("|---|---|---|---|---|---|---|---|")
    if not good:
        w("| _no counted runs yet_ | | | | | | | |")
    for key in ORDER:
        have = [r for r in good if key in (r.get("suites") or {}) and r["suites"][key].get("valid")]
        if not have:
            continue
        # one headline row per dataset variant (LongMemEval oracle vs s are different tests)
        newest: Dict[Any, Dict[str, Any]] = {}
        for r in have:
            newest[variant(key, r["suites"][key])] = r
        for r in newest.values():
            # repeat runs of the same code on the same subset: report their mean
            same = [x for x in have if variant(key, x["suites"][key]) == variant(key, r["suites"][key])
                    and commit_of(x) == commit_of(r) and subset(key, x["suites"][key]) == subset(key, r["suites"][key])]
            headline_row(w, key, r, same)
    w("")

    for key in ORDER:
        rows = [r for r in runs if key in (r.get("suites") or {})]
        if not rows:
            continue
        title, kind, metrics, note = SUITES[key]
        w(f"## {title}\n")
        w(f"{note}\n")
        w("| date | subset | " + " | ".join(label for _, label in metrics)
          + " | repeat | LLM / judge | hippocampy | harness | file |")
        w("|---|---|" + "---|" * len(metrics) + "---|---|---|---|---|")
        for r in rows:
            s = r["suites"][key]
            prov = r.get("provenance") or {}
            harness = prov.get("harness") or {}
            why = countable(r)
            rep = (r.get("repeat") or {}).get("n") or 1
            w(f"| {(r.get('timestamp') or '')[:10]}{' (not counted: ' + why + ')' if why else ''} | "
              f"{subset(key, s)} | " + " | ".join(fmt(s.get(m)) for m, _ in metrics)
              + f" | {rep} | {(prov.get('daemon_config') or {}).get('llm_model') or '?'} / {judge_model(r, key)} | "
              f"`{short((prov.get('hippocampy') or {}).get('commit'))}` | "
              f"`{short(harness.get('commit'))}` | "
              f"[{Path(r['_file']).name}]({r['_file']}) |")
        by_cat = None
        newest = [r for r in rows if countable(r) is None]
        if newest:
            by_cat = newest[-1]["suites"][key].get("by_category")
        if isinstance(by_cat, dict) and by_cat:
            w("\nNewest counted run, by category:\n")
            cols = sorted({c for v in by_cat.values() if isinstance(v, dict) for c in v})
            w("| category | " + " | ".join(cols) + " |")
            w("|---|" + "---|" * len(cols))
            for cat, v in by_cat.items():
                if isinstance(v, dict):
                    w(f"| {cat} | " + " | ".join(fmt(v.get(c)) for c in cols) + " |")
        w("")
    sys.stdout.write("\n".join(out))
    return 0


if __name__ == "__main__":
    args = sys.argv[1:] or sorted(str(p) for p in (Path(__file__).resolve().parent / "results" / "published").glob("*.json"))
    sys.exit(main([str(Path(a).resolve().relative_to(Path.cwd())) if Path(a).resolve().is_relative_to(Path.cwd()) else a
                   for a in args]))

#!/usr/bin/env python3
"""
campy-benchmarks / rejudge.py
Re-judge the answers already in a run_all.py result file -- no daemon, no
re-answering -- with the current judge prompt and, optionally, vote
consensus. Judging is offline, so a judge change can be applied to old runs
and the effect measured:

    python3 rejudge.py results/r24b-dmr-q50-fields.json --judge-model gemma4:26b \\
        --judge-votes 3 --out results/r24b-dmr-q50-fields-rejudged.json

Covers the `dmr` suite (persona rule on) and the `locomo10` suite (categories
1-4; adversarial abstention is lexical and untouched), for Campy and for any
baselines in the file. Records whose run failed (`error`) and non-answers are
left as they were. The output is the input with new verdicts, recomputed
aggregates and a top-level `rejudge` block: the judge config, the previous
judge config if the file had one, and per-suite flips (ids, Y->N and N->Y).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dmr.scoring import aggregate as dmr_aggregate, judge_details as dmr_judge_details
from locomo10.scoring import (JUDGE_TEMPLATE, PERSONA_RULE, aggregate as locomo10_aggregate,
                              judge_details as locomo10_judge_details)

SUITES = {
    "dmr": (dmr_judge_details, dmr_aggregate),
    "locomo10": (locomo10_judge_details, locomo10_aggregate),
}


def _targets(data: Dict[str, Any]) -> List[Tuple[str, str, Dict[str, Any]]]:
    """(label, suite, result dict) for every judgeable result in the file."""
    out = []
    for suite in SUITES:
        res = (data.get("suites") or {}).get(suite)
        if res and res.get("details"):
            out.append((suite, suite, res))
        for name, per_suite in (data.get("baselines") or {}).items():
            r = per_suite.get(suite) if name != "config" and isinstance(per_suite, dict) else None
            if r and r.get("details"):
                out.append((f"{suite}/{name}", suite, r))
    return out


def rejudge(data: Dict[str, Any], judge_llm, votes: int = 1, log=print) -> Dict[str, Any]:
    """Re-judge `data` in place; returns the `rejudge` report (also stored in data)."""
    report: Dict[str, Any] = {"votes": votes, "judge": judge_llm.describe() if hasattr(judge_llm, "describe") else {},
                              "template_sha256": hashlib.sha256(JUDGE_TEMPLATE.encode()).hexdigest()[:16],
                              "persona_rule_suites": ["dmr"], "persona_rule": PERSONA_RULE,
                              "previous": {k: data.get(k) for k in ("dmr_judge", "locomo10_judge") if data.get(k)},
                              "targets": {}}
    for label, suite, res in _targets(data):
        judge_fn, agg_fn = SUITES[suite]
        todo = [d for d in res["details"]
                if not d.get("error") and d.get("category") != 5 and d.get("judge") is not None]
        before = {d["id"]: bool(d["judge"]) for d in todo}
        for d in todo:
            d["judge_previous"] = d["judge"]
            for k in ("judge", "judge_votes", "judge_tiebreak", "judge_raw"):
                d.pop(k, None)
        calls = judge_fn(todo, judge_llm, votes=votes)
        old_acc = res.get("judge_accuracy")
        res.update(agg_fn(res["details"]))
        y_n = sorted(i for d in todo for i in [d["id"]] if before[i] and not d["judge"])
        n_y = sorted(i for d in todo for i in [d["id"]] if not before[i] and d["judge"])
        tie = sorted(d["id"] for d in todo if d.get("judge_tiebreak"))
        report["targets"][label] = {
            "judged": len(todo), "calls": calls, "changed": len(y_n) + len(n_y),
            "correct_to_wrong": y_n, "wrong_to_correct": n_y, "tiebreaks": tie,
            "judge_accuracy_before": old_acc, "judge_accuracy_after": res.get("judge_accuracy"),
        }
        log(f"{label}: {len(todo)} verdicts, {len(y_n) + len(n_y)} changed "
            f"(CORRECT->WRONG {len(y_n)}, WRONG->CORRECT {len(n_y)}), {len(tie)} tiebreaks; "
            f"judge_accuracy {old_acc} -> {res.get('judge_accuracy')}")
        for ids, arrow in ((y_n, "CORRECT->WRONG"), (n_y, "WRONG->CORRECT")):
            if ids:
                log(f"    {arrow}: {', '.join(ids)}")
    data["rejudge"] = report
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", type=Path, help="a run_all.py result file")
    ap.add_argument("--judge-provider", default="ollama")
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--judge-votes", type=int, choices=[1, 3], default=1,
                    help="3 = judge twice, a third call breaks a disagreement (default 1)")
    ap.add_argument("--out", type=Path, required=True, help="where to write the re-judged result file")
    args = ap.parse_args()
    if args.out.resolve() == args.results.resolve():
        ap.error("--out must differ from the input: the original verdicts are the baseline")

    from llm_client import BaselineLLM
    data = json.loads(args.results.read_text())
    judge = BaselineLLM(args.judge_provider, args.judge_model, args.judge_base_url)
    report = rejudge(data, judge, args.judge_votes)
    report["source"] = str(args.results)
    if not report["targets"]:
        print("no dmr or locomo10 details in this file; nothing re-judged")
        return 1
    args.out.write_text(json.dumps(data, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

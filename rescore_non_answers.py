#!/usr/bin/env python3
"""
campy-benchmarks / rescore_non_answers.py
B539: find the answers an existing run auto-failed as `non_answer` under the
legacy rule (any abstention phrase anywhere) that the default `judge` policy
would send to the judge, and optionally re-judge just those.

    # list only (no judge, no model, no GPU)
    python3 rescore_non_answers.py results/r98.json

    # re-judge them with the judge model/votes the run used; the original file
    # is never touched (frozen figures are never overwritten)
    python3 rescore_non_answers.py results/r98.json --judge

The --judge output is `<input stem>-nonanswer-judge.json` next to the input
(or --out): the input with the affected verdicts replaced, aggregates
recomputed, the old verdict kept as `judge_previous`, and a top-level
`non_answer_rescore` block (policy, judge config, affected ids, flips).
Covers `locomo10` and `dmr` (Campy and baselines), categories 1-4; adversarial
(category 5) abstention is lexical and untouched. `--judge-model/-provider/
-base-url/-votes` override what the file recorded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import scoring
from rejudge import SUITES, _targets


def affected(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Details auto-failed as non_answer whose answer is not a pure abstention."""
    return [d for d in res.get("details") or []
            if d.get("reason") == "non_answer" and d.get("category") != 5
            and d.get("judge") is False and d.get("answer", "").strip()
            and not scoring.is_pure_abstention(d["answer"])]


def _recorded_judge(data: Dict[str, Any], suite: str) -> Dict[str, Any]:
    return data.get(f"{suite}_judge") or {}


def _short(text: str, n: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n - 1] + "..."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", type=Path, help="a run_all.py result file")
    ap.add_argument("--judge", action="store_true", help="re-judge the affected answers (needs the judge model)")
    ap.add_argument("--judge-provider", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--judge-votes", type=int, choices=[1, 3], default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--full", action="store_true", help="print whole answers, not a 160-char summary")
    args = ap.parse_args()

    data = json.loads(args.results.read_text())
    found: List[Tuple[str, str, Dict[str, Any], List[Dict[str, Any]]]] = []
    for label, suite, res in _targets(data):
        found.append((label, suite, res, affected(res)))
    print(f"{args.results.name}: legacy-auto-failed answers the `judge` policy would send to the judge")
    total = 0
    for label, suite, res, rows in found:
        auto = sum(1 for d in res["details"] if d.get("reason") == "non_answer" and d.get("category") != 5)
        print(f"  {label}: {len(rows)} of {auto} non_answer auto-fails")
        total += len(rows)
        for d in rows:
            print(f"    {d['id']}: {d['answer'] if args.full else _short(d['answer'])}")
    if not args.judge:
        print(f"total {total} (list only; add --judge to re-judge them)")
        return 0
    if not total:
        print("nothing to re-judge")
        return 0

    from llm_client import BaselineLLM
    rep: Dict[str, Any] = {"policy": "judge", "source": str(args.results), "targets": {}}
    out_path = args.out or args.results.with_name(args.results.stem + "-nonanswer-judge.json")
    if out_path.resolve() == args.results.resolve():
        ap.error("--out must differ from the input")
    llms: Dict[str, Any] = {}
    for label, suite, res, rows in found:
        if not rows:
            continue
        rec = _recorded_judge(data, suite)
        llm_cfg = rec.get("llm") or {}
        provider = args.judge_provider or llm_cfg.get("provider")
        model = args.judge_model or llm_cfg.get("model")
        if not (provider and model):
            ap.error(f"{suite}: the file records no judge model; pass --judge-provider/--judge-model")
        votes = args.judge_votes or rec.get("votes") or 1
        key = (provider, model, args.judge_base_url or llm_cfg.get("base_url"))
        if key not in llms:
            llms[key] = BaselineLLM(*key)
        judge_fn, agg_fn = SUITES[suite]
        for d in rows:
            d["judge_previous"] = d["judge"]
            for k in ("judge", "judge_votes", "judge_tiebreak", "judge_raw", "reason"):
                d.pop(k, None)
        scoring.set_non_answer_policy("judge")
        calls = judge_fn(rows, llms[key], votes=votes)
        old = res.get("judge_accuracy")
        res.update(agg_fn(res["details"]))
        gained = sorted(d["id"] for d in rows if d["judge"])
        rep["targets"][label] = {"affected": [d["id"] for d in rows], "calls": calls, "votes": votes,
                                 "judge": llms[key].describe(), "wrong_to_correct": gained,
                                 "judge_accuracy_before": old, "judge_accuracy_after": res.get("judge_accuracy")}
        print(f"{label}: {len(rows)} re-judged, {len(gained)} now CORRECT; "
              f"judge_accuracy {old} -> {res.get('judge_accuracy')}")
    data["non_answer_rescore"] = rep
    out_path.write_text(json.dumps(data, indent=1))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

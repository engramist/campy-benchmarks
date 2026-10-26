#!/usr/bin/env python3
"""
campy-benchmarks / eval_gate.py   (e2e plan M0.4: the T0 gate table)

Replays the same kept stores through two hippocampy trees -- main and a
branch -- with the `asis` bundle, judges both sets of answers with the same
judge, and prints the gate table the plan asks for (C.5):

  | suite (split) | main | branch | Δ | gained | lost | p | verdict |

gained = failed on main, passes on the branch; lost = the reverse. Each store
is replayed on a copy (as diag_answer_replay does), so the kept stores are
never written. Run with no daemon on the stores, and not while a benchmark
holds the GPU.

p is the exact two-sided McNemar test on the discordant pairs (gained = b, lost = c):
p = min(1, 2 * BinomCDF(min(b, c); b + c, 0.5)). Verdict:
  real          p < 0.05 and the suite mean moved up
  regression    p < 0.05 and the suite mean moved down
  within noise  anything else
When several result files are gated, a pooled line sums gained/lost across suites
(`--pool` is accepted; the line is always printed for 2+ suites).

Judge verdicts are cached on (question, expected, answer), so an answer both trees
give identically cannot be judged two ways. The judge model and base URL are the
same for both sides.

    PY=~/Desktop/GitProjects/hippocampy/.venv/bin/python
    $PY eval_gate.py results/dmr-dev.json results/l10-dev.json --store <kept LoCoMo-10 store> \\
        --main ~/Desktop/GitProjects/worktrees/main --branch ~/Desktop/GitProjects/worktrees/my-branch \\
        --judge-model gemma4:26b --out gate.json

--store applies to LoCoMo-10 result files (one kept store per run); DMR result
files carry a store per question. `--llm-model` is passed through to both sides.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import diag_answer_replay as replay_mod  # noqa: E402

ALPHA = 0.05
HEADER = "| suite (split) | main | branch | Δ | gained | lost | p | verdict |"


def split_of(suite: str, options: Dict[str, Any]) -> str:
    """dev / held-out / dev-large, from the selection the run recorded.

    DMR: offset 50 (<= 50 questions) is the held-out split, offset 0 is dev,
    any other offset is dev-large. LoCoMo-10: conv-26 (or no selection) is dev,
    conv-30 is held-out, any other conversation selection is dev-large.
    """
    if suite == "dmr":
        off = int(options.get("offset") or 0)
        return "dev" if off == 0 else "held-out" if off == 50 else "dev-large"
    ids = options.get("conversation_ids") or []
    if not ids or ids == ["conv-26"]:
        return "dev"
    return "held-out" if ids == ["conv-30"] else "dev-large"


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p from the discordant counts (gained b, lost c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * cdf)


def verdict(p: float, delta: float) -> str:
    if p < ALPHA and delta > 0:
        return "real"
    if p < ALPHA and delta < 0:
        return "regression"
    return "within noise"


def fmt_p(p: float) -> str:
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def compare(main_rows: List[dict], branch_rows: List[dict]) -> Dict[str, Any]:
    """Paired comparison of two replays of the same questions (`asis` variant)."""
    b = {r["id"]: r for r in branch_rows}
    pairs = [(m, b[m["id"]]) for m in main_rows if m["id"] in b]
    n = len(pairs)
    m_pass = [bool(m["variants"]["asis"]["passed"]) for m, _ in pairs]
    b_pass = [bool(x["variants"]["asis"]["passed"]) for _, x in pairs]
    gained = [m["id"] for (m, _), mp, bp in zip(pairs, m_pass, b_pass) if bp and not mp]
    lost = [m["id"] for (m, _), mp, bp in zip(pairs, m_pass, b_pass) if mp and not bp]
    main_score = sum(m_pass) / n if n else 0.0
    branch_score = sum(b_pass) / n if n else 0.0
    delta = branch_score - main_score

    def lat(rows):
        xs = [r["variants"]["asis"].get("latency_ms") for r in rows]
        xs = [x for x in xs if x is not None]
        return {"p50": percentile(xs, 0.5), "p95": percentile(xs, 0.95)} if xs else None

    return {"n": n, "main": main_score, "branch": branch_score, "delta": delta,
            "gained": len(gained), "lost": len(lost), "gained_ids": gained, "lost_ids": lost,
            "p": mcnemar_p(len(gained), len(lost)), "verdict": verdict(mcnemar_p(len(gained), len(lost)), delta),
            "latency_ms": {"main": lat([m for m, _ in pairs]), "branch": lat([x for _, x in pairs])}}


def table_row(label: str, c: Dict[str, Any]) -> str:
    return (f"| {label} | {c['main']:.3f} | {c['branch']:.3f} | {c['delta']:+.3f} | {c['gained']} | "
            f"{c['lost']} | {fmt_p(c['p'])} | {c['verdict']} |")


def pooled(suites: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sum gained/lost (and the question-weighted means) over several suites."""
    n = sum(c["n"] for c in suites)
    g = sum(c["gained"] for c in suites)
    l = sum(c["lost"] for c in suites)  # noqa: E741
    main = sum(c["main"] * c["n"] for c in suites) / n if n else 0.0
    branch = sum(c["branch"] * c["n"] for c in suites) / n if n else 0.0
    p = mcnemar_p(g, l)
    return {"n": n, "main": main, "branch": branch, "delta": branch - main, "gained": g, "lost": l,
            "p": p, "verdict": verdict(p, branch - main)}


def pooled_line(suites: List[Dict[str, Any]]) -> Optional[str]:
    if len(suites) < 2:
        return None
    return table_row(f"pooled ({len(suites)} suites, n={sum(c['n'] for c in suites)})", pooled(suites))


def latency_line(label: str, c: Dict[str, Any]) -> Optional[str]:
    lm = c["latency_ms"]
    if not (lm["main"] and lm["branch"]):
        return None
    f = lambda d: f"p50 {d['p50'] / 1000:.1f}s / p95 {d['p95'] / 1000:.1f}s"  # noqa: E731
    return f"  {label}: main {f(lm['main'])}; branch {f(lm['branch'])}"


def gate(results: List[Path], passed, main_tree: Path, branch_tree: Path, store: Optional[Path] = None,
         ids=(), llm_model: Optional[str] = None, split: Optional[str] = None,
         log=print) -> Dict[str, Any]:
    """Run the replays and compare. `passed(d, answer)` is the judge (same for both sides)."""
    out: Dict[str, Any] = {"main": str(main_tree), "branch": str(branch_tree),
                           "llm_model": llm_model, "suites": [],
                           "provenance": {"main": replay_mod.tree_provenance(main_tree),
                                          "branch": replay_mod.tree_provenance(branch_tree)}}
    for path in results:
        sides = {}
        for side, tree in (("main", main_tree), ("branch", branch_tree)):
            suite, options, rows, info, _ = replay_mod.replay_result(
                path, ["asis"], passed, store, ids, tree, llm_model, log=log)
            sides[side] = {"rows": rows, "info": info}
        suite_label = f"{suite} ({split or split_of(suite, options)})"
        c = compare(sides["main"]["rows"], sides["branch"]["rows"])
        c.update(suite=suite, label=suite_label, result=str(path), options=options,
                 code={s: sides[s]["info"].get("campy_file") for s in sides},
                 answer_model={s: sides[s]["info"].get("llm_model") for s in sides},
                 rows={s: sides[s]["rows"] for s in sides})
        out["suites"].append(c)
    return out


def side_line(side: str, prov: Optional[Dict[str, Any]]) -> str:
    """`main <short sha>[+dirty] CAMPY_X=1 ...` for one side's provenance."""
    if not prov:
        return f"{side}: (no provenance recorded)"
    sha = (prov.get("sha") or "unknown")[:7]
    flags = " ".join(f"{k}={v}" for k, v in (prov.get("env") or {}).items()) or "no CAMPY_* flags"
    return f"{side}: {sha}{'+dirty' if prov.get('dirty') else ''} {flags}"


def provenance_lines(out: Dict[str, Any]) -> List[str]:
    prov = out.get("provenance") or {}
    return [side_line(s, prov.get(s)) for s in ("main", "branch")]


def render(out: Dict[str, Any]) -> str:
    lines = provenance_lines(out) + [""]
    lines += [HEADER, "|---|---|---|---|---|---|---|---|"]
    lines += [table_row(c["label"], c) for c in out["suites"]]
    pl = pooled_line(out["suites"])
    if pl:
        lines.append(pl)
    lat = [latency_line(c["label"], c) for c in out["suites"]]
    lat = [x for x in lat if x]
    if lat:
        lines += ["", "ask latency per question (replay, includes the answer model):"] + lat
    for c in out["suites"]:
        lines.append(f"\n{c['label']}: n={c['n']}; code main={c['code']['main']} branch={c['code']['branch']}")
        if c["gained_ids"]:
            lines.append(f"  gained: {', '.join(c['gained_ids'])}")
        if c["lost_ids"]:
            lines.append(f"  lost:   {', '.join(c['lost_ids'])}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", nargs="+", type=Path, help="run_all.py result files from --keep-store runs")
    ap.add_argument("--store", type=Path, help="LoCoMo-10: the run's kept store")
    ap.add_argument("--main", required=True, type=Path, dest="main_tree", help="hippocampy tree for main")
    ap.add_argument("--branch", required=True, type=Path, dest="branch_tree", help="hippocampy tree for the branch")
    ap.add_argument("--noise", type=float, default=None, help="DEPRECATED, ignored: the verdict is an exact McNemar test")
    ap.add_argument("--pool", action="store_true", help="also print a pooled line across suites (default for 2+ suites)")
    ap.add_argument("--llm-model", help="answer model for both sides ([llm].model in the store copy)")
    ap.add_argument("--split", choices=["dev", "held-out", "dev-large"], help="label for every suite (default: from the result's selection)")
    ap.add_argument("--ids", default="", help="comma-separated question ids (default: all)")
    ap.add_argument("--judge-provider", default="ollama")
    ap.add_argument("--judge-model", default="gemma4:26b")
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--non-answer-policy", choices=["judge", "legacy"], default="judge",
                    help="which answers are auto-failed WITHOUT the judge. judge (default, B539): only an "
                         "answer that opens with an abstention and asserts nothing after it; a hedged or "
                         "'abstain + fact' answer goes to the judge. legacy: any abstention phrase anywhere "
                         "(reproduces runs before B539)")
    ap.add_argument("--judge-votes", type=int, choices=[1, 3], default=1,
                    help="3 = judge twice, a third call breaks a disagreement (default 1)")
    ap.add_argument("--out", type=Path, help="write the table data and every answer and verdict as JSON")
    args = ap.parse_args()
    import scoring
    scoring.set_non_answer_policy(args.non_answer_policy)

    from llm_client import BaselineLLM
    judge = BaselineLLM(args.judge_provider, args.judge_model, args.judge_base_url)
    if args.noise is not None:
        print("note: --noise is deprecated and ignored; the verdict uses an exact McNemar test (p < 0.05)",
              file=sys.stderr)
    passed = replay_mod.make_passed(judge, args.judge_votes)
    out = gate(args.results, passed, args.main_tree, args.branch_tree, args.store,
               {i for i in args.ids.split(",") if i}, args.llm_model, args.split)
    out["judge"] = {"provider": args.judge_provider, "model": args.judge_model, "votes": args.judge_votes,
                    "non_answer_policy": args.non_answer_policy, "problems": dict(passed.problems)}
    print("\n" + render(out))
    print(f"\njudge problems (verdicts scored WRONG without a real verdict): {dict(passed.problems) or 'none'}")
    if args.out:
        args.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

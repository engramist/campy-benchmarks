#!/usr/bin/env python3
"""
campy-benchmarks / gate_report.py

Recompute the gate table (exact McNemar p, verdict, corrected split label) and
the pooled line from existing eval_gate.py `--out` JSON files, without replaying
anything:

    PY=~/Desktop/GitProjects/hippocampy/.venv/bin/python
    $PY gate_report.py gate-dmr.json gate-locomo.json

Each suite is recomputed from its per-side rows (variants.asis.passed) when they
are present, otherwise from the stored gained/lost counts. The split label is
re-derived from the recorded selection (old files may carry the old labelling).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eval_gate as g  # noqa: E402


def suites_of(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    out = []
    for s in data["suites"]:
        rows = s.get("rows") or {}
        if rows.get("main") and rows.get("branch"):
            c = g.compare(rows["main"], rows["branch"])
        else:  # no per-question rows: trust the stored counts
            p = g.mcnemar_p(s["gained"], s["lost"])
            c = {k: s[k] for k in ("n", "main", "branch", "delta", "gained", "lost")}
            c.update(p=p, verdict=g.verdict(p, s["delta"]))
        c["label"] = f"{s.get('suite', '?')} ({g.split_of(s.get('suite', ''), s.get('options') or {})}) {path.stem}"
        out.append(c)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gates", nargs="+", type=Path, help="eval_gate.py --out JSON files")
    args = ap.parse_args()
    suites = [c for p in args.gates for c in suites_of(p)]
    lines = [g.HEADER, "|---|---|---|---|---|---|---|---|"] + [g.table_row(c["label"], c) for c in suites]
    pl = g.pooled_line(suites)
    if pl:
        lines.append(pl)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

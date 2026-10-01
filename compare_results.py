#!/usr/bin/env python3
"""
campy-benchmarks / compare_results.py
Compare two saved result files without running anything.

`run_all.py --compare FILE` compares the run it just made against FILE. To
compare runs made earlier (e.g. one per hippocampy commit, B461), pass both
files here; the table is the same one --compare prints:

    python compare_results.py results/b461-base.json results/b461-b459.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from run_all import print_comparison_table


def main(argv: list) -> int:
    if len(argv) != 2:
        print("usage: compare_results.py BASELINE.json CURRENT.json", file=sys.stderr)
        return 2
    base, cur = (json.loads(Path(p).read_text()) for p in argv)
    for label, path, res in (("baseline", argv[0], base), ("current", argv[1], cur)):
        hc = (res.get("provenance", {}).get("hippocampy") or {})
        print(f"{label}: {path} (hippocampy {hc.get('commit', '?')}{' dirty' if hc.get('dirty') else ''})")
    print_comparison_table(base, cur)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

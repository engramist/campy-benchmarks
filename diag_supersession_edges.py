#!/usr/bin/env python3
"""
campy-benchmarks / diag_supersession_edges.py

Do the graph's edges say which value replaced which? For every LoCoMo
deprecation probe, finds the Concepts naming the current value (the probe's
`accept` patterns) and the retired ones (its `stale` patterns) in a store
kept with `run_all.py --isolated --keep-store`, and classifies every
Concept-Concept edge between them:

  right     current -REPLACES/CHOSEN_OVER-> retired
  INVERTED  retired -REPLACES/CHOSEN_OVER-> current (says the old value won)
  other     any other relation between them
  none      both sides exist but no edge links them

Edges like "PostgreSQL 14 CHOSEN_OVER PostgreSQL 16" (from "we
migrated from PostgreSQL 14 to PostgreSQL 16") are what this counts.

Run it with hippocampy's python on a copy of the store (no daemon on it):

    ~/Desktop/GitProjects/hippocampy/.venv/bin/python diag_supersession_edges.py /tmp/diag-store
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pyoxigraph as ox

from locomo.dataset import get_locomo_scenarios

NS = "https://campy.dev/ns#"
WINNER_FIRST = {"REPLACES", "CHOSEN_OVER"}  # "A <rel> B": A is the one in use


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    store = ox.Store.read_only(str(Path(sys.argv[1]) / "brain.db"))
    concepts = {sol["c"].value: sol["t"].value for sol in store.query(
        f"SELECT ?c ?t WHERE {{ ?c a <{NS}Concept> ; <{NS}text_raw> ?t }}")}
    edges = [(sol["a"].value, sol["p"].value.rsplit("#", 1)[-1], sol["b"].value) for sol in store.query(
        f"SELECT ?a ?p ?b WHERE {{ ?a ?p ?b . ?a a <{NS}Concept> . ?b a <{NS}Concept> }}")]
    print(f"{len(concepts)} concepts, {len(edges)} concept-concept edges\n")

    totals: Counter = Counter()
    for sc in get_locomo_scenarios():
        for p in sc.probes:
            if not (p.is_deprecation and p.stale and p.accept):
                continue
            match = lambda pats: {c for c, t in concepts.items() if any(re.search(x, t, re.I) for x in pats)}
            cur, old = match(p.accept) - match(p.stale), match(p.stale) - match(p.accept)
            found = []
            for a, rel, b in edges:
                if a in cur and b in old:
                    kind = "right" if rel in WINNER_FIRST else "other"
                elif a in old and b in cur:
                    kind = "INVERTED" if rel in WINNER_FIRST else "other"
                else:
                    continue
                found.append((kind, f"{concepts[a]} -{rel}-> {concepts[b]}"))
            if not cur or not old:
                verdict = "no concept for the " + ("current" if not cur else "retired") + " value"
            elif not found:
                verdict = "none"
            else:
                verdict = ", ".join(sorted({k for k, _ in found}))
            totals[verdict] += 1
            print(f"{p.id:<22} {verdict}")
            for kind, e in found:
                print(f"    {kind:<8} {e}")
    print("\nsummary:", dict(totals))
    return 0


if __name__ == "__main__":
    sys.exit(main())

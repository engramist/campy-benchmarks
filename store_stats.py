"""
campy-benchmarks / store_stats.py
What an isolated run left in the graph: node counts per type, and how many
Concepts are confirmed vs tentative and from which turn role.

B461 compares hippocampy save-gate changes on LoCoMo-10. Recall scores show
whether answers got better; these counts show what the Loop chose to save
to get there ("fewer, better-typed nodes is a result in itself").

Like isolation.py's preflight, this runs a short probe with the daemon's own
Python after the daemon has stopped, so the harness still imports nothing
from the engine. The probe touches only pyoxigraph and the store layout
($CAMPY_HOME/brain.db, an Oxigraph directory store) and IRIs in the
https://campy.dev/ns# namespace, which have been stable since the Oxigraph
cutover, so it works on older hippocampy commits too (a baseline commit
before Concept.origin_role simply reports every origin as "unset").

Counts cover everything the isolated daemon stored, i.e. every Campy suite
in the run. For per-suite numbers, run that suite alone (--suite locomo10).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

# Executed by the daemon's Python as `python -c PROBE <store_path>`.
PROBE = r'''
import json, sys
import pyoxigraph as ox
NS = "https://campy.dev/ns#"
store = ox.Store.read_only(sys.argv[1])
def rows(q):
    return list(store.query(q))
by_type = {}
for r in rows("SELECT ?t (COUNT(DISTINCT ?s) AS ?n) WHERE { ?s a ?t } GROUP BY ?t"):
    t = r["t"].value
    if t.startswith(NS):
        by_type[t[len(NS):]] = int(r["n"].value)
concepts = {}
for r in rows(
    "SELECT ?cl ?o (COUNT(DISTINCT ?s) AS ?n) WHERE { ?s a <%sConcept> . "
    "OPTIONAL { ?s <%sconfidence_low> ?cl } OPTIONAL { ?s <%sorigin_role> ?o } } "
    "GROUP BY ?cl ?o" % (NS, NS, NS)
):
    cl = r["cl"]
    o = r["o"]
    state = "unset" if cl is None else ("tentative" if cl.value.lower() == "true" else "confirmed")
    concepts.setdefault(state, {})[o.value if o is not None else "unset"] = int(r["n"].value)
print(json.dumps({"nodes_by_type": by_type, "concepts": concepts, "triples": len(store)}))
'''

# Node types the Step 4 save gate reifies (step4_pattern artifact types).
ARTIFACT_TYPES = ("Decision", "Constraint", "Requirement", "ActionItem")


def summarize(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Add the headline numbers B461 compares across commits."""
    by_type = raw.get("nodes_by_type", {})
    concepts = raw.get("concepts", {})

    def total(state: str) -> int:
        return sum(concepts.get(state, {}).values())

    out = dict(raw)
    out["total_nodes"] = sum(by_type.values())
    out["concept_nodes"] = by_type.get("Concept", 0)
    out["confirmed_concepts"] = total("confirmed")
    out["tentative_concepts"] = total("tentative")
    out["artifact_nodes"] = sum(by_type.get(t, 0) for t in ARTIFACT_TYPES)
    out["artifacts_by_type"] = {t: by_type.get(t, 0) for t in ARTIFACT_TYPES}
    return out


def collect(python: str, home: Path, timeout: float = 300.0) -> Dict[str, Any]:
    """Run the probe against $home/brain.db. The daemon must have stopped
    (the store is opened read-only, but a stopped daemon guarantees the
    counts are final). Never raises: failures are recorded, not fatal."""
    store = home / "brain.db"
    if not store.is_dir():
        return {"error": f"no Oxigraph store at {store}"}
    try:
        out = subprocess.run([python, "-c", PROBE, str(store)], cwd=home,
                             capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        return {"error": f"probe could not run: {e}"}
    if out.returncode != 0:
        return {"error": f"probe exited {out.returncode}: {out.stderr[-400:]}"}
    try:
        return summarize(json.loads(out.stdout.strip().splitlines()[-1]))
    except Exception as e:
        return {"error": f"unreadable probe output ({e}): {out.stdout[-200:]!r}"}


def headline(stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The numbers --compare prints, or {} when there are none."""
    if not stats or "error" in stats:
        return {}
    return {k: stats.get(k) for k in ("total_nodes", "concept_nodes", "confirmed_concepts",
                                      "tentative_concepts", "artifact_nodes")}

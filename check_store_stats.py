#!/usr/bin/env python3
"""
campy-benchmarks / check_store_stats.py

Self-test for store_stats.py (graph counts after an isolated run, B461):

  * summarize() derives the headline counts (nodes, confirmed/tentative
    Concepts, artifacts) from the probe's raw output;
  * --compare prints the graph rows and LoCoMo-10 per-category rows when a
    results file has them, and nothing extra when it doesn't;
  * the probe itself, run against a small Oxigraph store, when a Python with
    pyoxigraph is available (CAMPY_PYTHON, else the first word of
    CAMPY_MCP_CMD, else this interpreter). Skipped, with a note, otherwise.
    The fixture uses the same IRIs hippocampy writes (https://campy.dev/ns#,
    booleans as xsd:boolean literals).

Plain script, same convention as the other check_*.py files:

    python3 check_store_stats.py

Exits 0 and prints "OK" on success, exits 1 listing each failure.
"""

from __future__ import annotations

import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import store_stats
from run_all import print_comparison_table

failures = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


# --- summarize --------------------------------------------------------------
raw = {
    "nodes_by_type": {"Concept": 10, "Decision": 2, "Constraint": 1, "Message": 30},
    "concepts": {"confirmed": {"user": 4, "unset": 1}, "tentative": {"assistant": 3, "user": 2}},
    "triples": 500,
}
s = store_stats.summarize(raw)
check(s["total_nodes"] == 43, f"total_nodes: {s['total_nodes']}")
check(s["confirmed_concepts"] == 5 and s["tentative_concepts"] == 5, f"concept split: {s}")
check(s["artifact_nodes"] == 3 and s["artifacts_by_type"]["Requirement"] == 0, f"artifacts: {s}")
check(store_stats.headline({"error": "x"}) == {} and store_stats.headline(None) == {},
      "errors and missing stats give no headline")

# --- compare output -----------------------------------------------------------
def result(stats, judge):
    return {"provenance": {"store": {"isolated": True, "graph_stats": stats}},
            "suites": {"locomo10": {"ingest_seconds": 100.0, "by_category": {
                "single-hop": {"n": 5, "judge_accuracy": judge}, "adversarial": {"n": 2, "abstention": 0.5}}}}}

before = result(s, 0.6)
after = result(store_stats.summarize({**raw, "nodes_by_type": {**raw["nodes_by_type"], "Concept": 7}}), 0.7)
buf = io.StringIO()
with redirect_stdout(buf):
    print_comparison_table(before, after)
text = buf.getvalue()
check("| Concepts | 10 | 7 (-3) |" in text, f"graph row with delta missing:\n{text}")
check("LoCoMo-10 single-hop (judge_accuracy) | 0.6 |" in text, "per-category judge row missing")
check("LoCoMo-10 adversarial (abstention)" in text, "adversarial abstention row missing")
check("Memory Construction (s)" in text, "ingest time row missing")

buf = io.StringIO()
with redirect_stdout(buf):
    print_comparison_table({"suites": {}}, {"suites": {}})
check("Graph nodes" not in buf.getvalue(), "no graph rows without graph stats")

with tempfile.TemporaryDirectory() as tmp:
    a, b = Path(tmp) / "a.json", Path(tmp) / "b.json"
    a.write_text(json.dumps(before))
    b.write_text(json.dumps(after))
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "compare_results.py"),
                           str(a), str(b)], capture_output=True, text=True, timeout=120)
    check(proc.returncode == 0 and "| Concepts | 10 | 7 (-3) |" in proc.stdout,
          f"compare_results.py on two files: rc={proc.returncode} {proc.stderr[-300:]}")

# --- the probe against a real Oxigraph store ------------------------------------
NS = "https://campy.dev/ns#"
XSD_BOOL = "http://www.w3.org/2001/XMLSchema#boolean"
FIXTURE = "\n".join([
    f'<urn:c1> a <{NS}Concept> ; <{NS}confidence_low> "false"^^<{XSD_BOOL}> ; <{NS}origin_role> "user" .',
    f'<urn:c2> a <{NS}Concept> ; <{NS}confidence_low> "true"^^<{XSD_BOOL}> ; <{NS}origin_role> "assistant" .',
    f'<urn:c3> a <{NS}Concept> ; <{NS}confidence_low> "true"^^<{XSD_BOOL}> .',
    f'<urn:d1> a <{NS}Decision> .',
    '<urn:x> a <http://example.org/Other> .',
]) + "\n"

candidates = [os.environ.get("CAMPY_PYTHON"),
              shlex.split(os.environ["CAMPY_MCP_CMD"])[0] if os.environ.get("CAMPY_MCP_CMD") else None,
              sys.executable]
python = next((p for p in candidates if p and subprocess.run(
    [p, "-c", "import pyoxigraph"], capture_output=True).returncode == 0), None)
if python is None:
    print("note: no Python with pyoxigraph found; probe check skipped (set CAMPY_PYTHON)")
else:
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp)
        load = ("import sys, pyoxigraph as ox; s = ox.Store(sys.argv[1]); "
                "s.load(sys.stdin.buffer.read(), format=ox.RdfFormat.TURTLE); s.flush()")
        proc = subprocess.run([python, "-c", load, str(home / "brain.db")], input=FIXTURE.encode(),
                              capture_output=True)
        check(proc.returncode == 0, f"fixture load failed: {proc.stderr[-300:]!r}")
        got = store_stats.collect(python, home)
        check("error" not in got, f"probe failed: {got}")
        if "error" not in got:
            check(got["nodes_by_type"] == {"Concept": 3, "Decision": 1},
                  f"only campy types are counted: {got['nodes_by_type']}")
            check(got["concepts"] == {"confirmed": {"user": 1},
                                      "tentative": {"assistant": 1, "unset": 1}},
                  f"concept split by state and origin: {got['concepts']}")
            check(got["confirmed_concepts"] == 1 and got["tentative_concepts"] == 2
                  and got["artifact_nodes"] == 1, f"headline: {store_stats.headline(got)}")
    check("error" in store_stats.collect(python, Path(tempfile.gettempdir()) / "no-such-home"),
          "a missing store is reported, not raised")

if failures:
    print(f"FAIL -- {len(failures)} store-stats checks failed:")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)
print("OK -- store stats summary, compare rows" + ("" if python is None else " and probe") + " checks passed")

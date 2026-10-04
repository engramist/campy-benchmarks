#!/usr/bin/env python3
"""
campy-benchmarks / diag_bundle.py

What does `ask` see for a question, and what does the graph hold about a
topic? Against a store kept with `run_all.py --isolated --keep-store`:

  1. runs hippocampy's own bundle compiler (compile_bundle, the context `ask`
     answers from) on the question and prints every section in full -- the
     --trace-context summary only keeps section sizes and top items;
  2. prints every graph node with a literal matching the topic regex, with
     its properties and edges (both directions; embeddings left out).

Run it with hippocampy's python, on a COPY of the store (opening it may
write lock/WAL files), with no daemon running on it:

    cp -R /tmp/campy-bench-XXXX /tmp/diag-store
    ~/Desktop/GitProjects/hippocampy/.venv/bin/python diag_bundle.py \\
        /tmp/diag-store "What is our active production database engine and version?" \\
        "postgres|primary database"
"""

from __future__ import annotations

import asyncio
import json
import sys
import tomllib
from pathlib import Path

from campy.brain.hippocampus.graph.oxigraph_client import OxigraphClient
from campy.brain.thalamus.bundle_compiler import compile_bundle

SKIP = ("embedding",)  # predicates whose values are vectors, not evidence


def short(term) -> str:
    v = getattr(term, "value", str(term))
    return v.rsplit("/", 2)[-2] + "/" + v.rsplit("/", 1)[-1][:12] if v.startswith("https://campy.dev/id/") else (
        v.rsplit("#", 1)[-1] if v.startswith("https://campy.dev/ns#") else v)


def dump_graph(client: OxigraphClient, pattern: str) -> None:
    store = client.store
    hits = {sol["s"].value for sol in store.query(
        "SELECT DISTINCT ?s WHERE { ?s ?p ?o . FILTER(isLiteral(?o) && REGEX(STR(?o), %s, \"i\")) }"
        % json.dumps(pattern))}
    print(f"\n=== graph nodes with a literal matching /{pattern}/i: {len(hits)}")
    for s in sorted(hits):
        print(f"\n--- {short(type('t', (), {'value': s}))}")
        for sol in store.query(f"SELECT ?p ?o WHERE {{ <{s}> ?p ?o }}"):
            p = short(sol["p"])
            if not any(k in p for k in SKIP):
                print(f"    {p}: {str(getattr(sol['o'], 'value', sol['o']))[:160]}")
        for sol in store.query(f"SELECT ?s2 ?p WHERE {{ ?s2 ?p <{s}> }}"):
            print(f"    <- {short(sol['p'])} from {short(sol['s2'])}")


async def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    home, question = Path(sys.argv[1]), sys.argv[2]
    pattern = sys.argv[3] if len(sys.argv) > 3 else None
    cfg_path = home / "config.toml"
    config = tomllib.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    client = OxigraphClient(home / "brain.db")

    bundle = await compile_bundle(query=question, db=client, config=config)
    print(f"question: {question!r}\nbundle: {bundle.total_token_estimate} tokens, "
          f"{len(bundle.sections)} section(s)")
    for sec in bundle.sections:
        print(f"\n=== section {sec.section_type!r}: {len(sec.content)} item(s), ~{sec.token_estimate} tokens")
        for item in sec.content:
            if isinstance(item, dict):
                item = {k: v for k, v in item.items() if not any(s in k for s in SKIP)}
            print(f"  - {json.dumps(item, default=str)[:400]}")
    if pattern:
        dump_graph(client, pattern)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

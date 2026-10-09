#!/usr/bin/env python3
"""
campy-benchmarks / diag_graph_audit.py

B472 Phase 0: what does Campy's graph hold after a benchmark run? Reads kept
stores (`run_all.py --isolated --keep-store`) and reports, per store and in
total:

  nodes       count per node type (rdf:type in the campy namespace)
  edges       count per predicate between two typed nodes
  concepts    count, share confidence_low, by gist class, by schema.org type
  properties  every predicate used on Concept nodes (does any carry
              schema.org property values, e.g. a startDate?)
  variants    concepts that differ only by case, punctuation or a greeting
              ("Caroline", "Caroline!", "Congrats Caroline"): groups of
              surface forms one entity was split into
  junk        concepts that are harness prefix text ("Speaker", "Session")
              or unanchored time phrases ("ten years ago", "1, 7 days")
  top labels  the most frequent concept labels by pathway strength

Read-only. Run it after the run has finished (no daemon on the stores), with
hippocampy's python (it needs pyoxigraph):

    ~/Desktop/GitProjects/hippocampy/.venv/bin/python diag_graph_audit.py results/r20c-locomo10-q60-baselines.json
    ~/Desktop/GitProjects/hippocampy/.venv/bin/python diag_graph_audit.py /path/to/store [...] [--top 50] [--json out.json]

A results file stands for every store its records name (`store`, written
with --keep-store; LongMemEval and DMR keep one store per question). A
LoCoMo-10 run keeps one store for the run: pass its directory (printed as
"store kept: ..." at the end of the run).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

NS = "https://campy.dev/ns#"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

# Greetings and interjections that wrap a name ("Congrats Caroline", "Wow Caroline")
_WRAPPERS = {"congrats", "congratulations", "wow", "hey", "hi", "hello", "thanks", "thank", "you", "oh",
             "omg", "yay", "aww", "awesome", "great", "dear", "good", "morning", "night", "bye", "love"}
_PREFIX_JUNK = re.compile(r"^(speaker( \d)?|session( \d+)?|user|assistant)$", re.I)
_TIME_PHRASE = re.compile(
    r"\b(ago|yesterday|today|tomorrow|tonight|last|next|this (week|month|year|summer|winter|spring|fall)|"
    r"days?|weeks?|months?|years?|hours?|minutes?|mon|tue|wed|thu|fri|sat|sun|"
    r"january|february|march|april|may|june|july|august|september|october|november|december)\b"
    r"|\d{4}/\d{2}/\d{2}|^\d+[, ]", re.I)


def local(iri: str) -> str:
    return iri.rsplit("#", 1)[-1] if iri.startswith(NS) else iri


def normalize(label: str) -> str:
    """The surface-form key: lower case, punctuation and greeting wrappers dropped."""
    words = re.findall(r"[^\W_]+", label.lower())
    core = [w for w in words if w not in _WRAPPERS]
    return " ".join(core or words)


def stores_from(args: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for a in args:
        p = Path(a)
        if p.suffix == ".json" and p.is_file():
            data = json.loads(p.read_text())
            for suite in (data.get("suites") or {}).values():
                for d in suite.get("details") or []:
                    if d.get("store"):
                        out.append(Path(d["store"]))
        else:
            out.append(p)
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def audit(home: Path, top: int) -> Dict:
    import pyoxigraph as ox

    store = ox.Store.read_only(str(home / "brain.db"))
    q = lambda s: list(store.query(s))
    nodes = Counter({local(r["t"].value): int(r["n"].value) for r in q(
        f"SELECT ?t (COUNT(?s) AS ?n) WHERE {{ ?s <{RDF_TYPE}> ?t FILTER(STRSTARTS(STR(?t), '{NS}')) }} GROUP BY ?t")})
    edges = Counter({local(r["p"].value): int(r["n"].value) for r in q(
        f"SELECT ?p (COUNT(*) AS ?n) WHERE {{ ?a ?p ?b . ?a <{RDF_TYPE}> ?ta . ?b <{RDF_TYPE}> ?tb "
        f"FILTER(STRSTARTS(STR(?ta), '{NS}') && STRSTARTS(STR(?tb), '{NS}')) }} GROUP BY ?p")})
    props = Counter({local(r["p"].value): int(r["n"].value) for r in q(
        f"SELECT ?p (COUNT(*) AS ?n) WHERE {{ ?c <{RDF_TYPE}> <{NS}Concept> ; ?p ?o FILTER(isLiteral(?o)) }} GROUP BY ?p")})
    concepts = []
    for r in q(f"""SELECT ?c ?t ?g ?s ?low ?ps WHERE {{
                     ?c <{RDF_TYPE}> <{NS}Concept> ; <{NS}text_raw> ?t .
                     OPTIONAL {{ ?c <{NS}gist_class> ?g }} OPTIONAL {{ ?c <{NS}schema_org_type> ?s }}
                     OPTIONAL {{ ?c <{NS}confidence_low> ?low }} OPTIONAL {{ ?c <{NS}pathway_strength> ?ps }} }}"""):
        concepts.append({
            "text": r["t"].value,
            "gist": r["g"].value if r["g"] else "",
            "schema": r["s"].value if r["s"] else "",
            "low": (r["low"].value == "true") if r["low"] else False,
            "ps": float(r["ps"].value) if r["ps"] else 0.0,
        })
    groups: Dict[str, set] = defaultdict(set)
    for c in concepts:
        groups[normalize(c["text"])].add(c["text"])
    split = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
    junk_prefix = [c["text"] for c in concepts if _PREFIX_JUNK.match(c["text"].strip())]
    junk_time = [c["text"] for c in concepts if _TIME_PHRASE.search(c["text"])]
    by_label = Counter()
    for c in concepts:
        by_label[c["text"]] += 1
    top_labels = [c["text"] for c in sorted(concepts, key=lambda c: -c["ps"])[:top]]
    return {
        "store": str(home),
        "nodes": dict(nodes),
        "edges": dict(edges),
        "concept_literal_properties": dict(props),
        "concepts": len(concepts),
        "concepts_confidence_low": sum(c["low"] for c in concepts),
        "concepts_by_gist": dict(Counter(c["gist"] or "-" for c in concepts)),
        "concepts_by_schema": dict(Counter(c["schema"] or "-" for c in concepts)),
        "variant_groups": len(split),
        "concepts_in_variant_groups": sum(len(v) for v in split.values()),
        "variant_examples": dict(sorted(split.items(), key=lambda kv: -len(kv[1]))[:15]),
        "junk_prefix": len(junk_prefix),
        "junk_prefix_examples": sorted(set(junk_prefix))[:15],
        "time_phrase_concepts": len(junk_time),
        "time_phrase_examples": sorted(set(junk_time))[:20],
        "top_labels": top_labels,
    }


def merge(rows: List[Dict]) -> Dict:
    tot: Dict = {"stores": len(rows)}
    for key in ("nodes", "edges", "concept_literal_properties", "concepts_by_gist", "concepts_by_schema"):
        c: Counter = Counter()
        for r in rows:
            c.update(r[key])
        tot[key] = dict(c.most_common())
    for key in ("concepts", "concepts_confidence_low", "variant_groups", "concepts_in_variant_groups",
                "junk_prefix", "time_phrase_concepts"):
        tot[key] = sum(r[key] for r in rows)
    return tot


def show(r: Dict, label: str) -> None:
    print(f"\n=== {label} ===")
    print("nodes:", ", ".join(f"{k} {v}" for k, v in sorted(r["nodes"].items(), key=lambda kv: -kv[1])) or "none")
    print("edges:", ", ".join(f"{k} {v}" for k, v in sorted(r["edges"].items(), key=lambda kv: -kv[1])) or "none")
    n = r["concepts"] or 1
    print(f"concepts: {r['concepts']}  confidence_low: {r['concepts_confidence_low']} "
          f"({r['concepts_confidence_low'] / n:.0%})")
    print(f"  split into variants: {r['concepts_in_variant_groups']} concepts in {r['variant_groups']} groups "
          f"({r['concepts_in_variant_groups'] / n:.0%})")
    print(f"  harness-prefix junk: {r['junk_prefix']}  time phrases: {r['time_phrase_concepts']} "
          f"({r['time_phrase_concepts'] / n:.0%})")
    print("  by gist:", r["concepts_by_gist"])
    print("  by schema.org type:", r["concepts_by_schema"])
    print("  literal properties on Concept:", ", ".join(sorted(r["concept_literal_properties"])))
    for key in ("variant_examples", "junk_prefix_examples", "time_phrase_examples", "top_labels"):
        if r.get(key):
            print(f"  {key}: {r[key]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="store directories and/or result files with `store` fields")
    ap.add_argument("--top", type=int, default=50, help="top concept labels by pathway strength, per store")
    ap.add_argument("--show-each", action="store_true", help="print every store, not only the total")
    ap.add_argument("--json", type=Path, help="write every store's audit and the total here")
    args = ap.parse_args()
    homes = stores_from(args.paths)
    rows = []
    for h in homes:
        if not (h / "brain.db").exists():
            print(f"skipped (no brain.db): {h}", file=sys.stderr)
            continue
        rows.append(audit(h, args.top))
    if not rows:
        print("no store found", file=sys.stderr)
        return 1
    if args.show_each or len(rows) == 1:
        for r in rows:
            show(r, r["store"])
    if len(rows) > 1:
        tot = merge(rows)
        print(f"\n=== total over {tot['stores']} stores ===")
        show({**tot, "top_labels": [], "variant_examples": {}, "junk_prefix_examples": [],
              "time_phrase_examples": []}, f"{tot['stores']} stores")
        # examples from the first few stores, so the total isn't blind
        for r in rows[:3]:
            print(f"\n  examples from {r['store']}:")
            for key in ("variant_examples", "junk_prefix_examples", "time_phrase_examples"):
                print(f"    {key}: {r[key]}")
            print(f"    top_labels: {r['top_labels'][:20]}")
    if args.json:
        args.json.write_text(json.dumps({"stores": rows, "total": merge(rows)}, indent=1))
        print(f"\nwritten: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

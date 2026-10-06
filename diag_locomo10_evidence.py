#!/usr/bin/env python3
"""
campy-benchmarks / diag_locomo10_evidence.py

Why did Campy's bundle miss the gold evidence of LoCoMo-10 questions? For
every question in a `--suite locomo10` results file (by default only those
the judge failed, categories 1-4), finds the gold evidence turns among the
Messages of a store kept with `run_all.py --isolated --keep-store`, and shows
where each fell in the conversation stage's two channels
(GraphGateway._bundle_conversation):

  sim     cosine similarity to the question (the stage keeps >= 0.30)
  allrank rank by similarity among all Messages, ignoring the floor
  vrank   rank among vector hits >= 0.30 (- = below the floor)
  frank   rank among FTS hits (- = no lexical match)
  fts     how many Messages the FTS query matched at all

and a summary: how many evidence turns are below the floor, and how many
pure vector top-k (k = 6, 20, 50; no floor) would have reached. That
separates "the embedding can't find it" from "the floor or the limit cut
it", and shows how far the lexical channel helps.

Run it with hippocampy's python on a COPY of the store (no daemon on it):

    cp -R /tmp/campy-bench-XXXX /tmp/l10-store
    ~/Desktop/GitProjects/hippocampy/.venv/bin/python diag_locomo10_evidence.py \\
        /tmp/l10-store results/r2-locomo10-c1-q60.json [--all]
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import sys
from pathlib import Path

import sqlite_vec

from campy.brain.hippocampus.graph import embeddings as emb
from campy.brain.hippocampus.graph.vector_store import _fts_match_expression

from locomo10.dataset import ensure_dataset, load_conversations

MIN_SCORE = 0.30  # GraphGateway._bundle_conversation
ECHO = 0.985
KS = (6, 20, 50)


def norm(t: str) -> str:
    return " ".join((t or "").split())


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        print(__doc__)
        return 2
    home, results = Path(args[0]), json.load(open(args[1]))
    show_all = "--all" in sys.argv
    details = results["suites"]["locomo10"]["details"]

    model = "sentence-transformers/all-MiniLM-L6-v2"
    cfg = home / "config.toml"
    if cfg.exists():
        import tomllib
        model = tomllib.loads(cfg.read_text()).get("embeddings", {}).get("model", model)
    conn = sqlite3.connect(home / "vectors.db")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    texts = dict(conn.execute("SELECT uri, text FROM lexical WHERE uri LIKE '%/Message/%'"))
    by_text: dict[str, list[str]] = {}
    for uri, t in texts.items():
        by_text.setdefault(norm(t), []).append(uri)

    convs = {c.sample_id: c for c in load_conversations(ensure_dataset(log=lambda *a: None))}
    print(f"store: {home}  messages: {len(texts)}  embedding model: {model}\n")
    print(f"{'question':<16}{'cat':>4}{'judge':>6}  evidence  {'sim':>6}{'allrank':>8}{'vrank':>6}{'frank':>6}{'fts':>5}")

    stats = {"turns": 0, "missing": 0, "below_floor": 0, "lexical": 0, **{f"top{k}": 0 for k in KS}}
    allranks = []
    for d in details:
        if d.get("category") not in (1, 2, 3, 4) or (d.get("judge") and not show_all):
            continue
        conv = convs.get(d.get("conversation"))
        turn_of = {t.dia_id: t for t in conv.turns()} if conv else {}
        q = sqlite_vec.serialize_float32(list(emb.embed(d["question"], model_name=model)))
        sims = {uri: 1.0 - dist for uri, dist in conn.execute(
            "SELECT uri, vec_distance_cosine(embedding, ?) FROM vectors WHERE uri LIKE '%/Message/%'", (q,))}
        ordered = [u for u, s in sorted(sims.items(), key=lambda kv: -kv[1]) if s < ECHO]
        allrank = {u: i for i, u in enumerate(ordered)}
        vrank = {u: i for i, u in enumerate(u for u in ordered if sims[u] >= MIN_SCORE)}
        match = _fts_match_expression(d["question"])
        fts = [u for (u,) in conn.execute(
            "SELECT uri FROM lexical WHERE lexical MATCH ? ORDER BY bm25(lexical)", (match,))
            if "/Message/" in u] if match else []
        frank = {u: i for i, u in enumerate(fts)}

        for i, ev in enumerate(d.get("evidence") or []):
            turn = turn_of.get(ev)
            uris = by_text.get(norm(turn.content())) if turn else None
            head = f"{d['id'] if i == 0 else '':<16}{d['category'] if i == 0 else '':>4}" \
                   f"{({True: 'pass', False: 'FAIL'}.get(d.get('judge'), '-')) if i == 0 else '':>6}  {ev:<8}"
            stats["turns"] += 1
            if not uris:
                stats["missing"] += 1
                print(f"{head}  (turn not found among stored Messages)")
                continue
            u = min(uris, key=lambda x: allrank.get(x, 10**9))
            s, ar = sims.get(u, float("nan")), allrank.get(u)
            stats["below_floor"] += s < MIN_SCORE
            stats["lexical"] += u in frank
            for k in KS:
                stats[f"top{k}"] += ar is not None and ar < k
            if ar is not None:
                allranks.append(ar)
            print(f"{head}  {s:6.3f}{ar if ar is not None else '-':>8}{vrank.get(u, '-'):>6}"
                  f"{frank.get(u, '-'):>6}{len(fts):>5}")

    n = stats["turns"] - stats["missing"]
    print(f"\nevidence turns: {stats['turns']} ({stats['missing']} not found in the store)")
    if n:
        print(f"below the 0.30 floor: {stats['below_floor']}/{n}")
        print("pure vector top-k (no floor): " + ", ".join(f"k={k}: {stats[f'top{k}']}/{n}" for k in KS))
        print(f"lexical match (any rank): {stats['lexical']}/{n}")
        print(f"median rank by similarity: {statistics.median(allranks) if allranks else '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

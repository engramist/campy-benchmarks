#!/usr/bin/env python3
"""
campy-benchmarks / diag_conversation_stage.py

Why did `ask` miss a fact? Replays the vector and FTS gates of hippocampy's
conversation stage (GraphGateway._bundle_conversation, B454) against a store
kept with `run_all.py --isolated --keep-store`, and shows, for every stored
message that mentions a topic word, where it stopped:

  vec   cosine similarity to the question (the stage keeps >= 0.30)
  vrank rank among Message vector hits (- = below the floor)
  frank rank among Message FTS hits (- = no lexical match)
  terms question content words the message contains; a lexical-only hit
        needs 2 (or the only one there is), or -- hippocampy B459 -- the
        query's anchor word: its rarest word in the store that one of the
        top vector hits also contains (applied when the installed campy
        has VectorStore.document_frequencies)
  gate  why it was dropped, or "cand" for a candidate
  pick  * = one of the `limit` messages the stage hands to the bundle
        (fused vector+FTS rank, newest copy of repeated text)

Run it with hippocampy's python (it needs campy and sqlite_vec):

    ~/Desktop/GitProjects/hippocampy/.venv/bin/python diag_conversation_stage.py \\
        /tmp/campy-bench-XXXX "What is the current required tool for tracing?" tracing

Roles, timestamps and archived flags come from the graph (brain.db, opened
read-only); if that fails, assistant turns can't be told apart and the
"pick" column is approximate.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import re

import sqlite_vec

from campy.brain.hippocampus.graph import embeddings as emb
from campy.brain.hippocampus.graph.vector_store import (
    VectorStore, _fts_match_expression, fts_content_terms)

B459 = hasattr(VectorStore, "document_frequencies")

MIN_SCORE = 0.30  # GraphGateway._bundle_conversation
ECHO = 0.985
LIMIT = 6  # [retrieval] conversation_limit default


def graph_meta(home: Path, uris):
    """{uri: (role, created, archived)} from the Oxigraph store, or {}."""
    try:
        import pyoxigraph as ox
        store = ox.Store.read_only(str(home / "brain.db"))
        meta = {}
        uris = list(uris)
        for i in range(0, len(uris), 200):
            values = " ".join(f"<{u}>" for u in uris[i:i + 200])
            for sol in store.query(f"""
                SELECT ?s ?role ?created ?archived WHERE {{
                    VALUES ?s {{ {values} }}
                    OPTIONAL {{ ?s <https://campy.dev/ns#role> ?role }}
                    OPTIONAL {{ ?s <https://campy.dev/ns#created_at> ?created }}
                    OPTIONAL {{ ?s <https://campy.dev/ns#archived> ?archived }}
                }}"""):
                val = lambda k: sol[k].value if sol[k] is not None else None
                meta[sol["s"].value] = (val("role"), val("created") or "",
                                        str(val("archived")).lower() in ("true", "1"))
        return meta
    except Exception as e:  # locked by a running daemon, schema drift, ...
        print(f"(graph metadata unavailable: {e!r})\n")
        return {}


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    home, question, topics = Path(sys.argv[1]), sys.argv[2], [t.lower() for t in sys.argv[3:]]
    model = "sentence-transformers/all-MiniLM-L6-v2"
    cfg = home / "config.toml"
    if cfg.exists():
        import tomllib
        model = tomllib.loads(cfg.read_text()).get("embeddings", {}).get("model", model)
    conn = sqlite3.connect(home / "vectors.db")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    q = sqlite_vec.serialize_float32(list(emb.embed(question, model_name=model)))
    texts = dict(conn.execute("SELECT uri, text FROM lexical WHERE uri LIKE '%/Message/%'"))
    sims = {uri: 1.0 - d for uri, d in conn.execute(
        "SELECT uri, vec_distance_cosine(embedding, ?) FROM vectors WHERE uri LIKE '%/Message/%'", (q,))}

    vec_hits = [u for u, s in sorted(sims.items(), key=lambda kv: -kv[1]) if MIN_SCORE <= s < ECHO]
    vrank = {u: i for i, u in enumerate(vec_hits)}
    match = _fts_match_expression(question)
    fts = [u for (u,) in conn.execute(
        "SELECT uri FROM lexical WHERE lexical MATCH ? ORDER BY bm25(lexical)", (match,))
        if "/Message/" in u] if match else []
    frank = {u: i for i, u in enumerate(fts)}
    terms = fts_content_terms(question)
    need = min(2, len(terms) or 1)
    words = lambda t: set(re.findall(r"[^\W_]+", t.lower()))
    anchors = set()
    if B459:
        df = {}
        for t in terms:
            n = conn.execute("SELECT count(*) FROM lexical WHERE lexical MATCH ? AND uri LIKE '%/Message/%'",
                             (f'"{t}"',)).fetchone()[0]
            if n:
                df[t] = n
        top_words = set().union(*(words(texts.get(u, "")) for u in vec_hits[:LIMIT]))
        rarest = min(df.values(), default=0)
        anchors = {t for t, n in df.items() if n == rarest and t in top_words}

    print(f"question: {question!r}\nembedding model: {model}\ncontent terms: {terms} "
          f"(lexical-only hits need {need}"
          + (f", or the B459 anchor {sorted(anchors)})" if B459 else "; pre-B459 rule)")
          + f"\nmessages in store: {len(texts)}, vector hits >= "
          f"{MIN_SCORE}: {len(vec_hits)}, FTS hits: {len(fts)}\n")
    meta = graph_meta(home, texts)
    norm = lambda t: " ".join(t.lower().split())

    def gate(u):
        text = texts[u].strip()
        role, _, archived = meta.get(u, ("user", "", False))
        if sims.get(u, 0) >= ECHO:
            return "OUT: echo"
        if u not in vrank and u not in frank:
            return "OUT: no match"
        if role != "user":
            return f"OUT: role {role}"
        if archived:
            return "OUT: archived"
        if text.endswith("?") or norm(text) == norm(question):
            return "OUT: question"
        if u not in vrank:
            n = sum(1 for t in terms if t in text.lower())
            if n < need and not anchors & words(text):
                return f"OUT: {n} term(s)"
        return "cand"

    # the stage's selection: RRF over the two rank lists, newest copy of
    # repeated text, top LIMIT
    score = {u: (1 / (60 + vrank[u]) if u in vrank else 0) + (1 / (60 + frank[u]) if u in frank else 0)
             for u in texts}
    newest = {}
    for u in texts:
        if gate(u) == "cand":
            created = meta.get(u, (None, "", False))[1]
            k = norm(texts[u])
            if k not in newest or created > newest[k][0]:
                newest[k] = (created, u)
    picked = {u for _, u in sorted(newest.values(), key=lambda cu: -score[cu[1]])[:LIMIT]}

    print("picked for the bundle:")
    for u in sorted(picked, key=lambda u: meta.get(u, (None, "", False))[1]):
        print(f"  {texts[u][:130]}")
    rows = [u for u in texts if any(t in texts[u].lower() for t in topics)]
    rows.sort(key=lambda u: -sims.get(u, -1))
    print(f"\nmessages mentioning {topics}:")
    print(f"{'vec':>6} {'vrank':>5} {'frank':>5} {'terms':>5} pick gate            text")
    for u in rows:
        n = sum(1 for t in terms if t in texts[u].lower())
        print(f"{sims.get(u, float('nan')):6.3f} {vrank.get(u, '-'):>5} {frank.get(u, '-'):>5} "
              f"{n:>5} {'*' if u in picked else ' ':^4} {gate(u):<15} {texts[u][:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

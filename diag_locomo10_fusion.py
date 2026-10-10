#!/usr/bin/env python3
"""
campy-benchmarks / diag_locomo10_fusion.py

Replays the bundle's conversation stage (GraphGateway._bundle_conversation,
hippocampy babe6d3) offline on a kept LoCoMo-10 store, under variants of its
vector/FTS fusion, and counts how many gold evidence turns each variant puts
in the stage's top `limit`. Run on a COPY of the store, with no daemon on it.

Why: the stage fuses the two channels by reciprocal rank (1/(60+rank) per
list). In a LoCoMo conversation the speaker names are in most turns, so a
question naming a speaker matches nearly every Message lexically: nearly every
vector hit then gets a second RRF term from the FTS list, and a turn found
only lexically (its rare word at FTS rank 0, but under the 0.30 similarity
floor) cannot outscore it. The variants test that.

  base       the stage as it is
  nohdf      the FTS query drops terms in more than --hdf of the Messages
             (the speaker names), unless that would drop every term
  noname     the FTS list keeps its order (all terms, bm25) but drops the
             hits that match only high-df terms: a name still ranks, but a
             name-only match no longer adds an RRF term
  vec        vector channel only
  fts        FTS channel only (all terms)
  fts_nohdf  FTS channel only, high-df terms dropped
  rerank     base, then the best --rerank-n candidates re-ordered by a
             cross-encoder (hippocampy B477, plan M2.1) before the cut. The
             candidates are the stage's gated, de-duplicated list, as the
             stage has it after the B459 gate and before the top-`limit` cut.
             Scores "<speaker>: <text>", as the stage does for a turn that
             names a speaker; the speaker comes from the dataset turn with the
             same text (a store ingested with --turn-metadata keeps the bare
             text in the lexical table). Skipped if the model cannot load.

Not replayed: the B463 successor bridge (adds superseding statements; LoCoMo
has almost none), the user-role filter (LoCoMo turns are all `user`), and
the bundle's other stages. So `base` recall is a lower bound on the run's
evidence_recall; compare variants with each other, and `base` with the run
to check the replay is faithful.

    ~/Desktop/GitProjects/hippocampy/.venv/bin/python diag_locomo10_fusion.py \\
        /tmp/l10-store results/r2-locomo10-c1-q60.json [--limits 6,12,20] [--hdf 0.2] [--show]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

import sqlite_vec

from campy.brain.hippocampus.graph import embeddings as emb
from campy.brain.hippocampus.graph.vector_store import _fts_match_expression, fts_content_terms

from locomo10.dataset import ensure_dataset, load_conversations

MIN_SCORE, ECHO, RRF_K = 0.30, 0.985, 60  # GraphGateway._bundle_conversation
VARIANTS = ("base", "nohdf", "noname", "vec", "fts", "fts_nohdf", "rerank")
RERANK_MODEL, RERANK_N = "Xenova/ms-marco-MiniLM-L-6-v2", 50


def norm(t: str) -> str:
    return " ".join((t or "").lower().split())


def words(t: str) -> set:
    return set(re.findall(r"[^\W_]+", (t or "").lower()))


class Store:
    def __init__(self, home: Path):
        model = "sentence-transformers/all-MiniLM-L6-v2"
        cfg = home / "config.toml"
        if cfg.exists():
            import tomllib
            model = tomllib.loads(cfg.read_text()).get("embeddings", {}).get("model", model)
        self.model = model
        self.conn = sqlite3.connect(home / "vectors.db")
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.text = dict(self.conn.execute("SELECT uri, text FROM lexical WHERE uri LIKE '%/Message/%'"))
        self.n = len(self.text)
        self.speaker: dict = {}  # uri -> speaker, filled by main() for the rerank variant

    def vector_hits(self, question: str, limit: int) -> list:
        q = sqlite_vec.serialize_float32(list(emb.embed(question, model_name=self.model)))
        sims = self.conn.execute(
            "SELECT uri, 1.0 - vec_distance_cosine(embedding, ?) AS s FROM vectors "
            "WHERE uri LIKE '%/Message/%' ORDER BY s DESC", (q,)).fetchall()
        return [u for u, s in sims if MIN_SCORE <= s < ECHO][: limit * 40]

    def df(self, term: str) -> int:
        return self.conn.execute("SELECT count(*) FROM lexical WHERE lexical MATCH ? AND uri LIKE '%/Message/%'",
                                 (f'"{term}"',)).fetchone()[0]

    def fts_hits(self, match: str | None, limit: int) -> list:
        if not match:
            return []
        try:
            rows = self.conn.execute("SELECT uri FROM lexical WHERE lexical MATCH ? ORDER BY bm25(lexical) LIMIT ?",
                                     (match, limit * 60)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [u for (u,) in rows if "/Message/" in u][: limit * 40]


def nohdf_match(store: Store, question: str, hdf: float) -> str | None:
    """The stage's FTS expression without the terms in > hdf of the Messages."""
    kept = [t for t in fts_content_terms(question) if store.df(t) <= hdf * store.n]
    if not kept:
        return _fts_match_expression(question)
    return " OR ".join(f'"{t}"' for t in kept)


_RERANK: dict = {}


def cross_scores(model: str, question: str, texts: list) -> list:
    """Cross-encoder scores (fastembed/ONNX, CPU), loaded once; [] if unavailable."""
    if model not in _RERANK:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            _RERANK[model] = TextCrossEncoder(model_name=model, providers=["CPUExecutionProvider"])
        except Exception as e:
            print(f"rerank variant unavailable: {e}")
            _RERANK[model] = None
    enc = _RERANK[model]
    return [float(x) for x in enc.rerank(question, texts)] if enc else []


def stage(store: Store, question: str, limit: int, variant: str, hdf: float,
          rerank_model: str = RERANK_MODEL, rerank_n: int = RERANK_N) -> list:
    """The URIs the conversation stage would return (top `limit` by fused score)."""
    vec = [] if variant in ("fts", "fts_nohdf") else store.vector_hits(question, limit)
    if variant == "vec":
        fts = []
    elif variant in ("nohdf", "fts_nohdf"):
        fts = store.fts_hits(nohdf_match(store, question, hdf), limit)
    elif variant == "noname":
        rare = set(store.fts_hits(nohdf_match(store, question, hdf), limit * 10))
        fts = [u for u in store.fts_hits(_fts_match_expression(question), limit * 10) if u in rare][: limit * 40]
    else:
        fts = store.fts_hits(_fts_match_expression(question), limit)
    ranked: dict = {}
    for hits in (vec, fts):
        for rank, u in enumerate(hits):
            ranked[u] = ranked.get(u, 0.0) + 1.0 / (RRF_K + rank)
    # lexical-only hits: two query content words, or the anchor (B459)
    terms = fts_content_terms(question)
    need = min(2, len(terms) or 1)
    df = {t: n for t in terms if (n := store.df(t)) > 0}
    top_vec_words = set().union(*(words(store.text.get(u, "")) for u in vec[:limit])) if vec else set()
    rarest = min(df.values(), default=0)
    anchors = {t for t, n in df.items() if n == rarest and t in top_vec_words}
    vec_set, qn = set(vec), norm(question)
    best: dict = {}
    for u, score in ranked.items():
        text = (store.text.get(u) or "").strip()
        if not text or norm(text) == qn or text.endswith("?"):
            continue
        if u not in vec_set and sum(1 for t in terms if t in text.lower()) < need and not anchors & words(text):
            continue
        key = norm(text)
        if key not in best or score > best[key][0]:
            best[key] = (score, u)
    cands = [u for _, u in sorted(best.values(), key=lambda su: -su[0])]
    if variant == "rerank":
        window = cands[: max(1, rerank_n)]
        scores = cross_scores(rerank_model, question, [f"{store.speaker[u]}: {store.text.get(u, '')}" if u in store.speaker
                                    else store.text.get(u, "") for u in window])
        if scores:
            window = [u for _, u in sorted(zip(scores, window), key=lambda su: -su[0])]
            cands = window + cands[len(window):]
    return cands[:limit]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("store", type=Path)
    ap.add_argument("results", type=Path)
    ap.add_argument("--limits", default="6,12,20")
    ap.add_argument("--hdf", type=float, default=0.2, help="nohdf: drop terms in more than this share of Messages")
    ap.add_argument("--rerank-model", default=RERANK_MODEL, help="rerank: cross-encoder checkpoint")
    ap.add_argument("--rerank-n", type=int, default=RERANK_N, help="rerank: candidates re-ordered before the cut")
    ap.add_argument("--show", action="store_true", help="per-question rows")
    args = ap.parse_args()
    limits = [int(x) for x in args.limits.split(",")]

    store = Store(args.store)
    by_text: dict = {}
    for u, t in store.text.items():
        by_text.setdefault(norm(t), []).append(u)
    details = json.loads(args.results.read_text())["suites"]["locomo10"]["details"]
    convs = {c.sample_id: c for c in load_conversations(ensure_dataset(log=lambda *a: None))}
    speaker_of = {norm(t.text): t.speaker for c in convs.values() for t in c.turns() if t.speaker}
    for u, t in store.text.items():
        if norm(t) in speaker_of:
            store.speaker[u] = speaker_of[norm(t)]
    print(f"store: {args.store}  messages: {store.n}  model: {store.model}  hdf: {args.hdf}")

    totals = {(v, k): 0 for v in VARIANTS for k in limits}
    q_any = {(v, k): 0 for v in VARIANTS for k in limits}
    n_turns = n_q = 0
    run_recall = []
    for d in details:
        if d.get("category") not in (1, 2, 3, 4):
            continue
        conv = convs.get(d.get("conversation"))
        turn_of = {t.dia_id: t for t in conv.turns()} if conv else {}
        # content() carries the date/speaker stamp; a --turn-metadata store holds the bare text
        ev = [set(by_text.get(norm(turn_of[e].content()), []) or by_text.get(norm(turn_of[e].text), []))
              for e in d.get("evidence") or [] if e in turn_of]
        ev = [s for s in ev if s]
        if not ev:
            continue
        n_q += 1
        n_turns += len(ev)
        if d.get("evidence_recall") is not None:
            run_recall.append(d["evidence_recall"])
        row = []
        for k in limits:
            for v in VARIANTS:
                got = set(stage(store, d["question"], k, v, args.hdf, args.rerank_model, args.rerank_n))
                hits = sum(1 for s in ev if s & got)
                totals[(v, k)] += hits
                q_any[(v, k)] += hits > 0
                if k == limits[0]:
                    row.append(f"{v}={hits}")
        if args.show:
            print(f"{d['id']:<16} c{d['category']} {'pass' if d.get('judge') else 'FAIL':<5} "
                  f"ev={len(ev)} " + " ".join(row) + f"  {d['question'][:60]}")

    print(f"\n{n_q} questions (cat 1-4), {n_turns} evidence turns found in the store")
    if run_recall:
        print(f"the run's own evidence_recall (all bundle stages): {sum(run_recall) / len(run_recall):.3f}")
    print(f"\n{'variant':<11}" + "".join(f"{'limit ' + str(k):>22}" for k in limits))
    print(f"{'':<11}" + "".join(f"{'turns   questions':>22}" for _ in limits))
    for v in VARIANTS:
        print(f"{v:<11}" + "".join(f"{totals[(v, k)] / n_turns:>11.3f}{q_any[(v, k)]:>7}/{n_q:<3}" for k in limits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

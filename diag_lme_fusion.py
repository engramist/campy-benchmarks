#!/usr/bin/env python3
"""
campy-benchmarks / diag_lme_fusion.py

Replays the bundle's conversation stage offline on the stores a LongMemEval
run kept (`--keep-store`; each question's store path is in its result row),
under variants of the stage, and counts how many evidence turns each variant
puts in the stage's top `limit`. Run with no daemon on the stores.

Why: on the `s` variant (~500 turns per question) evidence recall is 0.139,
against 0.61 on `oracle` (the evidence sessions only). The variants separate
the possible causes: the evidence ranks too low in both channels, is cut by
the 0.30 similarity floor, loses to whole sessions that match a query word,
or is crowded out by turns from the wrong sessions.

  base      the stage as on hippocampy main (B470: FTS drops terms in > --hdf
            of the Messages; user turns only)
  nodate    base, the question without the harness's "(Current date: ...)"
            prefix: its weekday/month words match every turn of the sessions
            held on that day
  vec       vector channel only
  fts       FTS channel only (B470 query)
  nofloor   base without the 0.30 similarity floor
  sess3     base within the 3 sessions whose best user turn is most similar
  sess5     same, 5 sessions
  gold      base within the evidence sessions only: the ceiling a session
            prefilter could reach

Not replayed: the B463 successor bridge, B471's assistant_said section and
the bundle's other stages. An assistant evidence turn
(single-session-assistant) is out of reach of every variant; such questions
are counted but cannot score.

Per question (--show) it also prints where the first evidence turn sits in
the full vector ranking of user turns (rank, similarity) and in the FTS
ranking, which tells "ranked low" from "under the floor" from "no shared word".

    ~/Desktop/GitProjects/hippocampy/.venv/bin/python diag_lme_fusion.py \\
        results/r19-lme-s.json [--limits 6,12,20] [--hdf 0.2] [--show]
"""

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import sqlite_vec

from campy.brain.hippocampus.graph import embeddings as emb

import diag_locomo10_fusion as base_diag
from diag_locomo10_fusion import ECHO, MIN_SCORE, norm, stage
from longmemeval.dataset import ensure_dataset, load_questions
from longmemeval.runner import ask_text, turn_content
from longmemeval.scoring import INGEST_MAX_CHARS

VARIANTS = ("base", "nodate", "vec", "fts", "nofloor", "sess3", "sess5", "gold")
_KEY = 200  # characters of a turn's opening used to match it to a stored Message
_DATE = re.compile(r"^\[([^\]]+)\]")
_ASK_PREFIX = re.compile(r"^\(Current date: [^)]*\)\s*")


def key(text: str) -> str:
    return norm(text)[:_KEY]


class LMEStore(base_diag.Store):
    """A kept store, restricted to user turns (and optionally some sessions)."""

    def __init__(self, home: Path, role_of: dict):
        super().__init__(home)
        self.role = {u: role_of.get(key(t)) for u, t in self.text.items()}
        self.session = {u: (m.group(1) if (m := _DATE.match(t or "")) else None) for u, t in self.text.items()}
        self.user = {u for u, r in self.role.items() if r == "user"}
        self.allowed = self.user
        self.floor = MIN_SCORE

    @lru_cache(maxsize=64)
    def sims(self, question: str) -> tuple:
        q = sqlite_vec.serialize_float32(list(emb.embed(question, model_name=self.model)))
        return tuple(self.conn.execute(
            "SELECT uri, 1.0 - vec_distance_cosine(embedding, ?) AS s FROM vectors "
            "WHERE uri LIKE '%/Message/%' ORDER BY s DESC", (q,)).fetchall())

    def vector_hits(self, question: str, limit: int) -> list:
        return [u for u, s in self.sims(question) if self.floor <= s < ECHO and u in self.allowed][: limit * 40]

    @lru_cache(maxsize=256)
    def _fts_all(self, match: str) -> tuple:
        try:
            rows = self.conn.execute("SELECT uri FROM lexical WHERE lexical MATCH ? ORDER BY bm25(lexical) LIMIT 5000",
                                     (match,)).fetchall()
        except Exception:
            return ()
        return tuple(u for (u,) in rows if "/Message/" in u)

    def fts_hits(self, match: str | None, limit: int) -> list:
        return [u for u in self._fts_all(match) if u in self.allowed][: limit * 40] if match else []

    def top_sessions(self, question: str, n: int) -> set:
        best: dict = {}
        for u, s in self.sims(question):
            if u in self.user and s < ECHO and self.session.get(u):
                best.setdefault(self.session[u], s)  # sims are sorted: the first is the max
        return set(sorted(best, key=lambda k: -best[k])[:n])


def run_variant(store: LMEStore, question: str, limit: int, v: str, hdf: float, gold_sessions: set) -> list:
    store.allowed, store.floor = store.user, MIN_SCORE
    try:
        if v == "nodate":
            question = _ASK_PREFIX.sub("", question)
        elif v == "nofloor":
            store.floor = -1.0
        elif v in ("sess3", "sess5", "gold"):
            keep = gold_sessions if v == "gold" else store.top_sessions(question, int(v[4:]))
            store.allowed = {u for u in store.user if store.session.get(u) in keep}
        mode = {"vec": "vec", "fts": "fts_nohdf"}.get(v, "nohdf")
        return stage(store, question, limit, mode, hdf)
    finally:
        store.allowed, store.floor = store.user, MIN_SCORE


def where(store: LMEStore, question: str, ev_uris: set, hdf: float) -> str:
    users = [(u, s) for u, s in store.sims(question) if u in store.user and s < ECHO]
    vr = next(((i, s) for i, (u, s) in enumerate(users) if u in ev_uris), None)
    fts = [u for u in store._fts_all(base_diag.nohdf_match(store, question, hdf) or "") if u in store.user]
    fr = next((i for i, u in enumerate(fts) if u in ev_uris), None)
    v = f"vec#{vr[0]}@{vr[1]:.2f}" if vr else "vec#-"
    return f"{v} fts#{fr if fr is not None else '-'}/{len(fts)}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", type=Path, help="a run_all.py result file from a --keep-store LongMemEval run")
    ap.add_argument("--limits", default="6,12,20")
    ap.add_argument("--hdf", type=float, default=0.2, help="B470: drop FTS terms in more than this share")
    ap.add_argument("--show", action="store_true", help="per-question rows")
    args = ap.parse_args()
    limits = [int(x) for x in args.limits.split(",")]

    suite = json.loads(args.results.read_text())["suites"]["longmemeval"]
    variant = (suite.get("dataset") or {}).get("variant", "s")
    qs = {q.question_id: q for q in load_questions(ensure_dataset(variant, log=lambda *a: None))}
    totals = {(v, k): 0 for v in VARIANTS for k in limits}
    q_any = {(v, k): 0 for v in VARIANTS for k in limits}
    n_turns = n_q = n_asst = 0
    for d in suite["details"]:
        q = qs.get(d["id"])
        home = Path(d.get("store") or "")
        if q is None or not (home / "vectors.db").exists():
            print(f"{d['id']}: no kept store ({d.get('store')}), skipped")
            continue
        role_of, ev_keys, gold = {}, [], set()
        for s in q.sessions:
            for i, t in enumerate(s.turns):
                text = turn_content(s.date, t.content)[:INGEST_MAX_CHARS]
                role_of[key(text)] = t.role
                if t.has_answer:
                    ev_keys.append((key(text), t.role))
                    gold.add(s.date)
        store = LMEStore(home, role_of)
        by_key: dict = {}
        for u, t in store.text.items():
            by_key.setdefault(key(t), set()).add(u)
        ev = [by_key.get(k, set()) for k, _ in ev_keys]
        ev = [s for s in ev if s]
        if not ev:
            print(f"{d['id']}: no evidence turn found in the store, skipped")
            continue
        n_q += 1
        n_turns += len(ev)
        n_asst += sum(1 for _, r in ev_keys if r != "user")
        question = ask_text(q)
        row = []
        for k in limits:
            for v in VARIANTS:
                got = set(run_variant(store, question, k, v, args.hdf, gold))
                hits = sum(1 for s in ev if s & got)
                totals[(v, k)] += hits
                q_any[(v, k)] += hits > 0
                if k == limits[0]:
                    row.append(f"{v}={hits}")
        if args.show:
            unhit = set().union(*ev)
            print(f"{d['id']:<14} {q.question_type[:22]:<22} {'pass' if d.get('judge') else 'FAIL':<5} "
                  f"msgs={store.n} user={len(store.user)} ev={len(ev)} {where(store, question, unhit, args.hdf)}  "
                  + " ".join(row) + f"\n{'':<15}{q.question[:100]}")
    if not n_q:
        print("no question had a kept store with its evidence")
        return 1
    print(f"\n{n_q} questions, {n_turns} evidence turns found in the stores "
          f"({n_asst} assistant turns: out of reach of this stage)")
    print(f"\n{'variant':<9}" + "".join(f"{'limit ' + str(k):>22}" for k in limits))
    print(f"{'':<9}" + "".join(f"{'turns   questions':>22}" for _ in limits))
    for v in VARIANTS:
        print(f"{v:<9}" + "".join(f"{totals[(v, k)] / n_turns:>11.3f}{q_any[(v, k)]:>7}/{n_q:<3}" for k in limits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

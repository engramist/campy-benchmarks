"""
campy-benchmarks / dmr / runner.py
DMR (MSC-Self-Instruct) on Campy. Per question, on a fresh store (each
question has its own pair of speakers):

  * write the 4 earlier sessions' turns in order, role `user` (both speakers
    are people; the bundle's conversation stage keeps only user turns), with
    the session and its distance from the current session in the text:
    `[Session 2, 7 days 8 hours ago] Speaker 1: ...` -- the LoCoMo-10
    convention; then settle consolidation once;
  * compile_context for evidence recall (against the derived evidence);
  * ask the question, addressed to the speaker it asks about:
    `Speaker 2 asks Speaker 1: <question>` (the verbatim question when the
    answer's speaker can't be found).

The judge runs later (run_all.finalize_dmr).
"""

from __future__ import annotations

import time
from typing import Any, Callable, ContextManager, Dict, List, Optional

from dmr.dataset import ensure_dataset, load_questions, sha256
from dmr.scoring import aggregate, f1
from locomo10.scoring import evidence_recall_from_texts
from mcp_client import CampyMCPClient
from records import bundle_texts, summarize_bundle

SMOKE_QUESTIONS = 5


def dmr_options(args, smoke: bool) -> Dict[str, Any]:
    n = getattr(args, "dmr_questions", None)
    return {"max_questions": n if n is not None else (SMOKE_QUESTIONS if smoke else None)}


def turn_content(session, turn) -> str:
    return f"[Session {session.index}, {session.time_back}] {turn.speaker}: {turn.text}"


def ask_text(q) -> str:
    if q.answerer:
        asker = "Speaker 2" if q.answerer == "Speaker 1" else "Speaker 1"
        return f"{asker} asks {q.answerer}: {q.question}"
    return q.question


def run_dmr(new_store: Callable[[], ContextManager[CampyMCPClient]],
            opts: Optional[Dict[str, Any]] = None, log=print) -> Dict[str, Any]:
    opts = dict(opts or {})
    path = ensure_dataset(log=log)
    questions = load_questions(path, **opts)
    details: List[Dict[str, Any]] = []
    ingest_s = 0.0
    for n, q in enumerate(questions, 1):
        turns = sum(len(s.turns) for s in q.sessions)
        log(f"    [{n}/{len(questions)}] {q.question_id} ({turns} turns)")
        turn_text = {f"{s.index}:{i}": turn_content(s, t) for s in q.sessions for i, t in enumerate(s.turns)}
        rec: Dict[str, Any] = {
            "id": q.question_id, "question": ask_text(q), "raw_question": q.question,
            "expected": q.answer, "answerer": q.answerer, "turns": turns, "evidence": q.evidence_ids(),
        }
        with new_store() as client:
            t0 = time.perf_counter()
            for s in q.sessions:
                for t in s.turns:
                    client.notify_turn(role="user", content=turn_content(s, t),
                                       session_id=f"dmr_{q.question_id}_s{s.index}")
            client.run_sweep()
            ingest_s += time.perf_counter() - t0
            t1 = time.perf_counter()
            ctx = client.compile_context(rec["question"])
            rec["compile_latency_ms"] = round((time.perf_counter() - t1) * 1000.0, 1)
            rec["evidence_recall"] = (evidence_recall_from_texts(rec["evidence"], turn_text, bundle_texts(ctx))
                                      if rec["evidence"] else None)
            rec["context"] = summarize_bundle(ctx)
            t2 = time.perf_counter()
            rec["answer"] = client.ask(rec["question"], session_id=f"dmr_{q.question_id}_eval")
            rec["latency_ms"] = round((time.perf_counter() - t2) * 1000.0, 1)
        rec["f1"] = f1(rec["answer"], q.answer)
        rec["judge"] = None  # filled by run_all.finalize_dmr
        details.append(rec)
    return {
        "suite": "dmr",
        "dataset": {"file": str(path), "sha256": sha256(path)[:16], "options": opts,
                    "questions": len(details), "turns": sum(d["turns"] for d in details)},
        **aggregate(details),
        "ingest_seconds": round(ingest_s, 1),
        "avg_latency_ms": round(sum(d["latency_ms"] for d in details) / max(1, len(details)), 1),
        "details": details,
    }

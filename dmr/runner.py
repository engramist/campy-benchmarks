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
    `Speaker 2 asks Speaker 1 ("you" means Speaker 1): <question> Answer
    with what Speaker 1 said earlier.` (the verbatim question when the
    answer's speaker can't be found).

The judge runs later (run_all.finalize_dmr).
"""

from __future__ import annotations

import time
from typing import Any, Callable, ContextManager, Dict, List, Optional

from dmr.dataset import ensure_dataset, load_questions, sha256
from dmr.scoring import aggregate, f1
from locomo10.scoring import evidence_recall_from_texts
from fresh_store import run_on_fresh_store
from mcp_client import CampyClientError, CampyMCPClient
from records import bundle_texts, summarize_bundle

SMOKE_QUESTIONS = 5


def dmr_options(args, smoke: bool) -> Dict[str, Any]:
    n = getattr(args, "dmr_questions", None)
    return {"max_questions": n if n is not None else (SMOKE_QUESTIONS if smoke else None)}


def turn_content(session, turn) -> str:
    return f"[Session {session.index}, {session.time_back}] {turn.speaker}: {turn.text}"


def ask_text(q) -> str:
    """The question, with who is asking whom. DMR questions say "you" (the
    speaker being asked), and answered as is, the memory's own model reads
    "you" as itself ("I don't have a pet... that was between you and another
    user", R13a). So the framing names the speaker "you" refers to and says
    whose earlier words answer it."""
    if q.answerer:
        asker = "Speaker 2" if q.answerer == "Speaker 1" else "Speaker 1"
        return (f'{asker} asks {q.answerer} ("you" means {q.answerer}): {q.question} '
                f"Answer with what {q.answerer} said earlier.")
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
            "expected": q.answer, "answerer": q.answerer, "answerer_source": q.answerer_source, "turns": turns, "evidence": q.evidence_ids(),
        }
        def body(client: CampyMCPClient, q=q, rec=rec, turn_text=turn_text) -> None:
            t0 = time.perf_counter()
            for s in q.sessions:
                for t in s.turns:
                    client.notify_turn(role="user", content=turn_content(s, t),
                                       session_id=f"dmr_{q.question_id}_s{s.index}")
            client.run_sweep()
            rec["ingest_seconds"] = round(time.perf_counter() - t0, 1)
            t1 = time.perf_counter()
            ctx = client.compile_context(rec["question"])
            rec["compile_latency_ms"] = round((time.perf_counter() - t1) * 1000.0, 1)
            rec["evidence_recall"] = (evidence_recall_from_texts(rec["evidence"], turn_text, bundle_texts(ctx))
                                      if rec["evidence"] else None)
            rec["context"] = summarize_bundle(ctx)
            t2 = time.perf_counter()
            rec["answer"] = client.ask(rec["question"], session_id=f"dmr_{q.question_id}_eval")
            rec["latency_ms"] = round((time.perf_counter() - t2) * 1000.0, 1)

        err = run_on_fresh_store(new_store, body, q.question_id, log=log)
        if err is not None:
            # the daemon failed twice: no measurement, scored wrong, never judged
            rec.update({"error": err, "answer": "", "f1": 0.0, "evidence_recall": None, "latency_ms": None,
                        "judge": False, "reason": "error"})
            details.append(rec)
            continue
        ingest_s += rec["ingest_seconds"]
        rec["f1"] = f1(rec["answer"], q.answer)
        rec["judge"] = None  # filled by run_all.finalize_dmr
        details.append(rec)
    if details and all(d.get("error") for d in details):
        raise CampyClientError(f"every question failed; the last: {details[-1]['error']}")
    return {
        "suite": "dmr",
        # 3: every question framed (answerer from turns, else personas, else Speaker 1);
        # 2: framed only when a turn quotes the answer; 1 (unrecorded): no framing
        "dataset": {"file": str(path), "sha256": sha256(path)[:16], "options": opts, "question_framing": 3,
                    "questions": len(details), "turns": sum(d["turns"] for d in details)},
        **aggregate(details),
        "ingest_seconds": round(ingest_s, 1),
        "avg_latency_ms": round(sum(lat) / len(lat), 1) if (lat := [d["latency_ms"] for d in details
                                                                     if d.get("latency_ms") is not None]) else None,
        "details": details,
    }

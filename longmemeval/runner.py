"""
campy-benchmarks / longmemeval / runner.py
Run LongMemEval against Campy.

Every question has its own chat history about "the user", so each question
gets a FRESH store (`new_store()`: a new isolated daemon in a real run): a
shared store would let one question's history answer -- or contradict --
another's. Per question: write every history turn with its real role (user
or assistant) and the session date as a text prefix (notify_turn has no
timestamp parameter; the same convention as LoCoMo-10), settle, call
compile_context (turn-level evidence recall against the `has_answer` turns),
then ask the question with its date, as the official generation prompt does
("Current Date: ..."). The judge runs later (run_all.finalize_longmemeval).

Cost: the oracle variant writes ~2-30 turns per question; the `s` variant
~500, i.e. roughly 15-20 min of consolidation per question on a local 8B
model -- subset with --lme-questions.
"""

from __future__ import annotations

import time
from typing import Any, Callable, ContextManager, Dict, List, Optional

from longmemeval.dataset import ensure_dataset, load_questions, sha256
from longmemeval.scoring import INGEST_MAX_CHARS, aggregate, evidence_recall
from mcp_client import CampyMCPClient
from records import bundle_texts, summarize_bundle

SMOKE_DEFAULTS = {"variant": "oracle", "max_questions": 7}


def lme_options(args, smoke: bool) -> Dict[str, Any]:
    opts = {
        "variant": getattr(args, "lme_variant", None) or "oracle",
        "max_questions": getattr(args, "lme_questions", None),
        "types": getattr(args, "lme_types", None),
    }
    if smoke and opts["max_questions"] is None:
        opts["max_questions"] = SMOKE_DEFAULTS["max_questions"]
    return opts


def turn_content(date: str, content: str) -> str:
    return f"[{date}] {content}"


def ask_text(q) -> str:
    return f"(Current date: {q.question_date}) {q.question}"


def run_longmemeval(new_store: Callable[[], ContextManager[CampyMCPClient]],
                    opts: Optional[Dict[str, Any]] = None, log=print) -> Dict[str, Any]:
    opts = dict(opts or {})
    variant = opts.pop("variant", "oracle")
    path = ensure_dataset(variant, log=log)
    questions = load_questions(path, **opts)
    details: List[Dict[str, Any]] = []
    ingest_s = 0.0
    for n, q in enumerate(questions, 1):
        log(f"    [{n}/{len(questions)}] {q.question_id} ({q.category}, "
            f"{sum(len(s.turns) for s in q.sessions)} turns)")
        with new_store() as client:
            t0 = time.perf_counter()
            for s in q.sessions:
                for t in s.turns:
                    client.notify_turn(role=t.role, content=turn_content(s.date, t.content),
                                       session_id=f"lme_{q.question_id}_{s.session_id}")
            client.run_sweep()
            ingest_s += time.perf_counter() - t0
            turn_text = {f"{s.session_id}:{i}": turn_content(s.date, t.content)
                         for s in q.sessions for i, t in enumerate(s.turns)}
            rec: Dict[str, Any] = {
                "id": q.question_id, "question_type": q.question_type, "category": q.category,
                "abstention": q.abstention, "question": q.question, "question_date": q.question_date,
                "expected": q.answer, "sessions": len(q.sessions),
                "turns": sum(len(s.turns) for s in q.sessions), "evidence": q.evidence_ids(),
                # evidence turns longer than Campy stores (the rest of the turn is dropped)
                "evidence_over_ingest_limit": sum(len(turn_text[e]) > INGEST_MAX_CHARS
                                                  for e in q.evidence_ids()),
            }
            t1 = time.perf_counter()
            ctx = client.compile_context(ask_text(q))
            rec["compile_latency_ms"] = round((time.perf_counter() - t1) * 1000.0, 1)
            rec["evidence_recall"] = evidence_recall(rec["evidence"], turn_text, bundle_texts(ctx))
            rec["context"] = summarize_bundle(ctx)
            t2 = time.perf_counter()
            rec["answer"] = client.ask(ask_text(q), session_id=f"lme_{q.question_id}_eval")
            rec["latency_ms"] = round((time.perf_counter() - t2) * 1000.0, 1)
        rec["judge"] = None  # filled by run_all.finalize_longmemeval
        details.append(rec)
    return {
        "suite": "longmemeval",
        "dataset": {"variant": variant, "file": str(path), "sha256": sha256(path)[:16], "options": opts,
                    "questions": len(details), "turns": sum(d["turns"] for d in details)},
        **aggregate(details),
        "ingest_seconds": round(ingest_s, 1),
        "avg_latency_ms": round(sum(d["latency_ms"] for d in details) / max(1, len(details)), 1),
        "details": details,
    }

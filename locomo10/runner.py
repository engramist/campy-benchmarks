"""
campy-benchmarks / locomo10 / runner.py
Run the published LoCoMo QA task against Campy.

For each conversation: write every turn (role "user"; the session date and
speaker are in the text, see dataset.L10Turn.content), settle once, then per
question call compile_context (retrieval diagnostic: which evidence turns it
surfaced) and ask. The LLM judge runs later, over Campy and the baselines
together (run_all.finalize_locomo10), so every system gets the same judge.

Size: 10 conversations, 5,882 turns, 1,986 questions (~15-23k tokens of
dialogue per conversation). A full run on a local 8B model takes hours; use
--locomo10-conversations / --locomo10-max-questions to subset.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from locomo10.dataset import ensure_dataset, load_conversations
from locomo10.scoring import aggregate, evidence_recall_from_texts, score_answer
from mcp_client import CampyMCPClient
from records import bundle_texts, summarize_bundle

SMOKE_DEFAULTS = {"conversations": 1, "max_questions": 25}


def locomo10_options(args, smoke: bool) -> Dict[str, Any]:
    """Subset options from CLI args (shared with the baselines)."""
    opts = {
        "conversations": getattr(args, "locomo10_conversations", None),
        "max_questions": getattr(args, "locomo10_max_questions", None),
        "categories": getattr(args, "locomo10_categories", None),
    }
    if smoke:
        for k, v in SMOKE_DEFAULTS.items():
            if opts[k] is None:
                opts[k] = v
    return opts


def session_id(sample_id: str, session: int) -> str:
    return f"locomo10_{sample_id}_s{session}"


def run_locomo10(client: CampyMCPClient, smoke: bool = False, trace_context: bool = True,
                 opts: Optional[Dict[str, Any]] = None, turn_metadata: str = "text") -> Dict[str, Any]:
    """trace_context here defaults ON: evidence recall is this suite's main
    retrieval diagnostic (one compile_context call per question).

    turn_metadata: "text" writes "[date] Speaker: text" as the turn;
    "fields" writes the text alone, with the speaker and the session date in
    notify_turn's speaker/occurred_at (hippocampy B472)."""
    opts = opts or {}
    convs = load_conversations(ensure_dataset(), **opts)
    details: List[Dict[str, Any]] = []
    ingest_s = 0.0
    for conv in convs:
        t0 = time.perf_counter()
        fields = turn_metadata == "fields"
        for turn in conv.turns():
            if fields:
                client.notify_turn(role="user", content=turn.body(),
                                   session_id=session_id(conv.sample_id, turn.session),
                                   speaker=turn.speaker, occurred_at=turn.occurred_at())
            else:
                client.notify_turn(role="user", content=turn.content(),
                                   session_id=session_id(conv.sample_id, turn.session))
        client.run_sweep()  # settle once per conversation, before its questions
        ingest_s += time.perf_counter() - t0
        # what was stored, so evidence recall finds it in the bundle
        turn_text = {t.dia_id: (t.body() if fields else t.content()) for t in conv.turns()}
        for q in conv.questions:
            rec: Dict[str, Any] = {
                "id": q.id, "conversation": conv.sample_id, "category": q.category,
                "question": q.question, "raw_question": q.raw_question,
                "expected": q.answer, "evidence": q.evidence,
            }
            if trace_context:
                t1 = time.perf_counter()
                ctx = client.compile_context(q.question)
                rec["compile_latency_ms"] = round((time.perf_counter() - t1) * 1000.0, 1)
                rec["evidence_recall"] = evidence_recall_from_texts(q.evidence, turn_text, bundle_texts(ctx))
                rec["context"] = summarize_bundle(ctx)
            t2 = time.perf_counter()
            answer = client.ask(q.question, session_id=f"locomo10_{conv.sample_id}_eval")
            rec["latency_ms"] = round((time.perf_counter() - t2) * 1000.0, 1)
            rec["answer"] = answer
            rec.update(score_answer(answer, q.answer, q.category))
            details.append(rec)
    return {
        "suite": "locomo10",
        "dataset": {"conversations": [c.sample_id for c in convs], "options": opts,
                    "turn_metadata": turn_metadata,
                    "turns": sum(1 for c in convs for _ in c.turns())},
        **aggregate(details),
        "ingest_seconds": round(ingest_s, 1),
        "avg_latency_ms": round(sum(d["latency_ms"] for d in details) / max(1, len(details)), 1),
        "details": details,
    }

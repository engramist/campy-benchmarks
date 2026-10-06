"""
campy-benchmarks / dmr / scoring.py
DMR has no official scorer: MemGPT reported an LLM judge's accuracy, and
later work (Zep) reports the same metric. Here, per question:

  judge     LoCoMo-10's judge (locomo10/scoring.py, this harness's prompt):
            CORRECT/WRONG against the gold answer; a non-answer is WRONG
            without a call
  f1        token F1 against the gold answer (LoCoMo's normalization)
  evidence_recall
            the derived evidence turns (dmr/dataset.py) that
            compile_context surfaced; None when the answer is paraphrased
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from locomo10.scoring import _token_f1, judge_one
from qa_judge import is_non_answer


def judge_details(details: List[Dict[str, Any]], judge_llm, log=print) -> int:
    n = 0
    for d in details:
        if d.get("judge") is not None:
            continue
        if not d["answer"].strip() or is_non_answer(d["answer"]):
            d.update({"judge": False, "reason": "non_answer"})
        else:
            d.update(judge_one(judge_llm, d["question"], d["expected"], d["answer"]))
            n += 1
    return n


def _mean(xs: List[float]) -> Optional[float]:
    return round(sum(xs) / len(xs), 4) if xs else None


def f1(answer: str, gold: str) -> float:
    return round(_token_f1(answer, gold), 4)


def aggregate(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    judged = [d for d in details if d.get("judge") is not None]
    rec = [d["evidence_recall"] for d in details if d.get("evidence_recall") is not None]
    return {
        "questions": len(details),
        "judge_accuracy": _mean([1.0 if d["judge"] else 0.0 for d in judged]),
        "f1": _mean([d["f1"] for d in details if d.get("f1") is not None]),
        "evidence_recall": _mean(rec),
        "evidence_labelled": len(rec),
    }

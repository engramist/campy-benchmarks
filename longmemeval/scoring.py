"""
campy-benchmarks / longmemeval / scoring.py
LongMemEval's own answer check: an LLM judge with one prompt per question
type, copied verbatim from the official src/evaluation/evaluate_qa.py
(get_anscheck_prompt). The paper uses GPT-4o; here the judge is whichever
local model --judge-model names, so numbers are comparable across runs of
this harness with the same judge, not directly with published tables.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from qa_judge import is_non_answer

_QA = ("I will give you a question, a correct answer, and a response from a model. Please answer yes "
       "if the response contains the correct answer. Otherwise, answer no. If the response is equivalent "
       "to the correct answer or contains all the intermediate steps to get the correct answer, you "
       "should also answer yes. If the response only contains a subset of the information required by "
       "the answer, answer no. ")
TEMPLATES = {
    "default": _QA + "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model "
               "response correct? Answer yes or no only.",
    "temporal-reasoning": _QA + "In addition, do not penalize off-by-one errors for the number of days. "
               "If the question asks for the number of days/weeks/months, etc., and the model makes "
               "off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response "
               "is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the "
               "model response correct? Answer yes or no only.",
    "knowledge-update": "I will give you a question, a correct answer, and a response from a model. "
               "Please answer yes if the response contains the correct answer. Otherwise, answer no. If "
               "the response contains some previous information along with an updated answer, the "
               "response should be considered as correct as long as the updated answer is the required "
               "answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model "
               "response correct? Answer yes or no only.",
    "single-session-preference": "I will give you a question, a rubric for desired personalized "
               "response, and a response from a model. Please answer yes if the response satisfies the "
               "desired response. Otherwise, answer no. The model does not need to reflect all the points "
               "in the rubric. The response is correct as long as it recalls and utilizes the user's "
               "personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
               "Is the model response correct? Answer yes or no only.",
    "abstention": "I will give you an unanswerable question, an explanation, and a response from a model. "
               "Please answer yes if the model correctly identifies the question as unanswerable. The "
               "model could say that the information is incomplete, or some other information is given "
               "but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: "
               "{}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only.",
}


def anscheck_prompt(question_type: str, question: str, answer: str, response: str,
                    abstention: bool = False) -> str:
    if abstention:
        key = "abstention"
    elif question_type in ("single-session-user", "single-session-assistant", "multi-session"):
        key = "default"
    elif question_type in TEMPLATES:
        key = question_type
    else:
        raise ValueError(f"unknown LongMemEval question type {question_type!r}")
    return TEMPLATES[key].format(question, answer, response)


def judge_record(judge_llm, rec: Dict[str, Any]) -> Dict[str, Any]:
    """One verdict. A non-answer is wrong without a call -- except for an
    abstention question, where declining is the point."""
    answer = rec.get("answer") or ""
    if not rec["abstention"] and (not answer.strip() or is_non_answer(answer)):
        return {"judge": False, "judge_reason": "non_answer", "called": False}
    prompt = anscheck_prompt(rec["question_type"], rec["question"], rec["expected"], answer,
                             abstention=rec["abstention"])
    res = judge_llm.chat([{"role": "user", "content": prompt}])
    text = re.sub(r"<think>.*?</think>", "", res["text"], flags=re.S)
    ok = "yes" in text.lower()  # the official rule
    return {"judge": ok, "judge_reason": "judge_yes" if ok else "judge_no", "called": True}


def judge_details(details: List[Dict[str, Any]], judge_llm) -> int:
    calls = 0
    for d in details:
        if d.get("judge") is not None:
            continue
        v = judge_record(judge_llm, d)
        calls += v.pop("called")
        d.update(v)
    return calls


def _mean(xs: List[float]) -> Optional[float]:
    return round(sum(xs) / len(xs), 4) if xs else None


def aggregate(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Accuracy (the official headline), per category (6 types + abstention),
    the task-averaged accuracy, and turn-level evidence recall."""
    judged = [d for d in details if d.get("judge") is not None]
    by_cat: Dict[str, Dict[str, Any]] = {}
    for cat in sorted({d["category"] for d in details}):
        rows = [d for d in judged if d["category"] == cat]
        rec = [d["evidence_recall"] for d in details
               if d["category"] == cat and d.get("evidence_recall") is not None]
        by_cat[cat] = {"n": sum(1 for d in details if d["category"] == cat),
                       "accuracy": _mean([1.0 if d["judge"] else 0.0 for d in rows]),
                       "evidence_recall": _mean(rec)}
    per_cat = [v["accuracy"] for v in by_cat.values() if v["accuracy"] is not None]
    return {
        "questions": len(details),
        "accuracy": _mean([1.0 if d["judge"] else 0.0 for d in judged]),
        "task_averaged_accuracy": _mean(per_cat),
        "evidence_recall": _mean([d["evidence_recall"] for d in details
                                  if d.get("evidence_recall") is not None]),
        # questions whose evidence Campy could store only in part (see INGEST_MAX_CHARS)
        "evidence_over_ingest_limit": sum(1 for d in details if d.get("evidence_over_ingest_limit")),
        "by_category": by_cat,
    }


INGEST_MAX_CHARS = 4000  # hippocampy config ingestion.max_ingest_chars: notify_turn cuts longer turns
_KEY_CHARS = 300


def _norm_ws(t: str) -> str:
    return " ".join((t or "").split())


def evidence_recall(evidence: List[str], turn_text: Dict[str, str],
                    retrieved_texts: List[str]) -> Optional[float]:
    """Fraction of the evidence turns found among the retrieved bundle items.

    LongMemEval turns (assistant ones especially) often exceed what Campy
    stores per turn, so an item holds at most the turn's first
    INGEST_MAX_CHARS. A turn counts as retrieved when its opening
    (_KEY_CHARS, normalized) is inside an item, or >= 80% of the words of its
    stored part are in one item. None when no evidence id is known."""
    known = [e for e in evidence if e in turn_text]
    if not known:
        return None
    items = [_norm_ws(t) for t in retrieved_texts]
    item_words = [set(re.findall(r"\w+", t)) for t in items]
    hits = 0
    for e in known:
        tt = _norm_ws(turn_text[e])
        key = tt[:_KEY_CHARS]
        words = set(re.findall(r"\w+", tt[:INGEST_MAX_CHARS]))
        if any(key in it for it in items) or (
            len(words) >= 3 and any(len(words & iw) / len(words) >= 0.8 for iw in item_words)
        ):
            hits += 1
    return round(hits / len(known), 4)

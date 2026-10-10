"""
campy-benchmarks / locomo10 / scoring.py
Scoring for the published LoCoMo QA task. Written from the behaviour of the
official evaluation (snap-research/locomo task_eval/evaluation.py, CC BY-NC
4.0); none of its code is copied here.

Per question:
  f1          LoCoMo's token F1: lowercase, drop commas and punctuation, drop
              a/an/the/and, Porter-stem (only when nltk is installed; recorded
              as f1_stemmed), then token-overlap F1. Category 1 (multi-hop)
              splits prediction and gold on commas and averages, per gold
              part, its best match. Category 3 scores against the gold text
              before the first ';'. Category 5 has no F1.
  abstained   Category 5 (adversarial: the asked-about thing never happened).
              Correct behaviour is to decline. `abstained_strict` is the
              official rule (answer contains "no information available" or
              "not mentioned"); `abstained` also accepts any non-answer
              scoring.is_non_answer recognises ("No relevant context was
              found in memory"), which is how Campy declines.
  judge       An LLM grades categories 1-4 CORRECT/WRONG against the gold
              answer (the "J" metric memory-system evaluations report). The
              judge prompt below is this harness's own, so J here is
              comparable across systems scored by this harness with the same
              judge model -- not directly with published numbers from other
              judge prompts or models.

`passed` = judge verdict (categories 1-4) or `abstained` (category 5);
None until the judge has run.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional

from locomo10.dataset import CATEGORY_NAMES
from scoring import is_non_answer

try:  # optional: the official F1 stems; without nltk it doesn't
    from nltk.stem import PorterStemmer
    _stem = PorterStemmer().stem
    STEMMED = True
except Exception:  # pragma: no cover - depends on environment
    _stem = lambda w: w  # noqa: E731
    STEMMED = False

_ARTICLES = re.compile(r"\b(a|an|the|and)\b")
_PUNCT = set(string.punctuation)


def normalize(s: str) -> str:
    s = s.replace(",", "").lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _token_f1(pred: str, gold: str) -> float:
    p = [_stem(w) for w in normalize(pred).split()]
    g = [_stem(w) for w in normalize(gold).split()]
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def locomo_f1(pred: str, gold: str, category: int) -> Optional[float]:
    if category == 5:
        return None
    if category == 3:
        gold = gold.split(";")[0].strip()
    if category == 1:
        preds = [x.strip() for x in pred.split(",")]
        golds = [x.strip() for x in gold.split(",")]
        return sum(max(_token_f1(p, g) for p in preds) for g in golds) / len(golds)
    return _token_f1(pred, gold)


def abstention(answer: str) -> Dict[str, bool]:
    low = answer.lower()
    strict = "no information available" in low or "not mentioned" in low
    return {"abstained_strict": strict, "abstained": strict or is_non_answer(answer)}


def score_answer(answer: str, gold: str, category: int) -> Dict[str, Any]:
    """Judge-independent fields; judge_details() fills judge/passed later."""
    rec: Dict[str, Any] = {"f1": None, "judge": None, "passed": None}
    f1 = locomo_f1(answer, gold, category)
    rec["f1"] = round(f1, 4) if f1 is not None else None
    if category == 5:
        rec.update(abstention(answer))
        rec["passed"] = rec["abstained"]
        rec["reason"] = "abstained" if rec["abstained"] else "answered_adversarial"
    return rec


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = "You grade answers to questions about a long conversation between two people. Reply with one word."

JUDGE_TEMPLATE = """Question: {question}
Gold answer: {gold}
Generated answer: {answer}

Is the generated answer CORRECT or WRONG?

It is CORRECT if it states the key information in the gold answer. It may be
longer, worded differently, or add context, as long as nothing it says
contradicts the gold answer. A date or time is CORRECT if it refers to the
same date or period as the gold answer, in any format, including a relative
form that resolves to it. For a list, it must contain the gold items.

It is WRONG if it gives different or contradicting facts, only partly
answers, hedges between alternatives, or says the information is unavailable.

Reply with exactly one word: CORRECT or WRONG."""

_VERDICT = re.compile(r"\b(CORRECT|WRONG)\b")

# Added to the prompt for single-persona questions (DMR), where the question is
# put to the assistant about the user's own conversation. The gold answer is
# written in whichever person the dataset author chose, so "I used to ..." vs
# "You used to ..." is the same fact. Not used for LoCoMo-10, whose two named
# speakers make "you" ambiguous there.
PERSONA_RULE = """The question, the gold answer and the generated answer may be
written in different grammatical person: "I"/"my" and "you"/"your" can be
swapped between them, because the answer is about the speaker being asked. A
first/second-person flip is not an error; judge the facts, not the pronoun."""

# Field orders for the prompt. Variant 0 is the byte-identical original prompt;
# the others only reorder the three labelled fields (the labels stay explicit),
# which is what lets repeated votes be independent samples at temperature 0
# instead of the same request answered the same way.
_FIELD_ORDERS = (
    ("question", "gold", "answer"),
    ("answer", "gold", "question"),
    ("gold", "question", "answer"),
)
_LABELS = {"question": "Question", "gold": "Gold answer", "answer": "Generated answer"}


def build_judge_prompt(question: str, gold: str, answer: str, persona: bool = False,
                       variant: int = 0) -> str:
    vals = {"question": question, "gold": gold, "answer": answer}
    if variant == 0 and not persona:
        return JUDGE_TEMPLATE.format(question=question, gold=gold, answer=answer)
    _head, _, rest = JUDGE_TEMPLATE.partition("\n\n")
    head = "\n".join(f"{_LABELS[k]}: {vals[k]}" for k in _FIELD_ORDERS[variant % len(_FIELD_ORDERS)])
    if persona:  # the rule goes just before the one-word reply instruction
        body, _, tail = rest.rpartition("\n\n")
        rest = f"{body}\n\n{PERSONA_RULE}\n\n{tail}"
    return f"{head}\n\n{rest}"


JUDGE_MAX_TOKENS = 256  # the verdict is one word; a cap stops a degenerate generation
JUDGE_ATTEMPTS = 2       # one retry, then the verdict is recorded as a judge error


def _judge_call(judge_llm, question: str, gold: str, answer: str, persona: bool, variant: int) -> Dict[str, Any]:
    from llm_client import LLMError
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": build_judge_prompt(question, gold, answer, persona, variant)},
    ]
    err = None
    for _ in range(JUDGE_ATTEMPTS):
        try:
            try:
                res = judge_llm.chat(messages, max_tokens=JUDGE_MAX_TOKENS)
            except TypeError:  # a judge without the max_tokens argument (test fakes)
                res = judge_llm.chat(messages)
            break
        except LLMError as e:  # e.g. Ollama aborting a repeating generation (HTTP 500)
            err = e
    else:
        # one bad call must not end a run: WRONG, marked so it can be counted and re-judged
        return {"judge": False, "judge_raw": str(err)[:200], "reason": "judge_error"}
    m = _VERDICT.search(res["text"].upper())
    if not m:
        return {"judge": False, "judge_raw": res["text"][:200], "reason": "judge_unparseable"}
    ok = m.group(1) == "CORRECT"
    return {"judge": ok, "reason": "judge_correct" if ok else "judge_wrong"}


def judge_one(judge_llm, question: str, gold: str, answer: str, persona: bool = False,
              votes: int = 1) -> Dict[str, Any]:
    """One verdict. votes=1 is a single call (the default, unchanged). votes=3
    judges twice -- with the fields in two different orders, so the second
    call is a genuinely different request -- and, only if those disagree, a
    third (a third order) decides. Any other `votes` is rejected."""
    if votes not in (1, 3):
        raise ValueError(f"judge votes must be 1 or 3, got {votes}")
    first = _judge_call(judge_llm, question, gold, answer, persona, 0)
    if votes == 1:
        return first
    second = _judge_call(judge_llm, question, gold, answer, persona, 1)
    cast = [first["judge"], second["judge"]]
    if cast[0] == cast[1]:
        out = dict(first)
        out["judge_votes"] = cast
        return out
    third = _judge_call(judge_llm, question, gold, answer, persona, 2)
    cast.append(third["judge"])
    out = dict(third)
    out["judge_votes"] = cast
    out["judge_tiebreak"] = True
    return out


def judge_details(details: List[Dict[str, Any]], judge_llm, log=print, votes: int = 1) -> int:
    """Judge every category 1-4 record in place; returns the number judged.
    Records with an empty answer are WRONG without a call."""
    n = 0
    for d in details:
        if d["category"] == 5 or d.get("judge") is not None:
            continue
        if not d["answer"].strip() or is_non_answer(d["answer"]):
            d.update({"judge": False, "reason": "non_answer"})
        else:
            d.update(judge_one(judge_llm, d["raw_question"], d["expected"], d["answer"], votes=votes))
            n += 1
        d["passed"] = d["judge"]
    return n


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean(xs: List[float]) -> Optional[float]:
    return round(sum(xs) / len(xs), 4) if xs else None


def aggregate(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    qa = [d for d in details if d["category"] != 5]
    adv = [d for d in details if d["category"] == 5]
    judged = [d for d in qa if d.get("judge") is not None]
    recall = [d["evidence_recall"] for d in details if d.get("evidence_recall") is not None]
    out: Dict[str, Any] = {
        "questions": len(details),
        # headline: LLM-judge accuracy on categories 1-4
        "judge_accuracy": _mean([float(d["judge"]) for d in judged]) if len(judged) == len(qa) and qa else None,
        "f1": _mean([d["f1"] for d in qa if d.get("f1") is not None]),
        "f1_stemmed": STEMMED,
        "adversarial_abstention": _mean([float(d["abstained"]) for d in adv]),
        "adversarial_abstention_strict": _mean([float(d["abstained_strict"]) for d in adv]),
        "evidence_recall": _mean(recall),
        "by_category": {},
    }
    for cat, name in CATEGORY_NAMES.items():
        ds = [d for d in details if d["category"] == cat]
        if not ds:
            continue
        row: Dict[str, Any] = {"n": len(ds)}
        if cat == 5:
            row["abstention"] = _mean([float(d["abstained"]) for d in ds])
        else:
            j = [float(d["judge"]) for d in ds if d.get("judge") is not None]
            row["judge_accuracy"] = _mean(j) if len(j) == len(ds) else None
            row["f1"] = _mean([d["f1"] for d in ds if d.get("f1") is not None])
        r = [d["evidence_recall"] for d in ds if d.get("evidence_recall") is not None]
        if r:
            row["evidence_recall"] = _mean(r)
        out["by_category"][name] = row
    return out


# ---------------------------------------------------------------------------
# Evidence recall: did retrieval surface the turns the gold answer cites?
# ---------------------------------------------------------------------------

def _norm_ws(s: str) -> str:
    return " ".join(s.lower().split())


def evidence_recall_from_texts(evidence: List[str], turn_text: Dict[str, str],
                               retrieved_texts: List[str]) -> Optional[float]:
    """Fraction of evidence turns whose text appears in the retrieved items
    (exact normalized substring, or >=80% of the turn's words inside one
    item). None when no evidence id resolves to a known turn."""
    known = [e for e in evidence if e in turn_text]
    if not known:
        return None
    items = [_norm_ws(t) for t in retrieved_texts]
    item_words = [set(re.findall(r"\w+", t)) for t in items]
    hits = 0
    for e in known:
        tt = _norm_ws(turn_text[e])
        words = set(re.findall(r"\w+", tt))
        if any(tt in it for it in items) or (
            len(words) >= 3 and any(len(words & iw) / len(words) >= 0.8 for iw in item_words)
        ):
            hits += 1
    return round(hits / len(known), 4)

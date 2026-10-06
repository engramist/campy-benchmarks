"""
campy-benchmarks / longmemeval / dataset.py
The published LongMemEval data (Wu et al., ICLR 2025;
https://github.com/xiaowu0162/LongMemEval), 500 questions.

Each question carries its OWN timestamped user-assistant chat history
(`haystack_sessions`, with `haystack_dates`), asked on `question_date`.
Question types: single-session-user, single-session-assistant,
single-session-preference, temporal-reasoning, knowledge-update,
multi-session; a question id ending in `_abs` is an abstention question
(the asked information is not in the history). Turns holding the evidence
carry `has_answer: true`.

Variants (the cleaned 2025/09 release):
  oracle  only the evidence sessions per question (cheap; tests memory
          mechanics, not finding the needle)
  s       ~40 sessions / ~115k tokens of history per question
  m       ~500 sessions per question (too large for a local run)

The files are on HuggingFace; LONGMEMEVAL_PATH overrides the location.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

BASE_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"
FILES = {"oracle": "longmemeval_oracle.json", "s": "longmemeval_s_cleaned.json",
         "m": "longmemeval_m_cleaned.json"}
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TYPES = ("single-session-user", "single-session-assistant", "single-session-preference",
         "temporal-reasoning", "knowledge-update", "multi-session")


class DatasetError(RuntimeError):
    pass


@dataclass
class LMETurn:
    role: str  # "user" | "assistant"
    content: str
    has_answer: bool = False


@dataclass
class LMESession:
    session_id: str
    date: str  # as released, e.g. "2023/05/20 (Sat) 02:21"
    turns: List[LMETurn]


@dataclass
class LMEQuestion:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    sessions: List[LMESession]  # oldest first
    answer_session_ids: List[str] = field(default_factory=list)

    @property
    def abstention(self) -> bool:
        return self.question_id.endswith("_abs")

    @property
    def category(self) -> str:
        """The reporting bucket: abstention questions are their own."""
        return "abstention" if self.abstention else self.question_type

    def evidence_ids(self) -> List[str]:
        return [f"{s.session_id}:{i}" for s in self.sessions for i, t in enumerate(s.turns) if t.has_answer]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dataset(variant: str = "oracle", log=print) -> Path:
    """A local copy of one variant, downloaded from HuggingFace if missing."""
    if variant not in FILES:
        raise DatasetError(f"unknown LongMemEval variant {variant!r} (one of {sorted(FILES)})")
    path = Path(os.environ.get("LONGMEMEVAL_PATH") or DATA_DIR / FILES[variant])
    if path.exists():
        return path
    url = BASE_URL + FILES[variant]
    log(f"[+] Downloading LongMemEval ({variant}) from {url} ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as e:  # noqa: BLE001 -- offline, proxy, HF down: say what to do
        raise DatasetError(f"could not download {url} ({e}); download it by hand and set "
                           f"LONGMEMEVAL_PATH") from e
    tmp.rename(path)
    return path


def _interleave_by_type(qs: List[LMEQuestion]) -> List[LMEQuestion]:
    """Round-robin over the reporting categories (6 types + abstention), each
    in dataset order, so a prefix of N questions covers every category."""
    by_cat: dict = {}
    for q in qs:
        by_cat.setdefault(q.category, []).append(q)
    out: List[LMEQuestion] = []
    while any(by_cat.values()):
        for cat in sorted(by_cat):
            if by_cat[cat]:
                out.append(by_cat[cat].pop(0))
    return out


def load_questions(path: Path, max_questions: Optional[int] = None,
                   types: Optional[List[str]] = None) -> List[LMEQuestion]:
    data = json.loads(Path(path).read_text())
    qs: List[LMEQuestion] = []
    for e in data:
        sessions = [
            LMESession(str(sid), str(date), [LMETurn(t["role"], t["content"], bool(t.get("has_answer")))
                                             for t in sess])
            for sid, date, sess in zip(e["haystack_session_ids"], e["haystack_dates"], e["haystack_sessions"])
        ]
        # oracle files are not sorted by time; the released dates sort as text
        # ("YYYY/MM/DD (Day) HH:MM")
        sessions.sort(key=lambda s: s.date)
        q = LMEQuestion(str(e["question_id"]), e["question_type"], e["question"], str(e["answer"]),
                        str(e["question_date"]), sessions, [str(x) for x in e.get("answer_session_ids") or []])
        if types and q.category not in types:
            continue
        qs.append(q)
    qs = _interleave_by_type(qs)
    return qs[:max_questions] if max_questions else qs

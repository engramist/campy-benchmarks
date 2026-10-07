"""
campy-benchmarks / dmr / dataset.py
DMR, "Deep Memory Retrieval" (Packer et al., MemGPT, 2023): 500 questions,
each over its own Multi-Session Chat history (Xu et al., 2022). Released as
MemGPT/MSC-Self-Instruct on HuggingFace (Apache-2.0).

One record (JSONL line):
  previous_dialogs   4 earlier sessions: {dialog: [{text}], time_back, ...};
                     the turns have no speaker id and alternate, Speaker 1
                     first (the MSC convention)
  dialog             session 5, the current one -- not memory, not ingested
  self_instruct      {B: the question, A: the gold answer}; the question asks
                     the other speaker about something they said earlier
  personas, init_personas, personas_update*, summary_speaker_*
                     persona sentences and summaries; they hold the answer
                     verbatim, so they are never ingested

The dataset has no evidence labels. `evidence_ids()` marks the earlier turns
whose text contains the normalized gold answer: a derived, approximate label
(None when the answer is paraphrased). The same match picks the answering
speaker, who the question is addressed to; when no turn quotes the answer,
the speaker whose persona sentences share most words with it (personas are
never ingested), else Speaker 1.

The file is pinned by sha256; DMR_PATH points at an existing copy.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

URL = "https://huggingface.co/datasets/MemGPT/MSC-Self-Instruct/resolve/main/msc_self_instruct.jsonl"
SHA256 = "d3dbea36848b41dc46c0f1548d0ebf74eeaf6390d6f3fe9318e8480dc984495e"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "dmr" / "msc_self_instruct.jsonl"


class DatasetError(RuntimeError):
    pass


@dataclass
class DMRTurn:
    speaker: str  # "Speaker 1" | "Speaker 2"
    text: str


@dataclass
class DMRSession:
    index: int  # 1-4
    time_back: str  # e.g. "7 days 8 hours ago", relative to the current session
    turns: List[DMRTurn]


@dataclass
class DMRQuestion:
    question_id: str
    question: str
    answer: str
    sessions: List[DMRSession]
    answerer: Optional[str] = None  # the speaker the question is addressed to
    answerer_source: str = "turn"  # turn | persona | default (how `answerer` was found)
    evidence: List[str] = field(default_factory=list)  # "session:turn", derived

    def evidence_ids(self) -> List[str]:
        return list(self.evidence)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dataset(log=print, verify: bool = True) -> Path:
    path = Path(os.environ.get("DMR_PATH") or DATA_PATH)
    if not path.exists():
        log(f"[+] Downloading DMR from {URL} ...")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(URL, tmp)
        except Exception as e:  # noqa: BLE001 -- offline, proxy, HF down: say what to do
            raise DatasetError(f"could not download {URL} ({e}); download it by hand and set DMR_PATH") from e
        tmp.rename(path)
    if verify and not os.environ.get("DMR_PATH") and sha256(path) != SHA256:
        raise DatasetError(f"{path} does not match the pinned sha256 {SHA256[:16]}...; delete it to re-download")
    return path


_PUNCT = re.compile(r"[^\w\s]")


def norm(s: str) -> str:
    return " ".join(_PUNCT.sub(" ", (s or "").lower()).split())


def _locate_answer(sessions: List[DMRSession], answer: str):
    """The turns whose text contains the normalized answer, and the speaker
    who said most of them."""
    a = norm(answer)
    if len(a) < 3:
        return [], None
    hits = [(s.index, i, t.speaker) for s in sessions for i, t in enumerate(s.turns)
            if f" {a} " in f" {norm(t.text)} "]
    if not hits:
        return [], None
    by = {}
    for _, _, sp in hits:
        by[sp] = by.get(sp, 0) + 1
    answerer = max(sorted(by), key=lambda sp: by[sp])
    return [f"{si}:{ti}" for si, ti, sp in hits if sp == answerer], answerer


_WORDS = re.compile(r"[a-z0-9]+")
_STOP = set("i a an the my me to of and or in on at for is am are was it that this with".split())


def _persona_speaker(rec: dict, answer: str) -> Optional[str]:
    """The speaker whose persona sentences share the most words with the
    answer (the persona lists are never ingested; they only say who is
    asked). None when neither shares a content word."""
    want = set(_WORDS.findall(answer.lower())) - _STOP
    lists = rec.get("personas") or []
    best, best_n = None, 0
    for i, sentences in enumerate(lists[:2]):
        text = " ".join(str(x) for x in (sentences or [])).lower()
        n = len(want & set(_WORDS.findall(text)))
        if n > best_n:
            best, best_n = f"Speaker {i + 1}", n
    return best


def parse_record(rec: dict, n: int) -> DMRQuestion:
    sessions = []
    for k, sess in enumerate(rec["previous_dialogs"], 1):
        turns = [DMRTurn(f"Speaker {1 + i % 2}", str(t["text"])) for i, t in enumerate(sess["dialog"])]
        sessions.append(DMRSession(k, str(sess.get("time_back") or ""), turns))
    si = rec["self_instruct"]
    qid = str((rec.get("metadata") or {}).get("initial_data_id") or f"dmr_{n}")
    q = DMRQuestion(qid, str(si["B"]), str(si["A"]), sessions)
    q.evidence, q.answerer = _locate_answer(sessions, q.answer)
    if q.answerer is None:
        # no earlier turn quotes the answer: still tell the model who "you" is
        q.answerer = _persona_speaker(rec, q.answer)
        q.answerer_source = "persona" if q.answerer else "default"
        q.answerer = q.answerer or "Speaker 1"  # 21 of the 27 located answerers in R16
    return q


def load_questions(path: Path, max_questions: Optional[int] = None) -> List[DMRQuestion]:
    qs = []
    with open(path) as f:
        for n, line in enumerate(f):
            if line.strip():
                qs.append(parse_record(json.loads(line), n))
            if max_questions and len(qs) >= max_questions:
                break
    return qs

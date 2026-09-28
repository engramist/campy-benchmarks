"""
campy-benchmarks / locomo10 / dataset.py
Fetch, verify and iterate the published LoCoMo dataset.

LICENSE: the dataset (and the snap-research/locomo repo) is CC BY-NC 4.0 --
non-commercial use, with attribution. It is therefore NOT committed here:
it is downloaded at run time into data/ (gitignored), pinned to a commit and
a sha256 so every run scores the same bytes. Cite:
  Maharana et al., "Evaluating Very Long-Term Conversational Memory of LLM
  Agents", ACL 2024. https://github.com/snap-research/locomo
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

REPO = "https://github.com/snap-research/locomo"
COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
REL_PATH = "data/locomo10.json"
RAW_URL = f"https://raw.githubusercontent.com/snap-research/locomo/{COMMIT}/{REL_PATH}"
SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "locomo10.json"

# As named in the official evaluation code (task_eval/evaluation.py comments).
CATEGORY_NAMES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}

# LoCoMo's QA protocol appends this to temporal (category 2) questions. Its
# wording refers to its own "DATE:" prompt header; reworded here to fit a
# memory system's context. Applied identically to Campy and every baseline.
TEMPORAL_SUFFIX = " Use the dates of the conversation to answer with an approximate date."


class DatasetError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fetch_raw(dest: Path) -> None:
    with urllib.request.urlopen(RAW_URL, timeout=120) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def _fetch_git(dest: Path) -> None:
    """Fallback for networks that allow git but not raw.githubusercontent."""
    with tempfile.TemporaryDirectory() as tmp:
        run = lambda *a: subprocess.run(["git", *a], cwd=tmp, check=True,
                                        capture_output=True, text=True, timeout=300)
        run("init", "-q")
        run("remote", "add", "origin", REPO + ".git")
        run("fetch", "-q", "--depth", "1", "origin", COMMIT)
        run("checkout", "-q", "FETCH_HEAD", "--", REL_PATH)
        shutil.copyfile(Path(tmp) / REL_PATH, dest)


def ensure_dataset(path: Optional[Path] = None, log=print) -> Path:
    """Return a verified local copy, downloading it if needed.
    LOCOMO10_PATH overrides the location."""
    path = Path(os.environ.get("LOCOMO10_PATH") or path or DEFAULT_PATH)
    if path.exists():
        if _sha256(path) != SHA256:
            raise DatasetError(f"{path} does not match the pinned sha256 {SHA256[:12]}...; "
                               "delete it to re-download, or point LOCOMO10_PATH at the pinned file")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log(f"    downloading LoCoMo-10 ({REPO} @ {COMMIT[:8]}, CC BY-NC 4.0) -> {path}")
    tmp = path.with_suffix(".part")
    errors = []
    for fetch in (_fetch_raw, _fetch_git):
        try:
            fetch(tmp)
            if _sha256(tmp) != SHA256:
                raise DatasetError("downloaded file failed sha256 verification")
            tmp.replace(path)
            return path
        except Exception as e:
            errors.append(f"{fetch.__name__}: {type(e).__name__}: {e}")
            tmp.unlink(missing_ok=True)
    raise DatasetError("could not download LoCoMo-10: " + "; ".join(errors))


@dataclass
class L10Turn:
    dia_id: str
    session: int
    date_time: str
    speaker: str
    text: str
    image_caption: Optional[str] = None

    def content(self) -> str:
        """What gets written to memory: the session date and speaker are part
        of the text (notify_turn has no timestamp parameter, and temporal
        questions need the date)."""
        s = f"[{self.date_time}] {self.speaker}: {self.text}"
        if self.image_caption:
            s += f" [shares an image: {self.image_caption}]"
        return s


@dataclass
class L10Question:
    id: str
    question: str      # as asked (temporal suffix applied)
    raw_question: str
    answer: str        # gold; for adversarial, the adversarial (wrong) answer
    category: int
    evidence: List[str] = field(default_factory=list)


@dataclass
class L10Conversation:
    sample_id: str
    speakers: List[str]
    sessions: List[List[L10Turn]]
    questions: List[L10Question]

    def turns(self) -> Iterator[L10Turn]:
        for s in self.sessions:
            yield from s


def _interleave_by_category(qs: List[L10Question]) -> List[L10Question]:
    """The dataset lists questions grouped by category, so a plain prefix of
    N questions would cover only one or two categories. Take them
    round-robin across categories (each in dataset order) instead."""
    by_cat: Dict[int, List[L10Question]] = {}
    for q in qs:
        by_cat.setdefault(q.category, []).append(q)
    out: List[L10Question] = []
    queues = [by_cat[c] for c in sorted(by_cat)]
    i = 0
    while any(i < len(q) for q in queues):
        out.extend(q[i] for q in queues if i < len(q))
        i += 1
    return out


def load_conversations(path: Path, conversations: Optional[int] = None,
                       max_questions: Optional[int] = None,
                       categories: Optional[List[int]] = None) -> List[L10Conversation]:
    data = json.loads(Path(path).read_text())
    out: List[L10Conversation] = []
    for sample in data[: conversations or len(data)]:
        conv = sample["conversation"]
        sessions: List[List[L10Turn]] = []
        n = 1
        while f"session_{n}" in conv:
            date = conv.get(f"session_{n}_date_time", "")
            sessions.append([
                L10Turn(t["dia_id"], n, date, t["speaker"], t["text"], t.get("blip_caption"))
                for t in conv[f"session_{n}"]
            ])
            n += 1
        qs: List[L10Question] = []
        for i, qa in enumerate(sample["qa"]):
            cat = int(qa["category"])
            if categories and cat not in categories:
                continue
            q = qa["question"] + (TEMPORAL_SUFFIX if cat == 2 else "")
            gold = str(qa.get("adversarial_answer") if cat == 5 else qa.get("answer", ""))
            qs.append(L10Question(f"{sample['sample_id']}_q{i}", q, qa["question"], gold, cat,
                                  list(qa.get("evidence") or [])))
        if max_questions is not None:
            qs = _interleave_by_category(qs)[:max_questions]
        out.append(L10Conversation(sample["sample_id"], [conv.get("speaker_a"), conv.get("speaker_b")],
                                   sessions, qs))
    return out

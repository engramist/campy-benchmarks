#!/usr/bin/env python3
"""
campy-benchmarks / check_locomo10.py

Offline self-test for the LoCoMo-10 suite (no daemon, no LLM):

  * the downloaded dataset matches the pinned sha256 and has the published
    shape (10 conversations, 5,882 turns, 1,986 questions);
  * locomo10.scoring's F1 follows the official rules on hand-computed cases
    (it was also cross-checked against the official implementation on 7,418
    pairs with 0 mismatches when this suite was written);
  * adversarial abstention accepts declines and rejects answers;
  * evidence recall matches retrieved text back to the cited turns;
  * question subsets are balanced across categories.

    python3 check_locomo10.py      # downloads the dataset on first run

Exits 0 and prints "OK" on success, 1 listing each failure.
"""

from __future__ import annotations

import collections
import sys

from locomo10.dataset import DatasetError, ensure_dataset, load_conversations
from locomo10.scoring import STEMMED, abstention, evidence_recall_from_texts, locomo_f1


def main() -> int:
    fails = []

    def check(name, cond, detail=""):
        if not cond:
            fails.append(f"{name} {detail}".rstrip())

    path = ensure_dataset()
    convs = load_conversations(path)
    check("10 conversations", len(convs) == 10, str(len(convs)))
    check("5882 turns", sum(1 for c in convs for _ in c.turns()) == 5882)
    check("1986 questions", sum(len(c.questions) for c in convs) == 1986)
    cats = collections.Counter(q.category for c in convs for q in c.questions)
    check("category counts", cats == {4: 841, 5: 446, 2: 321, 1: 282, 3: 96}, str(dict(cats)))
    check("temporal suffix only on cat 2",
          all((q.question != q.raw_question) == (q.category == 2) for c in convs for q in c.questions))

    sub = load_conversations(path, conversations=1, max_questions=25)[0].questions
    check("balanced subset", collections.Counter(q.category for q in sub) == {1: 5, 2: 5, 3: 5, 4: 5, 5: 5})

    # M0.2: held-out selection by sample_id
    second = load_conversations(path, conversation_ids=["conv-30"])
    check("conversation_ids picks by sample_id", [c.sample_id for c in second] == ["conv-30"],
          str([c.sample_id for c in second]))
    check("conv-30 is the second conversation", convs[1].sample_id == "conv-30")
    check("ids honour the order given and max_questions",
          [c.sample_id for c in load_conversations(path, conversation_ids=["conv-30", "conv-26"], max_questions=5)]
          == ["conv-30", "conv-26"]
          and all(len(c.questions) == 5 for c in load_conversations(path, conversation_ids=["conv-30"], max_questions=5)))
    try:
        load_conversations(path, conversation_ids=["conv-nope"])
        check("unknown id raises", False)
    except DatasetError:
        pass
    from locomo10.runner import locomo10_options
    from types import SimpleNamespace as NS
    o = locomo10_options(NS(locomo10_conversation_ids=["conv-30"], locomo10_max_questions=60), False)
    check("run_all options carry the ids", o.get("conversation_ids") == ["conv-30"], str(o))
    check("no ids: options unchanged", "conversation_ids" not in locomo10_options(NS(), False))

    # F1 (unstemmed values; with nltk installed stemming can only raise overlap here)
    def close(a, b):
        return a is not None and abs(a - b) < 1e-4
    check("f1 exact", close(locomo_f1("mental health", "mental health", 4), 1.0))
    check("f1 articles/punct ignored", close(locomo_f1("The mental health.", "mental health", 4), 1.0))
    # official normalization deletes punctuation without a space: "mental-health" -> "mentalhealth"
    check("f1 hyphen joins words", close(locomo_f1("mental-health", "mental health", 4), 0.0))
    check("f1 partial", close(locomo_f1("it raised awareness for mental health", "mental health", 4), 2 * (2/6) * 1 / (2/6 + 1)))
    check("f1 cat3 uses text before ;", close(locomo_f1("psychology", "Psychology; counseling", 3), 1.0))
    check("f1 cat1 averages comma parts", close(locomo_f1("pottery", "pottery, camping", 1), 0.5))
    check("f1 none for adversarial", locomo_f1("anything", "x", 5) is None)

    # abstention
    for a in ("Not mentioned in the conversation.", "No information available.",
              "No relevant context was found in memory.", "I don't know."):
        check("abstains", abstention(a)["abstained"], repr(a))
    check("strict rule is the official phrase set", not abstention("I don't know.")["abstained_strict"])
    for a in ("She realized self-care is important.", "Caroline realized she loves running."):
        check("answer is not abstention", not abstention(a)["abstained"], repr(a))

    # evidence recall
    turns = {"D1:1": "[date] Caroline: I went to the LGBTQ support group yesterday.",
             "D1:2": "[date] Melanie: That sounds great, how was it?"}
    check("recall exact", evidence_recall_from_texts(["D1:1"], turns, ["xx [DATE] caroline: i went to the lgbtq support group yesterday. yy"]) == 1.0)
    check("recall partial", evidence_recall_from_texts(["D1:1", "D1:2"], turns, [turns["D1:1"]]) == 0.5)
    check("recall miss", evidence_recall_from_texts(["D1:2"], turns, ["unrelated item"]) == 0.0)
    check("recall unknown ids -> None", evidence_recall_from_texts(["D9:9"], turns, ["x"]) is None)

    if fails:
        print(f"FAIL -- {len(fails)} LoCoMo-10 checks failed:")
        for f in fails:
            print(f"  {f}")
        return 1
    print(f"OK -- LoCoMo-10 dataset and scoring checks passed (F1 stemming: {STEMMED})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

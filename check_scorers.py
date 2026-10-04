#!/usr/bin/env python3
"""
campy-benchmarks / check_scorers.py

Scorer self-test: a benchmark whose judge rejects correct answers or accepts
non-answers measures the judge, not the memory. For every LoCoMo and MemBench
probe this asserts:

  * the probe's own gold answer passes,
  * natural, correct LLM-style paraphrases pass (including ones that mention
    the superseded value while marking it superseded),
  * non-answers ("I don't know", the daemon's "No relevant context..." reply)
    fail,
  * wrong answers fail (the stale value stated as current, a stale value
    listed alongside the current one with no supersession cue, and "yes" to a
    must-be-"no" question).

Scorer v1 failed 2 gold answers, passed "I don't know" on 2 probes, and failed
nearly every natural paraphrase that mentioned history -- this script is the
regression gate for that. Plain script, same convention as
check_suite_order_independence.py:

    python3 check_scorers.py

Exits 0 and prints "OK" on success, exits 1 listing each failure.
"""

from __future__ import annotations

import sys
from typing import Dict, List

from locomo.dataset import get_locomo_scenarios
from membench.msc_dataset import get_msc_personas
from scoring import judge

NON_ANSWERS = [
    "I don't know.",
    "I do not know what we use for that.",
    "No relevant context was found in memory.",
    "I'm not sure.",
    "There is no information about this in memory.",
    "",
]

# Hand-written probes: correct answers phrased the way an LLM actually answers.
CORRECT: Dict[str, List[str]] = {
    "p1_current_db": [
        "Your active database is PostgreSQL 16; you migrated off PostgreSQL 14.",
        "PostgreSQL 16 (PostgreSQL 14 is deprecated).",
        "Postgres 16.",
    ],
    "p2_deprecated_check": [
        "No. PostgreSQL 14 is deprecated; deploy on PostgreSQL 16 instead.",
        "You cannot -- all new deployments must target PostgreSQL 16.",
    ],
    "p3_analytics_pk": [
        "Analytics tables use BIGINT sequence IDs instead of UUIDs to avoid index bloat.",
        "BIGINT sequence IDs.",
    ],
    "p4_auth_algo": [
        "Tokens must be signed with RS256 using asymmetric keys rotated via JWKS.",
        "RS256 -- HS256 is forbidden.",
        "RS256 only; HS256 is not available for internal APIs.",
    ],
    "p5_hs256_rejection": [
        "No, HS256 is forbidden; only RS256 via JWKS is accepted.",
        "HS256 tokens are not acceptable -- HS256 was deprecated after the security audit.",
    ],
    "p6_cache_engine": [
        "Memcached was retired; Redis on port 6379 is the active cache and lock store.",
        "Redis cluster on port 6379.",
    ],
    "p_alex_diet": [
        "Alex is currently pescatarian; they are no longer vegan.",
        "Pescatarian -- Alex switched from a vegan diet after a doctor visit.",
    ],
    "p_alex_city": ["Alex lives in Seattle."],
    "p_jordan_editor": [
        "Jordan now uses VS Code with Cursor, having stopped using Vim.",
        "VS Code with Cursor.",
    ],
    "p_taylor_3": ["Taylor's favorite coffee is now cold brew with cinnamon, no longer an oat milk latte."],
    "p_morgan_4": ["Olympic weightlifting; Morgan switched from marathon running."],
    "p_casey_5": ["Casey prefers Google Cloud Run serverless now instead of AWS ECS."],
}

# Real answers (llama3.1:8b via `ask`) that a scorer version wrongly failed.
# Checked in addition to each probe's other cases.
REAL: Dict[str, List[str]] = {
    # 2026-09-30, scorer v2: "prohibits" was not a supersession cue.
    "p_6_active": [
        "The current required tool for logging_backend is Vector to OpenSearch. This is "
        "based on the final decision made on 2026-09-30 20:29, which strictly requires the "
        "use of Vector to OpenSearch for logging_backend, and prohibits the use of Local "
        "Syslog or Logstash.",
    ],
    # 2026-10-01, scorer v3: "updates ... from X to Y" was not a supersession cue.
    "p_casey_5": [
        "Based on the provided context, Casey's current cloud preference is Google Cloud Run "
        "serverless. This is stated in the conversation log, where Casey updates their "
        "preference from AWS ECS to Google Cloud Run serverless on 2026-10-01 04:58.",
        "Based on the provided context, Casey's current cloud preference is Google Cloud Run "
        "serverless. This is stated in the conversation log where Casey updates their "
        "preference from AWS ECS to Google Cloud Run serverless.",
    ],
    # 2026-10-04, lexical: grounded in "not accept ... under any circumstances",
    # none of the probe's grounding words (B459 baseline, run 1 of 3).
    "p5_hs256_rejection": [
        "Based on the provided memory context, I can see that there is a constraint to not "
        "accept HS256 tokens under any circumstances, as stated in the final rollout message. "
        "Therefore, an HS256 signed token is not acceptable for internal APIs.",
    ],
    # 2026-10-02, LLM judge (gemma4:26b): failed for leaving out Cursor while the
    # gold read "VS Code with Cursor"; correct, and Vim is retired.
    "p_jordan_editor": [
        "Based on the provided context, I can see that Jordan mentioned they stopped using "
        "Vim and fully transitioned to VS Code. Therefore, the primary code editor that "
        "Jordan uses is VS Code.",
    ],
    "p_8_active": [
        "The current required tool for ci_provider is GitHub Actions. This is based on the "
        "final decision made on 2026-09-30, which strictly requires the use of GitHub Actions "
        "for ci_provider, and prohibits the use of Jenkins or Travis CI.",
    ],
}

# Hand-written probes: answers that must fail (beyond the shared non-answers).
WRONG: Dict[str, List[str]] = {
    "p1_current_db": ["The active database is PostgreSQL 14.", "PostgreSQL 14 and PostgreSQL 16."],
    "p2_deprecated_check": ["Yes, you can deploy it on PostgreSQL 14.", "No."],
    "p3_analytics_pk": ["Analytics tables use UUID primary keys."],
    "p4_auth_algo": ["HS256 with a symmetric secret.", "HS256 or RS256."],
    "p5_hs256_rejection": ["Yes, HS256 tokens are acceptable internally, no problem.", "No."],
    "p6_cache_engine": ["Memcached on port 11211.", "Memcached and Redis."],
    "p_alex_diet": ["Alex follows a strict vegan diet.", "Vegan and pescatarian."],
    "p_alex_city": ["Alex lives in Portland."],
    "p_jordan_editor": ["Jordan uses Vim.", "Vim and Cursor."],
    "p_taylor_3": ["An oat milk latte."],
    "p_morgan_4": ["Marathon running."],
    "p_casey_5": ["AWS ECS."],
}


def labelled_cases():
    """Every hand-labelled (probe id, probe, gold, question, answer, should_pass)
    case: the contract check_scorers.py holds the lexical scorer to, and
    check_qa_judge.py measures an LLM judge against. A probe with no cases
    yields (pid, probe, gold, question, None, None)."""
    probes = []
    for sc in get_locomo_scenarios(smoke=False):
        for p in sc.probes:
            probes.append((p.id, p, p.expected, sc))
    for persona in get_msc_personas(smoke=False):
        for p in persona.probes:
            probes.append((p.id, p, p.expected_active, None))

    for pid, probe, gold, sc in probes:
        base = (pid, probe, gold, probe.question)
        if probe.kind == "value":
            yield (*base, gold, True)  # the gold answer itself
        for a in NON_ANSWERS:
            yield (*base, a, False)
        for a in REAL.get(pid, []):
            yield (*base, a, True)

        if pid in CORRECT:
            for a in CORRECT[pid]:
                yield (*base, a, True)
            for a in WRONG.get(pid, []):
                yield (*base, a, False)
            continue

        # Templated LoCoMo topic: pull v1/v2/v3 from the scenario's own turns.
        if sc is None:
            yield (*base, None, None)
            continue
        final_turn = sc.sessions[2][0]["content"]  # "... do NOT use {v1} or {v2}."
        v1, v2 = final_turn.rsplit("do NOT use ", 1)[1].rstrip(".").split(" or ")
        v3 = probe.expected
        domain = sc.id.split("_", 2)[2]
        for a in [
            f"{v3}.",
            f"The current required tool for {domain} is {v3}.",
            f"We use {v3} for {domain}; {v1} and {v2} are deprecated.",
            f"{domain} moved from {v1} to {v2} and has now been replaced by {v3}.",
            f"{v3} -- do not use {v1} or {v2}.",
            f"{v3} is required for {domain}; the use of {v1} or {v2} is prohibited.",
            f"We use {v3}, which prohibits the use of {v1} or {v2}.",
            f"{v3}; {v1} and {v2} are forbidden.",
            f"{v3}. The final decision forbids {v1} and {v2}.",
            f"{v3}; {v1} and {v2} are disallowed.",
            f"{v3} ({v1} and {v2} are banned).",
            f"The decision was updated from {v1} to {v2} and then to {v3}.",
            f"{domain} changed from {v2} to {v3}.",
        ]:
            yield (*base, a, True)
        for a in [
            f"{v1}.",
            f"We use {v2} for {domain}.",
            f"We use {v1} and {v3}.",
        ]:
            yield (*base, a, False)


def main() -> int:
    failures: List[str] = []
    checked = 0
    pids = set()
    for pid, probe, gold, question, answer, should_pass in labelled_cases():
        pids.add(pid)
        if answer is None:
            failures.append(f"{pid}: no self-test cases defined")
            continue
        checked += 1
        passed, reason = judge(answer, probe.kind, probe.accept, probe.stale)
        if passed != should_pass:
            want = "pass" if should_pass else "fail"
            failures.append(f"{pid}: expected {want}, got {reason}: {answer!r}")

    if failures:
        print(f"FAIL -- {len(failures)} of {checked} scorer checks failed:")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"OK -- {checked} scorer checks passed across {len(pids)} probes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

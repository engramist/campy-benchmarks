#!/usr/bin/env python3
"""
campy-benchmarks / check_suite_order_independence.py

B438 regression check: a suite's mocked result must not depend on which
other suites ran before it on the same CampyMCPClient. Runs MemBench in
isolation, then again after LoCoMo + MemoryGym on a fresh client with
reset_mock_state() called between suites (matching run_all.py's own
sequencing), and asserts the results match.

This repo has no pytest setup (see README/pyproject.toml) -- this is a
plain script, run directly:

    python3 check_suite_order_independence.py

Exits 0 and prints "OK" on success, exits 1 with a diff on failure.
"""

from __future__ import annotations

import sys

from mcp_client import CampyMCPClient
from locomo.runner import run_locomo
from memory_gym.runner import run_memory_gym
from membench.runner import run_membench


def run_membench_isolated() -> dict:
    client = CampyMCPClient(mcp_cmd=None)
    client.reset_mock_state()
    result = run_membench(client, smoke=True)
    client.close()
    return result


def run_membench_after_other_suites() -> dict:
    client = CampyMCPClient(mcp_cmd=None)
    client.reset_mock_state()
    run_locomo(client, smoke=True)
    client.reset_mock_state()
    run_memory_gym(client, smoke=True)
    client.reset_mock_state()
    result = run_membench(client, smoke=True)
    client.close()
    return result


def main() -> int:
    isolated = run_membench_isolated()
    after_others = run_membench_after_other_suites()

    # Compare only the fields that should be order-independent (latency
    # legitimately varies run to run and isn't part of this check).
    fields = [
        "fact_precision", "fact_recall", "contradiction_score",
        "raw_tokens_avg", "bundle_tokens_avg", "token_savings_pct",
    ]
    mismatches = [f for f in fields if isolated.get(f) != after_others.get(f)]

    if mismatches:
        print("FAIL — MemBench result depends on suite run order:")
        for f in mismatches:
            print(f"  {f}: isolated={isolated.get(f)!r} after_others={after_others.get(f)!r}")
        return 1

    print("OK — MemBench result is order-independent:")
    print(f"  token_savings_pct: {isolated['token_savings_pct']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

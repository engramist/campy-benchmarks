"""
campy-benchmarks / fresh_store.py
Runs one question of a per-question suite (LongMemEval, DMR) on its own
fresh store, so a daemon that dies mid-run costs that question, not the run.

`run_on_fresh_store(new_store, body)` calls `body(client)` inside
`new_store()`. When the daemon fails (CampyClientError: offline, crashed,
never became ready), it retries once on another fresh store; after a second
failure it returns the error text, and the caller records the question as
an error (scored wrong) and goes on. Any other exception is a harness bug
and propagates.
"""

from __future__ import annotations

from typing import Callable, ContextManager, Optional

from mcp_client import CampyClientError, CampyMCPClient


def run_on_fresh_store(new_store: Callable[[], ContextManager[CampyMCPClient]],
                       body: Callable[[CampyMCPClient], None], label: str, log=print) -> Optional[str]:
    err = ""
    for attempt in (1, 2):
        try:
            with new_store() as client:
                body(client)
            return None
        except CampyClientError as e:
            err = str(e)[:600]
            log(f"      !! {label}: {err} -- "
                f"{'retrying on a fresh store' if attempt == 1 else 'recorded as an error (scored wrong)'}")
    return err

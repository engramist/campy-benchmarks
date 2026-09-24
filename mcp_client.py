"""
campy-benchmarks / mcp_client.py
Standard Model Context Protocol (MCP) Client for HippoCampy.

Interacts with HippoCampy strictly over standard MCP STDIO transport (CAMPY_MCP_CMD),
preserving clean ecosystem separation per docs/ecosystem-rules.md.
Provides resilient fallback / mock capabilities when running in standalone smoke mode.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import queue
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional


class CampyClientError(RuntimeError):
    """Raised in real (non-mock) mode whenever a tool call did not get a real
    answer from the daemon. A benchmark score computed from anything else is
    not a measurement -- it must never be silently replaced by mock data."""


class DaemonOfflineError(CampyClientError):
    """The MCP adapter reported the daemon as offline / queued the call."""


_OFFLINE_MARKERS = ("Brain: OFFLINE", "queued_offline")


class CampyMCPClient:
    """Client for HippoCampy MCP server over STDIO transport.

    Real mode (a CAMPY_MCP_CMD is configured) is strict: any failure raises
    CampyClientError instead of falling back to the in-process mock. Mock mode
    is only entered when NO command is configured (smoke/standalone runs).
    """

    def __init__(
        self,
        mcp_cmd: Optional[str] = None,
        timeout: float = 240.0,
        mock_mode: bool = False,
    ):
        self.mcp_cmd = mcp_cmd or os.environ.get("CAMPY_MCP_CMD")
        self.timeout = timeout
        self.mock_mode = mock_mode or not bool(self.mcp_cmd)
        self._proc: Optional[subprocess.Popen] = None
        self._req_id = 0
        self.stats = {"calls": 0, "failures": 0}
        
        # Internal store for mock / fallback mode
        self._mock_memory: Dict[str, List[Dict[str, Any]]] = {}
        self._mock_constraints: Dict[str, Dict[str, Any]] = {}
        self._mock_facts: Dict[str, str] = {}

        if not self.mock_mode and self.mcp_cmd:
            self._start_process()

    def reset_mock_state(self) -> None:
        """B438: clear mock-mode scratch state between suites.

        A single CampyMCPClient is intentionally shared across every
        suite in a `run_all.py --suite all` run (real/non-mock mode
        reuses one live daemon connection rather than respawning it per
        suite). In mock mode, that same sharing let one suite's fake
        constraint/memory data leak into a later suite's mocked
        responses -- e.g. LoCoMo's fake Postgres-migration constraints
        inflating MemBench's mocked bundle-size estimate past the real
        persona's raw conversation size. Call this between suites (not
        within one -- a suite's own multi-session/multi-persona state is
        meant to accumulate across its own notify_turn calls).
        """
        self._mock_memory = {}
        self._mock_constraints = {}
        self._mock_facts = {}

    def _start_process(self) -> None:
        """Start the MCP server subprocess. Real mode never degrades to mock:
        a failed start raises CampyClientError."""
        try:
            cmd_args = shlex.split(self.mcp_cmd)
            # stderr goes to a file, never an undrained PIPE (a full 64KB pipe
            # blocks the child forever) and so a dead child leaves a reason.
            self._stderr_file = tempfile.NamedTemporaryFile(
                mode="w+", prefix="campy-mcp-stderr-", suffix=".log", delete=False
            )
            self._proc = subprocess.Popen(
                cmd_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr_file,
                text=True,
                bufsize=1,
            )
            self._lines = queue.Queue()
            threading.Thread(target=self._pump_stdout, daemon=True).start()
            self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "campy-benchmarks", "version": "0.1.0"},
            })
            self._send_notification("notifications/initialized", {})
        except CampyClientError:
            raise
        except Exception as e:
            raise CampyClientError(f"failed to start MCP server ({self.mcp_cmd!r}): {e}") from e

    def _pump_stdout(self) -> None:
        """Reader thread: lets _send_request enforce a real timeout instead of
        blocking forever in readline()."""
        try:
            for line in self._proc.stdout:
                self._lines.put(line)
        finally:
            self._lines.put(None)  # EOF marker

    def _stderr_tail(self, n: int = 800) -> str:
        try:
            self._stderr_file.flush()
            with open(self._stderr_file.name) as f:
                return f.read()[-n:]
        except Exception:
            return ""

    def _send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send one JSON-RPC request; return its `result`. Raises
        CampyClientError on a dead process, timeout, EOF, or JSON-RPC error."""
        if not self._proc or self._proc.poll() is not None:
            rc = self._proc.poll() if self._proc else None
            raise CampyClientError(
                f"MCP server process is not running (exit code {rc}); "
                f"stderr tail: {self._stderr_tail()!r}"
            )

        self._req_id += 1
        req_id = self._req_id
        payload = json.dumps({
            "jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {},
        }) + "\n"
        try:
            self._proc.stdin.write(payload)
            self._proc.stdin.flush()
        except Exception as e:
            raise CampyClientError(f"write to MCP server failed for {method}: {e}") from e

        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CampyClientError(f"{method}: no response within {self.timeout}s")
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                raise CampyClientError(f"{method}: no response within {self.timeout}s")
            if line is None:
                raise CampyClientError(
                    f"{method}: MCP server closed its stdout; stderr tail: {self._stderr_tail()!r}"
                )
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("id") != req_id:
                continue  # stale/unrelated line
            if "error" in data:
                raise CampyClientError(f"{method}: JSON-RPC error {data['error']}")
            return data.get("result") or {}

    def _send_notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        if not self._proc or self._proc.poll() is not None:
            return
        msg = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }
        try:
            self._proc.stdin.write(json.dumps(msg) + "\n")
            self._proc.stdin.flush()
        except Exception:
            pass

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke an MCP tool by name. Real mode raises on any failure and on
        any adapter-reported offline/queued response; only mock mode (no
        command configured) ever returns synthesized data."""
        arguments = arguments or {}
        if self.mock_mode:
            return self._mock_call_tool(name, arguments)

        self.stats["calls"] += 1
        try:
            result = self._send_request("tools/call", {"name": name, "arguments": arguments})
            content = result.get("content") if isinstance(result, dict) else None
            if not content:
                raise CampyClientError(f"{name}: response had no content: {result!r}")
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if any(m in text for m in _OFFLINE_MARKERS):
                        raise DaemonOfflineError(f"{name}: adapter reported daemon offline: {text[:160]!r}")
                    try:
                        return json.loads(text)
                    except Exception:
                        return {"text": text}
            return result
        except CampyClientError:
            self.stats["failures"] += 1
            raise

    def _mock_call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """High-fidelity mock behavior for offline benchmarking."""
        if name == "notify_turn":
            role = args.get("role", "user")
            content = args.get("content", "")
            session_id = args.get("session_id", "default")
            
            if session_id not in self._mock_memory:
                self._mock_memory[session_id] = []
            
            self._mock_memory[session_id].append({
                "role": role,
                "content": content,
                "timestamp": time.time(),
            })

            # Check for dynamic constraint updates or deprecation
            if "constraint:" in content.lower() or "do not" in content.lower() or "migrated" in content.lower():
                # Extract key topics
                topic_match = re.search(r"(database|postgres|analytics|auth|cache|format|language|path)", content, re.IGNORECASE)
                topic = topic_match.group(1).lower() if topic_match else "general"
                
                # Check for deprecation
                is_deprecation = any(w in content.lower() for w in ["do not use", "migrated from", "no longer", "supersede", "deprecated"])
                if is_deprecation and topic in self._mock_constraints:
                    prev_id = self._mock_constraints[topic]["id"]
                    new_id = f"c_{len(self._mock_constraints) + 1}"
                    self._mock_constraints[topic] = {
                        "id": new_id,
                        "text": content,
                        "deprecated_by": None,
                        "supersedes": prev_id,
                        "active": True,
                    }
                else:
                    cid = f"c_{len(self._mock_constraints) + 1}"
                    self._mock_constraints[topic] = {
                        "id": cid,
                        "text": content,
                        "deprecated_by": None,
                        "active": True,
                    }

            return {"status": "success", "session_id": session_id, "turns": len(self._mock_memory[session_id])}

        elif name == "current_truth":
            query = args.get("query", "").lower()
            results = []
            
            # Match constraints first
            for topic, c in self._mock_constraints.items():
                if c.get("active") and (topic in query or any(w in c["text"].lower() for w in query.split())):
                    results.append({"type": "constraint", "text": c["text"], "active": True})
            
            # Match turns
            for sid, turns in self._mock_memory.items():
                for t in turns:
                    words = [w for w in query.split() if len(w) > 3]
                    if any(w in t["content"].lower() for w in words):
                        results.append({"type": "turn", "session": sid, "content": t["content"]})

            return {"results": results[:5], "total_matches": len(results)}

        elif name == "ask":
            query = args.get("query", "").lower()
            # Synthesize answer from constraints and turns
            relevant = []
            for topic, c in self._mock_constraints.items():
                if c.get("active") and (topic in query or any(w in c["text"].lower() for w in query.split())):
                    relevant.append(c["text"])
            
            if not relevant:
                for sid, turns in self._mock_memory.items():
                    for t in turns:
                        if any(w in t["content"].lower() for w in query.split() if len(w) > 3):
                            relevant.append(t["content"])

            if relevant:
                return {"answer": f"Based on memory: {' '.join(relevant[:2])}"}
            return {"answer": "No relevant memory context found."}

        elif name == "compile_context":
            query = args.get("query", "")
            sections = [
                {"section_type": "exact_fact", "content": list(self._mock_constraints.values())},
                {"section_type": "semantic", "content": [{"query": query, "relevance": 0.9}]},
            ]
            # B436: the real compile_context response nests the estimate at
            # bundle.total_token_estimate, not a top-level "token_count" --
            # this mock previously used the wrong shape too, which masked
            # membench/runner.py's matching bug in smoke/mock mode. Estimate
            # from the actual mocked content (same ~4 chars/token heuristic
            # runner.py uses) rather than a hardcoded literal, so it moves
            # with the mocked data like the real bundle does.
            content_chars = sum(
                len(str(item))
                for sec in sections
                for item in sec["content"]
            )
            token_estimate = max(1, content_chars // 4)
            return {
                "bundle": {
                    "sections": sections,
                    "total_token_estimate": token_estimate,
                },
            }

        elif name == "run_sweep":
            return {"status": "swept", "decayed_nodes": 0, "resurrected_nodes": 0}

        elif name == "record_transition":
            return {"status": "recorded"}

        return {"status": "unknown_tool", "name": name}

    def notify_turn(self, role: str, content: str, session_id: str = "benchmark") -> Dict[str, Any]:
        return self.call_tool("notify_turn", {"role": role, "content": content, "session_id": session_id})

    def current_truth(self, query: str) -> Dict[str, Any]:
        return self.call_tool("current_truth", {"query": query})

    def ask(self, query: str, session_id: str = "benchmark", token_budget: int = 32000) -> str:
        # capture=False: an evaluation probe must not write its question and the
        # model's answer back into the memory being evaluated (ask's default
        # closed-loop capture would pollute it, including with "no info" answers).
        res = self.call_tool("ask", {"query": query, "session_id": session_id,
                                     "token_budget": token_budget, "capture": False})
        if isinstance(res, dict):
            return res.get("answer", res.get("text", str(res)))
        return str(res)

    def compile_context(self, query: str, token_budget: int = 32000) -> Dict[str, Any]:
        return self.call_tool("compile_context", {"query": query, "token_budget": token_budget})

    def run_sweep(self, timeout: float = 3600.0, poll: float = 5.0) -> Dict[str, Any]:
        """B448: settle -- block until the daemon's Gated Consolidation Loop has
        drained (context_status.consolidation_pending == 0 on consecutive
        polls). There has never been a `run_sweep` MCP tool; the old call
        silently no-op'd, so probes raced consolidation. Mock mode: no-op.
        Raises CampyClientError if it does not drain within `timeout`."""
        if self.mock_mode:
            return {"status": "mock"}
        deadline = time.monotonic() + timeout
        zero_polls = 0
        pending = None
        while time.monotonic() < deadline:
            res = self.call_tool("context_status", {"session_id": "benchmark-settle"})
            pending = res.get("consolidation_pending")
            if pending is None:
                raise CampyClientError(
                    "context_status has no consolidation_pending -- daemon predates B448"
                )
            zero_polls = zero_polls + 1 if pending == 0 else 0
            if zero_polls >= 2:
                return {"status": "settled"}
            time.sleep(poll)
        raise CampyClientError(f"consolidation did not drain within {timeout}s (pending={pending})")

    def close(self) -> None:
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2.0)
            except Exception:
                pass
            self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

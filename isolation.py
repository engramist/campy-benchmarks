"""
campy-benchmarks / isolation.py
Run the benchmark against a throwaway daemon with its own store.

Without this, every run reads and writes the user's personal ~/.campy store:
benchmark turns land in personal memory, earlier runs' identical turns (fixed
session ids and seeds) contaminate later runs, and results depend on
whatever else happens to be in the store.

Isolated mode:
  * creates a temp CAMPY_HOME (hippocampy >= the CAMPY_HOME change; the
    daemon then keeps socket, graph store, vector/FTS index, logs and
    offline queue there, and loads config only from $CAMPY_HOME/config.toml);
  * writes that config from the base config with overrides (capture off,
    no self-restart/watchdog exits mid-run, a free web port);
  * starts `python -m campy.brain_daemon` there and waits for its socket;
  * gives the MCP adapter an environment pinned to that daemon: its socket
    AND its HTTP URL. The adapter falls back to HTTP when the socket fails,
    and the default HTTP URL is the personal daemon on :7799, so pinning
    the socket alone would not be isolation. Env vars that would outrank
    those (BRAIN_URL, SIDEQUESTS_*, CAMPY_BRAIN_SOCKET) are removed.

The harness still imports nothing from the engine: it launches the daemon
module by name, exactly like `campy start` would.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import store_stats

# Env vars that outrank CAMPY_SOCKET_PATH / CAMPY_BRAIN_URL in
# campy.brain_transport and could route calls to the personal daemon.
_OUTRANKING_ENV = (
    "SIDEQUESTS_BRAIN_SOCKET",
    "SIDEQUESTS_SOCKET_PATH",
    "CAMPY_BRAIN_SOCKET",
    "SIDEQUESTS_BRAIN_URL",
    "BRAIN_URL",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Minimal TOML writer (stdlib has a reader only). Handles what campy config
# uses: tables, arrays of tables ([[x]]), strings, bools, ints, floats, and
# lists of scalars.
# ---------------------------------------------------------------------------

def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"unsupported TOML value {v!r}")


def _toml_key(k: str) -> str:
    return k if k.replace("_", "").replace("-", "").isalnum() else _toml_value(k)


def dump_toml(data: Dict[str, Any]) -> str:
    lines: list = []

    def is_table_array(v: Any) -> bool:
        return isinstance(v, list) and bool(v) and all(isinstance(x, dict) for x in v)

    def emit(table: Dict[str, Any], prefix: str, header: str = "[{}]") -> None:
        scalars = {k: v for k, v in table.items()
                   if not isinstance(v, dict) and not is_table_array(v)}
        tables = {k: v for k, v in table.items() if isinstance(v, dict)}
        arrays = {k: v for k, v in table.items() if is_table_array(v)}
        if prefix and (scalars or header != "[{}]" or not (tables or arrays)):
            lines.append(header.format(prefix))
        for k, v in scalars.items():
            lines.append(f"{_toml_key(k)} = {_toml_value(v)}")
        lines.append("")
        for k, v in tables.items():
            emit(v, f"{prefix}.{_toml_key(k)}" if prefix else _toml_key(k))
        for k, items in arrays.items():
            for item in items:
                emit(item, f"{prefix}.{_toml_key(k)}" if prefix else _toml_key(k), "[[{}]]")

    emit(data, "")
    return "\n".join(lines) + "\n"


def _set(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    node = cfg
    *parents, leaf = dotted.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = value


class IsolatedDaemon:
    """Context manager: a throwaway daemon plus the env for talking to it."""

    def __init__(
        self,
        python: str,
        base_config: Optional[Path],
        keep_store: bool = False,
        ready_timeout: float = 900.0,
    ):
        self.python = python
        self.base_config = base_config
        self.keep_store = keep_store
        self.ready_timeout = ready_timeout
        self.home: Optional[Path] = None
        self.port: Optional[int] = None
        self.proc: Optional[subprocess.Popen] = None
        self.overrides: Dict[str, Any] = {}
        self.config: Dict[str, Any] = {}
        self.config_text = ""
        self.ready_seconds: Optional[float] = None
        self._log = None
        self._final_activity_lines: Optional[int] = None
        self.graph_stats: Optional[Dict[str, Any]] = None

    # -- config ------------------------------------------------------------

    def _build_config(self) -> Dict[str, Any]:
        import tomllib

        cfg: Dict[str, Any] = {}
        if self.base_config is not None:
            cfg = tomllib.loads(self.base_config.read_text())
        self.port = _free_port()
        self.overrides = {
            # The daemon doesn't start transcript capture itself today; this
            # keeps it that way if it ever does -- an isolated run must never
            # ingest the user's real agent sessions.
            "capture.enabled": False,
            "capture.codex.enabled": False,
            "capture.claude_code.enabled": False,
            "capture.vscode.enabled": False,
            # Both of these restart by exiting and rely on launchd to respawn;
            # a standalone benchmark daemon would just die mid-run.
            "daemon.restart_interval_hours": 0,
            "watchdog.enabled": False,
            # Don't collide with the personal daemon's :7799.
            "web.port": self.port,
            "server.bind_host": "127.0.0.1",
            "activity.log_path": str(self.home / "activity.log"),
        }
        for k, v in self.overrides.items():
            _set(cfg, k, v)
        return cfg

    # -- env ---------------------------------------------------------------

    def client_env(self) -> Dict[str, str]:
        """Environment for the MCP adapter subprocess (and the daemon)."""
        env = {k: v for k, v in os.environ.items() if k not in _OUTRANKING_ENV}
        env["CAMPY_HOME"] = str(self.home)
        env["CAMPY_SOCKET_PATH"] = str(self.home / "brain.sock")
        env["CAMPY_BRAIN_URL"] = f"http://127.0.0.1:{self.port}/mcp"
        return env

    # -- lifecycle ---------------------------------------------------------

    def _preflight(self) -> None:
        """Refuse to launch a hippocampy that ignores CAMPY_HOME (pre-B456).
        Such a daemon still honors CAMPY_SOCKET_PATH, so it would look ready
        while opening the PERSONAL ~/.campy store."""
        probe = ("from campy.paths import runtime_dir, home_override; "
                 "assert home_override() is not None; print(runtime_dir())")
        try:
            out = subprocess.run(
                [self.python, "-c", probe], cwd=self.home, env=self.client_env(),
                capture_output=True, text=True, timeout=120,
            )
        except Exception as e:
            raise RuntimeError(f"isolation preflight could not run {self.python!r}: {e}") from e
        resolved = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
        if out.returncode != 0 or Path(resolved) != self.home:
            raise RuntimeError(
                "refusing --isolated: this hippocampy does not honor CAMPY_HOME (needs B456); "
                f"runtime_dir resolved to {resolved or '?'!r}. stderr: {out.stderr[-400:]!r}"
            )

    def _assert_store_inside_home(self) -> None:
        if not any((self.home / f).exists() for f in ("brain.db", "vectors.db")):
            raise RuntimeError(
                f"isolated daemon is up but created no store under {self.home}; "
                "it may be using the personal store -- stopped it"
            )

    def start(self) -> "IsolatedDaemon":
        # Short base dir: macOS caps AF_UNIX paths at ~104 bytes and its
        # $TMPDIR (/var/folders/...) is long.
        base = os.environ.get("CAMPY_BENCH_TMPDIR") or ("/tmp" if Path("/tmp").is_dir() else None)
        self.home = Path(tempfile.mkdtemp(prefix="campy-bench-", dir=base))
        self.config = self._build_config()
        self.config_text = dump_toml(self.config)
        (self.home / "config.toml").write_text(self.config_text)
        try:
            self._preflight()
        except Exception:
            self.stop()
            raise
        self._log = open(self.home / "daemon.stdout.log", "w")
        t0 = time.monotonic()
        self.proc = subprocess.Popen(
            [self.python, "-m", "campy.brain_daemon"],
            cwd=self.home,  # no campy.toml here, so $CAMPY_HOME/config.toml is used
            env=self.client_env(),
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )
        sock = self.home / "brain.sock"
        deadline = t0 + self.ready_timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                msg = (f"isolated daemon exited with {self.proc.returncode} during startup; "
                       f"log tail: {self.log_tail()!r}")
                self.stop()  # removes the temp CAMPY_HOME unless --keep-store
                raise RuntimeError(msg)
            if sock.exists() and self._socket_accepts(sock):
                self.ready_seconds = round(time.monotonic() - t0, 1)
                try:
                    self._assert_store_inside_home()
                except Exception:
                    self.stop()
                    raise
                return self
            time.sleep(1.0)
        self.stop()
        raise RuntimeError(f"isolated daemon not ready within {self.ready_timeout}s")

    @staticmethod
    def _socket_accepts(path: Path) -> bool:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(2.0)
                s.connect(str(path))
            return True
        except OSError:
            return False

    def log_tail(self, n: int = 1200) -> str:
        try:
            return (self.home / "daemon.stdout.log").read_text()[-n:]
        except Exception:
            return ""

    def activity_lines(self) -> int:
        """Lines in the isolated activity log -- nonzero after a run is
        evidence the calls reached this daemon's runtime dir."""
        try:
            return sum(1 for _ in open(self.home / "activity.log"))
        except OSError:
            return 0

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        if self._log:
            self._log.close()
        if self.home and self._final_activity_lines is None:
            self._final_activity_lines = self.activity_lines()
        if self.home and self.proc is not None and self.graph_stats is None:
            # Count what the run stored before the store is deleted (B461).
            self.graph_stats = store_stats.collect(self.python, self.home)
        if self.home and not self.keep_store:
            shutil.rmtree(self.home, ignore_errors=True)

    def describe(self) -> Dict[str, Any]:
        return {
            "isolated": True,
            "campy_home": str(self.home) if self.keep_store else "(deleted after run)",
            "web_port": self.port,
            "config_overrides": self.overrides,
            "daemon_ready_seconds": self.ready_seconds,
            "activity_log_lines": (self._final_activity_lines
                                   if self._final_activity_lines is not None
                                   else self.activity_lines()),
            "graph_stats": self.graph_stats,
        }

    def __enter__(self) -> "IsolatedDaemon":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

"""
campy-benchmarks / provenance.py
Records what a run actually measured, so two result files can be compared
honestly (or refused comparison) without guessing.

Nothing here imports the engine: the hippocampy commit is read with `git`
from the checkout CAMPY_MCP_CMD points into, and the LLM/embedding config is
read from a TOML file. The daemon resolves its config from ITS working
directory, which the harness cannot see, so the config is best-effort and
labelled with where it came from. Set CAMPY_BENCH_CONFIG to the daemon's
actual config file to make it exact.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

HARNESS_ROOT = Path(__file__).resolve().parent


def _git(cwd: Path, *args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def git_info(repo: Optional[Path]) -> Dict[str, Any]:
    if repo is None or not repo.exists():
        return {"path": str(repo) if repo else None, "commit": None}
    commit = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain", "--untracked-files=no")
    return {
        "path": str(repo),
        "commit": commit,
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def hippocampy_repo_from_cmd(mcp_cmd: Optional[str]) -> Optional[Path]:
    """Walk up from the MCP command's executable to the enclosing git
    checkout (e.g. .../hippocampy/.venv/bin/python -> .../hippocampy)."""
    explicit = os.environ.get("CAMPY_REPO_PATH")
    if explicit:
        return Path(explicit)
    if not mcp_cmd:
        return None
    try:
        exe = Path(shlex.split(mcp_cmd)[0])
    except (ValueError, IndexError):
        return None
    for parent in exe.parents:
        if (parent / ".git").exists():
            return parent
    return None


def _config_candidates(repo: Optional[Path]):
    explicit = os.environ.get("CAMPY_BENCH_CONFIG")
    if explicit:
        yield Path(explicit), "CAMPY_BENCH_CONFIG"
        return
    home = Path.home()
    yield home / ".campy" / "config.toml", "guessed:~/.campy/config.toml"
    if repo is not None:
        yield repo / "campy.toml", "guessed:hippocampy-repo/campy.toml"


def _summarize(cfg: Dict[str, Any]) -> Dict[str, Any]:
    llm = cfg.get("llm", {})
    return {
        "llm_provider": llm.get("provider"),
        "llm_model": llm.get("model"),
        "llm_step_overrides": {
            k: {"provider": v.get("provider"), "model": v.get("model")}
            for k, v in llm.items()
            if isinstance(v, dict)
        },
        "embeddings_model": cfg.get("embeddings", {}).get("model"),
        "compression_model": cfg.get("compression", {}).get("compression_model"),
    }


def base_config_path(repo: Optional[Path]) -> Optional[Path]:
    """The config an isolated daemon is derived from: the same candidates
    as the best-effort guess (CAMPY_BENCH_CONFIG, ~/.campy/config.toml,
    the hippocampy repo's campy.toml), first that exists."""
    for path, _source in _config_candidates(repo):
        if path.exists():
            return path
    return None


def daemon_config(repo: Optional[Path]) -> Dict[str, Any]:
    for path, source in _config_candidates(repo):
        if not path.exists():
            continue
        raw = path.read_bytes()
        info: Dict[str, Any] = {
            "source": source,
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        try:
            import tomllib

            info.update(_summarize(tomllib.loads(raw.decode())))
        except Exception as e:  # unparseable config is still worth hashing
            info["parse_error"] = str(e)[:200]
        return info
    return {"source": None, "note": "no config found; set CAMPY_BENCH_CONFIG"}


def isolated_daemon_config(isolated) -> Dict[str, Any]:
    """Exact: this is the config file the harness itself wrote."""
    info: Dict[str, Any] = {
        "source": "isolated (written by harness)",
        "derived_from": str(isolated.base_config) if isolated.base_config else None,
        "sha256": hashlib.sha256(isolated.config_text.encode()).hexdigest(),
    }
    info.update(_summarize(isolated.config))
    return info


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (list, tuple)):
        return [_jsonable(o) for o in obj]
    return obj


def dataset_fingerprint(smoke: bool) -> Dict[str, str]:
    """sha256 of each suite's fixtures (turns + probe specs). A changed hash
    means scores are not comparable even at the same scorer version."""
    from locomo.dataset import get_locomo_scenarios
    from membench.msc_dataset import get_msc_personas

    def h(obj: Any) -> str:
        blob = json.dumps(_jsonable(obj), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    return {
        "locomo": h(get_locomo_scenarios(smoke=smoke)),
        "membench": h(get_msc_personas(smoke=smoke)),
    }


def collect(mcp_cmd: Optional[str], smoke: bool, argv, isolated=None) -> Dict[str, Any]:
    from scoring import SCORER_VERSION

    repo = hippocampy_repo_from_cmd(mcp_cmd)
    if isolated is not None:
        config_info = isolated_daemon_config(isolated)
    elif mcp_cmd:
        config_info = daemon_config(repo)
    else:
        config_info = {"source": "mock"}
    return {
        "scorer_version": SCORER_VERSION,
        "argv": list(argv),
        "harness": git_info(HARNESS_ROOT),
        "hippocampy": git_info(repo),
        "daemon_config": config_info,
        "store": isolated.describe() if isolated is not None else {
            "isolated": False,
            "note": "personal ~/.campy store; contains earlier runs' data",
        },
        "dataset_sha": dataset_fingerprint(smoke),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

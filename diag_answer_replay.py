#!/usr/bin/env python3
"""
campy-benchmarks / diag_answer_replay.py   (was diag_dmr_answer.py)

Re-asks a run's questions on the stores it kept (`--keep-store`), through
hippocampy's own `ask` pipeline (run_ask, capture off), with the bundle
changed one way at a time, and judges each answer with the run's judge.
Each store is replayed in a subprocess on a copy, so the kept store is never
written. Run with no daemon on the stores.

Suites: DMR (one kept store per question, from the result rows) and
LoCoMo-10 (one store per conversation: pass it with --store; category 5 is
scored by abstention, as the run does).

Why: R24 (hippocampy 9c5c9d8, fields mode) retrieves better than R20 (text
mode) on DMR but answers worse (0.500 vs 0.620), and R26 showed the turns,
not the semantic section, decide it. The conversation stage picks its top
`conversation_limit` turns by rank and then lists them OLDEST FIRST (B454: a
later statement visibly supersedes an earlier one), so R26's `top3` -- the
first 3 listed -- was the 3 oldest of the 6, not the 3 best. These variants
separate "fewer turns" from "earlier turns" from "turn order":

  asis       the bundle as compiled (should reproduce the run's answer)
  nosem      without the semantic section
  convonly   the conversation section alone
  oldest3    the 3 oldest of the 6 turns (what R26 called top3)
  limit3     conversation_limit 3: the 3 best-ranked turns, oldest first
  limit4     conversation_limit 4
  rankorder  the 6 turns, best-ranked first (rank from limits 1..6)

    ~/Desktop/GitProjects/hippocampy/.venv/bin/python diag_answer_replay.py \\
        results/r20a-dmr.json results/r24b-dmr-q50-fields.json --judge-model gemma4:26b --show
    ~/Desktop/GitProjects/hippocampy/.venv/bin/python diag_answer_replay.py \\
        results/r24a-locomo10-q60-fields.json --store <R24a's kept store> --judge-model gemma4:26b

Replaying a branch's code (M0.4) and another answer model (M0.5):

  --hippocampy <path>   run the worker with PYTHONPATH=<path> in front, so the
                        campy package under <path> (a git worktree of a branch)
                        is imported instead of the installed one. The worker
                        records the campy.__file__ it imported, so you can see
                        which code answered; a path that does not take effect
                        is an error.
  --llm-model <name>    set [llm].model in the COPIED store's config.toml
                        (the kept store is never written).

eval_gate.py runs the `asis` replay for two trees and prints the gate table.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

VARIANTS = ("asis", "nosem", "convonly", "oldest3", "limit3", "limit4", "rankorder")


def _limit_of(variant: str):
    return int(variant[5:]) if variant.startswith("limit") else None


def _apply(bundle, variant: str, rank: dict):
    secs = list(bundle.sections or [])
    if variant == "nosem":
        secs = [s for s in secs if s.section_type != "semantic"]
    elif variant == "convonly":
        secs = [s for s in secs if s.section_type == "conversation"]
    elif variant in ("oldest3", "rankorder"):
        for s in secs:
            if s.section_type != "conversation":
                continue
            if variant == "oldest3":
                s.content = s.content[:3]
            else:
                s.content = sorted(s.content, key=lambda c: rank.get(c.get("text"), len(rank)))
            s.token_estimate = sum(max(1, len(c.get("text", "")) // 4) for c in s.content)
    bundle.sections = secs
    return bundle


def _with_limit(config: dict, limit: int) -> dict:
    cfg = copy.deepcopy(config)
    cfg.setdefault("retrieval", {})["conversation_limit"] = limit
    return cfg


def worker(questions: list, variants: list) -> dict:
    """In a subprocess with CAMPY_HOME set to a copy of one kept store."""
    import campy
    from campy.brain.brainstem.config import load_config
    from campy.brain.hippocampus.graph.oxigraph_client import OxigraphClient
    from campy.brain.thalamus import ask as ask_mod
    from campy.brain.thalamus.bundle_compiler import _stage_conversation
    from campy.paths import get_database_path

    config = load_config()
    base_limit = int((config.get("retrieval", {}) or {}).get("conversation_limit", 6))
    db = OxigraphClient(str(get_database_path()))
    real_compile, real_get_llm = ask_mod.compile_bundle, ask_mod._get_llm
    seen: dict = {}

    async def turn_ranks(question: str) -> dict:
        """Stamped turn text -> rank, from the stage at limits 1..N: the turn a
        limit adds is the next best-ranked."""
        rank: dict = {}
        for k in range(1, base_limit + 1):
            sec = await _stage_conversation(db, question, _with_limit(config, k))
            for c in (sec.content if sec else []):
                rank.setdefault(c.get("text"), len(rank))
        return rank

    async def one(question: str) -> dict:
        rank = await turn_ranks(question) if "rankorder" in variants else {}
        out = {}
        for v in variants:
            seen.clear()

            async def compile_variant(*a, _v=v, **k):
                b = _apply(await real_compile(*a, **k), _v, rank)
                seen["sections"] = [(s.section_type, len(s.content)) for s in b.sections]
                return b

            ask_mod.compile_bundle = compile_variant
            cfg = _with_limit(config, _limit_of(v)) if _limit_of(v) else config
            meta: dict = {}
            t0 = time.perf_counter()
            answer = await ask_mod.run_ask(question, "diag_eval", db, cfg, capture=False, meta=meta)
            out[v] = {"answer": answer, "sections": seen.get("sections"), "prompt": seen.get("prompt"),
                      "compressed": meta.get("compression_bypassed") is False,
                      "latency_ms": round((time.perf_counter() - t0) * 1000)}
        return out

    def get_llm(cfg):
        llm = real_get_llm(cfg)
        if llm is None:
            return None

        class Recorder:  # keeps the prompt the model was sent
            async def achat(self, messages, **kw):
                seen["prompt"] = messages[-1]["content"]
                return await llm.achat(messages, **kw)

            def chat(self, messages, **kw):
                return llm.chat(messages, **kw)
        return Recorder()

    async def run_all() -> list:
        return [await one(q) for q in questions]

    ask_mod._get_llm = get_llm
    try:
        return {"campy_file": str(campy.__file__), "llm_model": (config.get("llm") or {}).get("model"),
                "rows": asyncio.run(run_all())}
    finally:
        ask_mod.compile_bundle, ask_mod._get_llm = real_compile, real_get_llm


def set_llm_model(config_toml: Path, model: str) -> None:
    """Set [llm].model in `config_toml` (a COPY), keeping the rest of the file."""
    import re
    import tomllib
    if not config_toml.exists():
        raise RuntimeError(f"--llm-model: {config_toml} does not exist in the store copy")
    lines = config_toml.read_text().splitlines(keepends=True)
    new = f'model = "{model}"\n'
    head = next((i for i, l in enumerate(lines) if re.match(r"\s*\[llm\]\s*(#.*)?$", l)), None)
    if head is None:
        lines += ["\n", "[llm]\n", new]
    else:
        end = next((i for i in range(head + 1, len(lines)) if re.match(r"\s*\[", lines[i])), len(lines))
        at = next((i for i in range(head + 1, end) if re.match(r"\s*model\s*=", lines[i])), None)
        if at is None:
            lines.insert(head + 1, new)
        else:
            lines[at] = new
    config_toml.write_text("".join(lines))
    got = tomllib.loads(config_toml.read_text()).get("llm", {}).get("model")
    if got != model:
        raise RuntimeError(f"--llm-model: wrote {model!r} but [llm].model reads {got!r}")


def replay(store: Path, questions: list, variants: list, hippocampy: Path | None = None,
           llm_model: str | None = None, info: dict | None = None) -> list:
    """One subprocess on a copy of `store` for all of `questions`.

    hippocampy: a tree whose `campy` package the worker imports (PYTHONPATH in
    front); None = whatever is installed. llm_model: [llm].model for the copy.
    info, if given, receives the worker's `campy_file` and `llm_model`."""
    with tempfile.TemporaryDirectory(prefix="diag-replay-") as tmp:
        home = Path(tmp) / "store"
        shutil.copytree(store, home, ignore=shutil.ignore_patterns("*.pid", "*.sock", "*.lock"))
        if llm_model:
            set_llm_model(home / "config.toml", llm_model)
        qfile = Path(tmp) / "questions.json"
        qfile.write_text(json.dumps(questions))
        env = {**os.environ, "CAMPY_HOME": str(home)}
        if hippocampy is not None:
            tree = Path(hippocampy).resolve()
            if not (tree / "campy" / "__init__.py").exists():
                raise RuntimeError(f"--hippocampy {tree}: no campy/__init__.py there")
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(tree), env.get("PYTHONPATH")]))
        # cwd = the temp dir: load_config looks at ./campy.toml before $CAMPY_HOME/config.toml,
        # so running from a checkout would silently replace the store's config.
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(qfile),
                               "--variants", ",".join(variants)],
                              env=env, cwd=tmp, capture_output=True, text=True, timeout=3600 * 4)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-1500:])
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        if hippocampy is not None:
            tree = Path(hippocampy).resolve()
            if not Path(out["campy_file"]).resolve().is_relative_to(tree):
                raise RuntimeError(f"--hippocampy {tree} did not take effect: the worker imported "
                                   f"campy from {out['campy_file']}")
        if info is not None:
            info.update(campy_file=out["campy_file"], llm_model=out.get("llm_model"))
        return out["rows"]


def make_passed(judge, votes: int = 1):
    """passed(d, answer) with `judge` (a BaselineLLM). Verdicts are cached on
    (question, expected, answer): two trees that give the same answer get the same
    verdict, so judge noise cannot show up as a gained or lost question. DMR
    records (no `category`) get the persona rule, as in dmr/scoring.py; `votes`
    is as in locomo10.scoring.judge_one."""
    from locomo10.scoring import judge_one, score_answer
    from qa_judge import is_non_answer
    cache: dict = {}
    from collections import Counter
    problems: Counter = Counter()  # verdicts the judge did not really give

    def passed(d: dict, answer: str) -> bool:
        if d.get("category") == 5:  # LoCoMo-10 adversarial: declining is right
            return bool(score_answer(answer, d["expected"], 5)["passed"])
        if not answer.strip() or is_non_answer(answer):
            return False
        persona = d.get("category") is None  # DMR; every LoCoMo-10 record has a category
        key = (d["question"], d["expected"], answer, persona)
        if key not in cache:
            v = judge_one(judge, d["question"], d["expected"], answer, persona=persona, votes=votes)
            if v.get("reason") in ("judge_unparseable", "judge_error"):
                problems[v["reason"]] += 1
            cache[key] = v["judge"]
        return cache[key]
    passed.problems = problems  # e.g. a capped or failing judge: report it next to any score
    return passed


def run_passed(d: dict) -> bool:
    return bool(d.get("passed") if d.get("category") == 5 else d.get("judge"))


def replay_result(path: Path, variants: list, passed, store: Path | None = None, ids=(),
                  hippocampy: Path | None = None, llm_model: str | None = None,
                  show: bool = False, log=print) -> tuple:
    """Replay one result file. Returns (suite, options, rows, info, same) where rows hold, per
    question, run_passed and variants[v] = {answer, passed, latency_ms, ...}."""
    data = json.loads(Path(path).read_text())
    suites = data["suites"]
    suite = "dmr" if "dmr" in suites else "locomo10"
    options = (suites[suite].get("dataset") or {}).get("options") or {}
    details = [d for d in suites[suite]["details"] if not ids or d["id"] in ids]
    groups: dict = {}  # store -> records
    for d in details:
        st = store if suite == "locomo10" else Path(d.get("store") or "")
        if st is None or not (Path(st) / "brain.db").exists():
            log(f"{Path(path).name} {d['id']}: no kept store ({st}), skipped")
            continue
        groups.setdefault(Path(st), []).append(d)
    rows, same, info = [], 0, {}
    for st, ds in groups.items():
        try:
            got = replay(st, [d["question"] for d in ds], variants, hippocampy, llm_model, info)
        except Exception as e:  # one bad store must not end the run
            log(f"{Path(path).name} {st}: replay failed: {str(e)[-300:]}")
            continue
        for d, g in zip(ds, got):
            row = {"id": d["id"], "category": d.get("category"), "run_passed": run_passed(d),
                   "run_answer": d.get("answer"), "expected": d["expected"], "variants": {}}
            for v in variants:
                row["variants"][v] = {**g[v], "passed": passed(d, g[v]["answer"])}
            same += g.get("asis", {}).get("answer", "").strip() == (d.get("answer") or "").strip()
            rows.append(row)
            if show:
                marks = " ".join(f"{v}={'Y' if row['variants'][v]['passed'] else '-'}" for v in variants)
                log(f"{d['id']:<10} run={'Y' if row['run_passed'] else '-'} {marks}  gold: {str(d['expected'])[:60]}")
                for v in variants:
                    log(f"{'':<12}{v:<10} {row['variants'][v]['sections']}  "
                        f"{row['variants'][v]['answer'][:110]!r}")
    return suite, options, rows, info, same


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", nargs="*", type=Path, help="run_all.py result files from --keep-store runs")
    ap.add_argument("--store", type=Path, help="LoCoMo-10: the run's kept store (its 'store kept:' path)")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--ids", default="", help="comma-separated question ids (default: all)")
    ap.add_argument("--judge-provider", default="ollama")
    ap.add_argument("--judge-model", default="gemma4:26b")
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--judge-votes", type=int, choices=[1, 3], default=1,
                    help="3 = judge twice, a third call breaks a disagreement (default 1)")
    ap.add_argument("--show", action="store_true", help="per-question verdicts and answers")
    ap.add_argument("--out", type=Path, help="write every answer and verdict as JSON")
    ap.add_argument("--hippocampy", type=Path,
                    help="a hippocampy tree (e.g. a git worktree of a branch) whose campy package the worker imports")
    ap.add_argument("--llm-model", help="answer model: sets [llm].model in the copied store's config.toml")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    args = ap.parse_args()
    variants = [v for v in args.variants.split(",") if v]
    bad = set(variants) - set(VARIANTS)
    if bad:
        ap.error(f"unknown variants {sorted(bad)}; choose from {VARIANTS}")
    if args.worker is not None:
        print(json.dumps(worker(json.loads(Path(args.worker).read_text()), variants)))
        return 0
    if not args.results:
        ap.error("give at least one results file")

    from llm_client import BaselineLLM

    judge = BaselineLLM(args.judge_provider, args.judge_model, args.judge_base_url)
    passed = make_passed(judge, args.judge_votes)
    ids = {i for i in args.ids.split(",") if i}

    report = {}
    for path in args.results:
        suite, _opts, rows, info, same = replay_result(path, variants, passed, args.store, ids,
                                                        args.hippocampy, args.llm_model, args.show)
        n = len(rows)
        tally = {v: [sum(r["variants"][v]["passed"] for r in rows), n] for v in variants}
        print(f"\n{path.name} ({suite}): {n} questions replayed; run score on them "
              f"{sum(r['run_passed'] for r in rows) / n if n else 0:.3f}")
        if info:
            print(f"  code: {info.get('campy_file')}   answer model: {info.get('llm_model')}")
        if "asis" in variants:
            print(f"  asis reproduced the run's answer verbatim on {same}/{n}")
        for v in variants:
            ok, m = tally[v]
            line = f"  {v:<10} {ok / m if m else 0:.3f}  ({ok}/{m})"
            if suite == "locomo10":
                adv = [r for r in rows if r["category"] == 5]
                if adv:
                    line += f"   adversarial {sum(r['variants'][v]['passed'] for r in adv)}/{len(adv)}"
            print(line)
        report[str(path)] = rows
    print(f"\njudge problems (verdicts scored WRONG without a real verdict): {dict(passed.problems) or 'none'}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

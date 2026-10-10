#!/usr/bin/env python3
"""
campy-benchmarks / diag_dmr_answer.py

Re-asks a DMR run's questions on the stores it kept (`--keep-store`), through
hippocampy's own `ask` pipeline (run_ask, capture off), with the bundle
changed one way at a time, and judges each answer with the run's judge.
Each question runs in a subprocess on a copy of its store, so the kept store
is never written. Run with no daemon on the stores.

Why: R24 (hippocampy 9c5c9d8, fields mode) retrieves better than R20 (text
mode) -- DMR evidence recall 0.580 vs 0.531 -- yet answers worse (0.500 vs
0.620): in every lost question the evidence turn is in the bundle, often
first. Between the two the bundle lost most of its semantic section (4.7
items -> 0.4) and two-thirds of its tokens. The variants separate "the model
needed the semantic section" from "the turns are presented badly":

  asis      the bundle as compiled (should reproduce the run's answer)
  nosem     without the semantic section
  convonly  the conversation section alone
  chrono    the conversation turns in time order instead of rank order
  top3      only the conversation section's first 3 turns

Run it on both runs' results and compare the tables: if R20's accuracy falls
under `nosem` to R24's level, the semantic section was carrying it.

    ~/Desktop/GitProjects/hippocampy/.venv/bin/python diag_dmr_answer.py \\
        results/r20a-dmr-q50-baselines.json results/r24b-dmr-q50-fields.json \\
        --judge-model gemma4:26b [--variants asis,nosem] [--ids valid_29,valid_70] [--show] [--out x.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

VARIANTS = ("asis", "nosem", "convonly", "chrono", "top3")


def _apply(bundle, variant: str):
    secs = list(bundle.sections or [])
    if variant == "nosem":
        secs = [s for s in secs if s.section_type != "semantic"]
    elif variant == "convonly":
        secs = [s for s in secs if s.section_type == "conversation"]
    elif variant in ("chrono", "top3"):
        for s in secs:
            if s.section_type != "conversation":
                continue
            if variant == "chrono":
                s.content = sorted(s.content, key=lambda c: str(c.get("created_at") or "~"))  # undated last
            else:
                s.content = s.content[:3]
            s.token_estimate = sum(max(1, len(c.get("text", "")) // 4) for c in s.content)
    bundle.sections = secs
    return bundle


def worker(question: str, variants: list) -> dict:
    """In a subprocess with CAMPY_HOME set to a copy of one kept store."""
    from campy.brain.brainstem.config import load_config
    from campy.brain.hippocampus.graph.oxigraph_client import OxigraphClient
    from campy.brain.thalamus import ask as ask_mod
    from campy.paths import get_database_path

    config = load_config()
    db = OxigraphClient(str(get_database_path()))
    real_compile, real_get_llm = ask_mod.compile_bundle, ask_mod._get_llm
    seen: dict = {}

    async def run_all() -> dict:
        out = {}
        for v in variants:
            seen.clear()

            async def compile_variant(*a, _v=v, **k):
                b = _apply(await real_compile(*a, **k), _v)
                seen["sections"] = [(s.section_type, len(s.content)) for s in b.sections]
                return b

            ask_mod.compile_bundle = compile_variant
            meta: dict = {}
            answer = await ask_mod.run_ask(question, "diag_eval", db, config, capture=False, meta=meta)
            out[v] = {"answer": answer, "sections": seen.get("sections"), "prompt": seen.get("prompt"),
                      "compressed": meta.get("compression_bypassed") is False}
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

    ask_mod._get_llm = get_llm
    try:
        return asyncio.run(run_all())
    finally:
        ask_mod.compile_bundle, ask_mod._get_llm = real_compile, real_get_llm


def run_one(store: Path, question: str, variants: list) -> dict:
    with tempfile.TemporaryDirectory(prefix="diag-dmr-") as tmp:
        home = Path(tmp) / "store"
        shutil.copytree(store, home, ignore=shutil.ignore_patterns("*.pid", "*.sock", "*.lock"))
        env = {**os.environ, "CAMPY_HOME": str(home)}
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", question, "--variants", ",".join(variants)],
                              env=env, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-1500:])
        return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", nargs="*", type=Path, help="run_all.py result files from --keep-store DMR runs")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--ids", default="", help="comma-separated question ids (default: all)")
    ap.add_argument("--judge-provider", default="ollama")
    ap.add_argument("--judge-model", default="gemma4:26b")
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--show", action="store_true", help="per-question answers")
    ap.add_argument("--out", type=Path, help="write every answer and verdict as JSON")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    args = ap.parse_args()
    variants = [v for v in args.variants.split(",") if v]
    bad = set(variants) - set(VARIANTS)
    if bad:
        ap.error(f"unknown variants {sorted(bad)}; choose from {VARIANTS}")
    if args.worker is not None:
        print(json.dumps(worker(args.worker, variants)))
        return 0
    if not args.results:
        ap.error("give at least one results file")

    from dmr.scoring import f1
    from llm_client import BaselineLLM
    from locomo10.scoring import judge_one
    from qa_judge import is_non_answer

    judge = BaselineLLM(args.judge_provider, args.judge_model, args.judge_base_url)
    ids = {i for i in args.ids.split(",") if i}
    report = {}
    for path in args.results:
        details = json.loads(path.read_text())["suites"]["dmr"]["details"]
        rows, tally = [], {v: [0, 0] for v in variants}
        same = 0
        for d in details:
            if ids and d["id"] not in ids:
                continue
            store = Path(d.get("store") or "")
            if not (store / "brain.db").exists():
                print(f"{path.name} {d['id']}: no kept store ({d.get('store')}), skipped")
                continue
            try:
                got = run_one(store, d["question"], variants)
            except Exception as e:  # one bad store must not end the run
                print(f"{path.name} {d['id']}: replay failed: {str(e)[-300:]}")
                continue
            row = {"id": d["id"], "run_judge": d.get("judge"), "run_answer": d.get("answer"),
                   "expected": d["expected"], "variants": {}}
            for v in variants:
                a = got[v]["answer"]
                verdict = (False if not a.strip() or is_non_answer(a)
                           else judge_one(judge, d["question"], d["expected"], a)["judge"])
                tally[v][0] += verdict
                tally[v][1] += 1
                row["variants"][v] = {**got[v], "judge": verdict, "f1": f1(a, d["expected"])}
            same += got.get("asis", {}).get("answer", "").strip() == (d.get("answer") or "").strip()
            rows.append(row)
            if args.show:
                marks = " ".join(f"{v}={'Y' if row['variants'][v]['judge'] else '-'}" for v in variants)
                print(f"{d['id']:<10} run={'Y' if d.get('judge') else '-'} {marks}  gold: {d['expected'][:60]}")
                for v in variants:
                    print(f"{'':<12}{v:<9} {row['variants'][v]['sections']}  {row['variants'][v]['answer'][:110]!r}")
        n = len(rows)
        print(f"\n{path.name}: {n} questions replayed; run accuracy on them "
              f"{sum(1 for r in rows if r['run_judge']) / n if n else 0:.3f}")
        if "asis" in variants:
            print(f"  asis reproduced the run's answer verbatim on {same}/{n}")
        for v in variants:
            ok, m = tally[v]
            print(f"  {v:<9} {ok / m if m else 0:.3f}  ({ok}/{m})")
        report[str(path)] = rows
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

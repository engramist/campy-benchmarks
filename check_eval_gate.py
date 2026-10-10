#!/usr/bin/env python3
"""
campy-benchmarks / check_eval_gate.py

Self-check for diag_answer_replay.py --hippocampy / --llm-model and eval_gate.py,
with no model and no real store: two fake hippocampy trees (a `campy` package each,
with just the modules the replay worker imports) answer a fixed question set
differently, and the real worker/replay/gate code runs against them in real
subprocesses. A fake judge compares the gold answer to the answer.

What it pins:
  - the worker imports campy from the tree given by --hippocampy (campy_file),
    not from the other one, and a path with no campy package is an error
  - the gate table counts gained/lost correctly in both directions and gives
    real / regression / within noise per the plan (C.4), including the
    "gained 2" and "noise" edges, with the same judge for both sides
  - identical trees can never report a change (the broken-comparison case)
  - --llm-model changes [llm].model in the COPY only; the kept store is untouched
  - set_llm_model on config shapes: no [llm], missing model, comments, [llm.sub]
  - judge verdicts are cached, so an identical answer is never judged two ways
  - a LoCoMo-10 result with --store (category 5 scored by abstention)
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import diag_answer_replay as rep  # noqa: E402
import eval_gate as gate  # noqa: E402

fails = []


def check(name, cond, detail=""):
    if not cond:
        fails.append(f"{name} {detail}".rstrip())


QS = {f"q{i}": f"a{i}" for i in range(1, 7)}  # id -> gold


def make_tree(root: Path, mark: str, correct: set) -> Path:
    """A fake hippocampy tree: campy/ with the modules diag_answer_replay's worker imports."""
    pkg = root / "campy"
    files = {
        "__init__.py": f"MARK = {mark!r}\n",
        "paths.py": "import os\nfrom pathlib import Path\n"
                    "def get_database_path():\n    return Path(os.environ['CAMPY_HOME']) / 'brain.db'\n",
        "brain/__init__.py": "",
        "brain/brainstem/__init__.py": "",
        "brain/brainstem/config.py": textwrap.dedent("""
            import os, tomllib
            from pathlib import Path
            def load_config():
                with open(Path(os.environ['CAMPY_HOME']) / 'config.toml', 'rb') as f:
                    return tomllib.load(f)
            """),
        "brain/hippocampus/__init__.py": "",
        "brain/hippocampus/graph/__init__.py": "",
        "brain/hippocampus/graph/oxigraph_client.py": "class OxigraphClient:\n    def __init__(self, path):\n        self.path = path\n",
        "brain/thalamus/__init__.py": "",
        "brain/thalamus/bundle_compiler.py": "async def _stage_conversation(db, q, cfg):\n    return None\n",
        "brain/thalamus/ask.py": textwrap.dedent(f"""
            import campy
            CORRECT = {sorted(correct)!r}
            class _Sec:
                section_type = 'conversation'
                def __init__(self):
                    self.content = [{{'text': campy.MARK}}]
                    self.token_estimate = 1
            class _Bundle:
                def __init__(self):
                    self.sections = [_Sec()]
            async def compile_bundle(query, *a, **k):
                return _Bundle()
            class _LLM:
                def __init__(self, model):
                    self.model = model
                async def achat(self, messages, **kw):
                    q = messages[-1]['content']
                    ans = ('The answer is ' + 'a' + q[1:]) if q in CORRECT else 'It was something else entirely.'
                    return ans + ' [' + self.model + ']'
            def _get_llm(config):
                return _LLM(config['llm']['model'])
            async def run_ask(query, session_id, db, config, token_budget=32000, capture=True, meta=None, budget_tokens=None):
                await compile_bundle(query)
                return await _get_llm(config).achat([{{'role': 'user', 'content': query}}])
            """),
    }
    for rel, text in files.items():
        p = pkg / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


def make_store(path: Path, model: str = "base-model") -> Path:
    path.mkdir(parents=True)
    (path / "brain.db").write_text("x")
    (path / "config.toml").write_text(f'[llm]\nprovider = "ollama"\nmodel = "{model}"  # kept\n')
    return path


def dmr_result(path: Path, stores) -> Path:
    # The fake LLM answers by question text, so the "question" is the id (q1..); gold is a1..
    details = [{"id": qid, "question": qid, "expected": gold, "answer": "old", "judge": False,
                "store": str(stores[0] if i < 3 else stores[1])} for i, (qid, gold) in enumerate(QS.items())]
    path.write_text(json.dumps({"suites": {"dmr": {"details": details,
                                                    "dataset": {"options": {"max_questions": 6, "offset": 50}}}}}))
    return path


class FakeJudge:
    """Reads gold and answer from the judge prompt; CORRECT when the answer contains the gold."""
    def __init__(self):
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        m = re.search(r"Gold answer: (.*)\nGenerated answer: (.*)\n", messages[-1]["content"])
        return {"text": "CORRECT" if m.group(1) in m.group(2) else "WRONG"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        stores = [make_store(t / "s1"), make_store(t / "s2")]
        res = dmr_result(t / "dmr.json", stores)
        before = {p: sha(p / "config.toml") for p in stores}
        judge = FakeJudge()
        passed = rep.make_passed(judge)

        def trees(main_ok, branch_ok):
            return (make_tree(t / f"m{len(main_ok)}{sorted(main_ok)}", "main", main_ok),
                    make_tree(t / f"b{len(branch_ok)}{sorted(branch_ok)}", "branch", branch_ok))

        def run(main_ok, branch_ok, **kw):
            m, b = trees(set(main_ok), set(branch_ok))
            return gate.gate([res], passed, m, b, **kw), m, b

        # 1. improvement: gained 3, lost 0, +0.5 -> real; code really comes from each tree
        out, m, b = run({"q1", "q2"}, {"q1", "q2", "q3", "q4", "q5"})
        c = out["suites"][0]
        check("real: counts", (c["gained"], c["lost"], c["n"]) == (3, 0, 6), str(c["gained_ids"]))
        check("real: scores", abs(c["main"] - 2 / 6) < 1e-9 and abs(c["branch"] - 5 / 6) < 1e-9, str((c["main"], c["branch"])))
        check("real: verdict", c["verdict"] == "real", c["verdict"])
        check("worker imported main's campy", str(m.resolve()) in c["code"]["main"] and str(b.resolve()) not in c["code"]["main"],
              str(c["code"]))
        check("worker imported the branch's campy", str(b.resolve()) in c["code"]["branch"], str(c["code"]))
        check("the bundle came from the right tree",
              c["rows"]["main"][0]["variants"]["asis"]["sections"] == [["conversation", 1]], "")
        check("latency recorded", c["latency_ms"]["main"] and c["latency_ms"]["main"]["p50"] is not None, str(c["latency_ms"]))
        check("split from the result's selection", c["label"] == "dmr (held-out)", c["label"])
        table = gate.render(out)
        check("table header", gate.HEADER in table, table)
        check("table row", "| dmr (held-out) | 0.333 | 0.833 | +0.500 | 3 | 0 | ±0.04 | real |" in table, table)
        check("gained ids listed", "gained: q3, q4, q5" in table, table)

        # 2. regression: the swap
        out, _, _ = run({"q1", "q2", "q3", "q4", "q5"}, {"q1", "q2"})
        c = out["suites"][0]
        check("regression", (c["gained"], c["lost"], c["verdict"]) == (0, 3, "regression"), str(c["verdict"]))

        # 3. the broken comparison: same answers on both sides can never report a change
        out, _, _ = run({"q1", "q2", "q3"}, {"q1", "q2", "q3"})
        c = out["suites"][0]
        check("identical trees: nothing gained or lost", (c["gained"], c["lost"], c["delta"]) == (0, 0, 0), str(c["delta"]))
        check("identical trees: within noise", c["verdict"] == "within noise", c["verdict"])

        # 4. edges: net 2 is not enough; net 3 with a swap of one is; mean inside noise is not real
        out, _, _ = run({"q1"}, {"q1", "q2", "q3"})
        check("net +2: within noise", out["suites"][0]["verdict"] == "within noise", out["suites"][0]["verdict"])
        out, _, _ = run({"q1", "q2"}, {"q1", "q3", "q4", "q5", "q6"})
        c = out["suites"][0]
        check("gained 4 lost 1: real", (c["gained"], c["lost"], c["verdict"]) == (4, 1, "real"), str(c["verdict"]))
        out, _, _ = run({"q1"}, {"q1", "q2", "q3", "q4"}, noise=0.60)
        check("a larger noise floor turns real into within noise",
              out["suites"][0]["verdict"] == "within noise" and out["suites"][0]["noise"] == 0.60, "")
        check("verdict function: no sign confusion",
              gate.verdict(3, 0, -0.5, 0.04) == "within noise" and gate.verdict(0, 3, 0.5, 0.04) == "within noise", "")

        # 5. one judge, cached: same answer text is never judged twice
        calls = judge.calls
        run({"q1", "q2", "q3"}, {"q1", "q2", "q3"})
        check("judge cached on (question, gold, answer)", judge.calls == calls, f"{judge.calls - calls} extra calls")

        # 6. --llm-model: set in the copy, the kept store is untouched
        out, _, _ = run({"q1"}, {"q1"}, llm_model="big-model")
        a = out["suites"][0]["rows"]["main"][0]["variants"]["asis"]["answer"]
        check("--llm-model reaches the worker", a.endswith("[big-model]"), a)
        check("answer model recorded", out["suites"][0]["answer_model"] == {"main": "big-model", "branch": "big-model"}, "")
        check("without it the store's model is used",
              run({"q1"}, {"q1"})[0]["suites"][0]["rows"]["main"][0]["variants"]["asis"]["answer"].endswith("[base-model]"), "")
        check("the kept stores were not written", {p: sha(p / "config.toml") for p in stores} == before, "")

        # 7. a tree with no campy package is an error, not a silent fallback
        empty = t / "empty"
        empty.mkdir()
        try:
            rep.replay(stores[0], ["q1"], ["asis"], hippocampy=empty)
            check("no campy in --hippocampy raises", False)
        except RuntimeError as e:
            check("no campy in --hippocampy raises", "no campy" in str(e), str(e))

        # 8. set_llm_model on config shapes
        import tomllib

        def cfg(text, model="m2"):
            p = t / "c.toml"
            p.write_text(text)
            rep.set_llm_model(p, model)
            return tomllib.loads(p.read_text()), p.read_text()

        d, _ = cfg('[retrieval]\nk = 1\n')
        check("no [llm]: appended", d["llm"]["model"] == "m2" and d["retrieval"]["k"] == 1, str(d))
        d, _ = cfg('[llm]\nprovider = "x"\n[retrieval]\nmodel = "keep"\n')
        check("[llm] without model: inserted, other tables' model untouched",
              d["llm"] == {"provider": "x", "model": "m2"} and d["retrieval"]["model"] == "keep", str(d))
        d, text = cfg('[llm]\nmodel = "old"  # c\nprovider = "x"\n[llm.step]\nmodel = "sub"\n')
        check("model replaced, [llm.step] untouched",
              d["llm"]["model"] == "m2" and d["llm"]["step"]["model"] == "sub" and d["llm"]["provider"] == "x", text)
        try:
            rep.set_llm_model(t / "absent.toml", "m")
            check("missing config.toml raises", False)
        except RuntimeError:
            pass

        # 9. LoCoMo-10: --store, category 5 by abstention
        l10_store = make_store(t / "l10")
        details = [
            {"id": "conv-30_q0", "category": 1, "question": "q1", "expected": "a1", "answer": "old", "judge": False},
            {"id": "conv-30_q1", "category": 5, "question": "q2", "expected": "a2", "answer": "old", "passed": False},
        ]
        l10 = t / "l10.json"
        l10.write_text(json.dumps({"suites": {"locomo10": {
            "details": details, "dataset": {"options": {"conversation_ids": ["conv-30"], "max_questions": 60}}}}}))
        m = make_tree(t / "l10m", "main", set())
        # category 5 passes when the answer abstains; the fake tree's wrong answer does not abstain
        o = gate.gate([l10], passed, m, m, store=l10_store)
        c = o["suites"][0]
        check("locomo10 replayed with --store", c["n"] == 2 and c["label"] == "locomo10 (held-out)", str((c["n"], c["label"])))
        check("locomo10 identical trees: within noise", c["verdict"] == "within noise", "")
        check("locomo10 conv-26 is dev", gate.split_of("locomo10", {"conversation_ids": ["conv-26"]}) == "dev"
              and gate.split_of("locomo10", {}) == "dev" and gate.split_of("dmr", {"offset": 0}) == "dev", "")

        # 10. percentile
        check("p50/p95", gate.percentile([1, 2, 3, 4, 5], 0.5) == 3 and abs(gate.percentile([0, 100], 0.95) - 95) < 1e-9, "")

    if fails:
        print(f"FAIL -- {len(fails)} eval-gate checks failed:")
        for f in fails:
            print(f"  {f}")
        return 1
    print("OK -- replay --hippocampy/--llm-model and the eval gate table checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

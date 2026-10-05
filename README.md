# Campy Benchmarks Harness

Independent evaluation and benchmarking suite for HippoCampy.

## Architecture
This repository is an isolated consumer harness that tests HippoCampy strictly over standard Model Context Protocol (MCP) transport (`CAMPY_MCP_CMD`). It contains zero internal engine imports, preserving the clean layer boundaries defined in `hippocampy/docs/ecosystem-rules.md`.

## Benchmark Suites
1. **LoCoMo fixture (`locomo/`):** a small synthetic fixture (25 hand-written scenarios, 28 probes) for multi-session recall and constraint updates (deprecation). It borrows the name, not the data; for the real benchmark see LoCoMo-10 below.
2. **MemoryGym (`memory_gym/`):** 2D Grid RL/spatial persistence across extended step horizons (NeurIPS 2023).
3. **MemBench (`membench/`):** Multi-Session Chat (MSC) persona memory and contradictory belief resolution.
4. **ARC Bridge (`arc_bridge/`):** Integrates memory-transfer diagnostics from the sibling `ARC_AGI` repo.
5. **LoCoMo-10 (`locomo10/`, `--suite locomo10`):** the published LoCoMo dataset (Maharana et al., ACL 2024): 10 long conversations, 5,882 turns, 1,986 questions. Not part of `--suite all`, because a full run takes hours on a local model.

## Quickstart

```bash
# Set pointer to Campy MCP server
export CAMPY_MCP_CMD="/Users/djshelton/Desktop/GitProjects/hippocampy/.venv/bin/python -m campy.adapters.mcp_server"

# Scorer + isolation self-tests (no daemon needed; run before trusting a score)
python check_scorers.py
python check_suite_order_independence.py
python check_repeat.py
python check_qa_judge.py

# Before trusting a judge model: grade the ~600 hand-labelled answers with it
python check_qa_judge.py --live --judge-model gemma4:26b

# Run smoke test
python run_all.py --smoke

# Full run on a throwaway daemon with its own empty store (see "Isolated mode");
# writes results/<utc>-<harness sha>.json with per-probe details
python run_all.py --baseline --isolated

# Score reference systems next to Campy (same LLM, same judge; see "Baselines")
python run_all.py --baseline --isolated --baselines all

# The published LoCoMo benchmark: start with one conversation, then scale up
python check_locomo10.py      # downloads + verifies the dataset, checks scoring
python run_all.py --baseline --isolated --suite locomo10 --baselines all --locomo10-conversations 1

# Before trusting a change: 3 runs, each on a fresh store (see "Repeated runs")
python run_all.py --baseline --isolated --repeat 3

# Compare against an earlier result file (warns when runs are not like-for-like)
python run_all.py --baseline --isolated --compare results/<earlier>.json
```

## Scoring (scorer v4)

LoCoMo and MemBench answers are judged by `scoring.py`, not per-probe raw
regexes. A "value" probe passes when the answer states the current value and,
if it also mentions a superseded value, marks it as superseded. A "no" probe
passes when the answer is negative and gives a grounded reason. Non-answers
("I don't know", "No relevant context was found in memory") always fail.

`check_scorers.py` is the contract: every probe's gold answer and natural
paraphrases must pass; non-answers and wrong answers must fail. Scorer v1
failed two of its own gold answers and passed "I don't know" on two probes, so
**results recorded before scorer v2 (`post_cutover_live*.json`) are not
comparable** with later ones. `--compare` says so when the versions differ.
Scorer v3 (2026-09-30) only widens the words that count as marking an old
value retired ("prohibits", "forbids", "disallowed", "banned"). v2 failed
correct answers phrased "...and prohibits the use of X or Y", so v2 LoCoMo
scores can be slightly low; v3 never fails an answer v2 passed.
Scorer v4 (2026-10-01) adds "update" and "change" for the same reason
("Casey updates their preference from AWS ECS to Google Cloud Run").
Each word-list fix only covers phrasings seen so far, which is why the LLM
judge below now runs alongside it.

The judge is still lexical and has known blind spots (e.g. "we use Docker
Swarm, not Kubernetes" passes). An LLM judge is the planned replacement.

Headline metrics: LoCoMo `accuracy` / `deprecation_accuracy`, MemBench
`accuracy` / `contradiction_score`. LoCoMo `f1` is token overlap with a short
gold string and understates full-sentence answers. MemBench `avg_latency_ms`
is `ask` alone (`compile_context` is `avg_compile_latency_ms`). ARC scores are
measured values: a retrieval miss is 0, not the old hard-coded 0.85/0.9.

## Isolated mode (`--isolated`)

With a real daemon (`CAMPY_MCP_CMD` set), a run needs `--isolated`; it is
refused otherwise. Without isolation a run uses your personal daemon and
`~/.campy` store: benchmark turns land in your memory, earlier runs'
identical turns contaminate later runs, and scores depend on whatever else is
in the store. That happened: earlier runs left ~2,000 fixture turns in a
personal store, and consolidation turned them into Concepts and edges that
`ask` read as the user's own decisions (hippocampy B467). `--shared-store`
allows such a run when that is really intended. `--isolated`
(requires hippocampy with `CAMPY_HOME` support, B456):

1. creates a temp `CAMPY_HOME` under `/tmp` and writes its `config.toml` from
   your config (`CAMPY_BENCH_CONFIG`, else `~/.campy/config.toml`, else the
   repo's `campy.toml`) with overrides: capture off, self-restart and
   watchdog off (both restart by exiting and rely on launchd), a free web port;
2. starts `python -m campy.brain_daemon` there, using the python from
   `CAMPY_MCP_CMD`, and waits for its socket (`--daemon-ready-timeout`,
   default 900 s, because cold start loads models);
3. runs the MCP adapter pinned to that daemon's socket **and** HTTP URL. The
   adapter falls back to HTTP when the socket fails, and its default URL is
   your personal daemon on :7799, so pinning only the socket would not be
   isolation. `BRAIN_URL`, `SIDEQUESTS_*` and `CAMPY_BRAIN_SOCKET`, which
   outrank the pins, are removed from its environment;
4. stops the daemon and deletes the store (`--keep-store` keeps it).

Two guards refuse to run instead of silently using the personal store. Before
launch, the daemon's Python must resolve `runtime_dir()` to the temp home; a
pre-B456 hippocampy fails this, because it would honor the socket pin but
still open `~/.campy`. After startup, a `brain.db`/`vectors.db` must exist
inside the temp home. Use `--daemon-python` when `CAMPY_MCP_CMD` doesn't
start with the Python executable (e.g. `uv run ...`).

The store starts **empty**: no personal memory and no earlier runs. Scores
are reproducible, but they are not comparable with personal-store runs,
which contain thousands of unrelated messages as distractors. `--compare`
warns when isolation differs. `provenance.store` records the overrides,
ready time, and the isolated activity-log line count, which should roughly
match `client_calls` and shows the calls reached the isolated daemon.

## LLM judge for LoCoMo and MemBench

The lexical scorer above needs a cue word whenever an answer mentions an old
value. Each new phrasing an LLM uses ("prohibits the use of X", "updates
their preference from X to Y") produced false failures until the word list
caught up (scorer v3, v4). On real-daemon runs, an LLM judge now grades every
LoCoMo and MemBench answer as well. It asks: does the answer give the current
value without presenting an old one as current? (`qa_judge.py`, using the
same approach as LoCoMo-10's judge.)

- It runs **alongside** the lexical scorer. Each record keeps `passed` /
  `reason` and gains `llm_judge` / `llm_judge_reason`. The suites gain
  `judge_accuracy` and `judge_deprecation_accuracy` (LoCoMo) or
  `judge_contradiction_score` (MemBench), all in `--compare`.
- **Disagreements** between the two are printed after the summary and stored
  in `judge_disagreements`. That is where to look first: either a scorer gap
  or a judge mistake.
- Baselines are graded by the same judge in the same run. With `--repeat N`,
  each run is graded before averaging, and per-probe judge verdicts merge by
  majority, like the lexical ones.
- **Judge model:** `--judge-model` (and `--judge-provider`/`--judge-base-url`).
  The default is the baselines' LLM, i.e. the same model that answered,
  which is a weak judge of itself. Prefer a different, stronger local model.
  `--compare` warns when the judge model differs.
- **Validate it first:** `check_qa_judge.py --live --judge-model <m>` grades
  every hand-labelled answer from `check_scorers.py` and exits non-zero below
  95% agreement, listing each lenient or strict call.
- `--judge none` turns it off. Mock runs (no daemon) skip it.

## Repeated runs (`--repeat N`)

One run of an LLM-backed suite is one sample. llama3.1:8b on Ollama at
temperature 0 still rephrases answers between runs, so the lexical judge can
flip a probe with no code change. On 2026-09-30, two LoCoMo probes flipped
between two runs of retrieval-identical code, moving accuracy from 0.89 to
0.82.

`--repeat N` runs the suites N times, each on a **fresh isolated store**.
With a real daemon it requires `--isolated`, because on a shared store run k
would read runs 1..k-1's data. It reports:

- `suites`: each numeric metric as the **mean** over the runs. Each LoCoMo
  and MemBench probe gets a majority verdict (a tie fails) and a `pass_rate`.
  `--compare` and the summary table read this as before.
- `repeat.spread`: min, max and stdev for every metric.
- `repeat.unstable_probes`: probes whose verdict differed between runs. A
  flip on one of these is not evidence of a change.
- `repeat.runs`: every run's full results.

`--compare` marks a change with `≈ (within run-to-run range)` when the
difference is no bigger than the metric's min-max range in either file. Use
`--repeat 3` or more on both sides before calling something a regression.
Repeats multiply the run time: about 20 minutes per full run on a local
llama3.1:8b. `--repeat` can't yet be combined with `--baselines` or
`--suite locomo10`.

## Baselines (`--baselines`)

A Campy score means little alone. `--baselines all` (or a comma list) scores
three reference systems on the QA suites (LoCoMo, MemBench) with the **same
LLM, temperature 0, and the same judge** as Campy. Not to be confused with
`--baseline`, which only saves the results file.

| Baseline | Context in the prompt | What it tells you |
|---|---|---|
| `no_memory` | none; plain "answer as best you can" | Probes it passes are guessable from priors and aren't testing memory |
| `full_context` | every turn the suite has written so far | A near-ceiling on fixtures this small (LoCoMo ≈ 7.5k tokens as formatted) |
| `naive_rag` | top-k raw turns by similarity, oldest first | The simplest retrieval system; Campy's consolidation should beat it |

- **Same prompt as Campy.** The context-bearing baselines reuse `ask`'s
  system prompt, its `<retrieved_memory>` wrapping and its empty/non-empty
  instruction lines, copied into `baselines.py` (keep them in sync by hand).
  They differ from Campy only in the context supplied.
- **LLM:** the base `[llm]` section Campy's `ask` uses: the isolated
  daemon's config, else the config provenance finds. Override it with
  `--baseline-provider/--baseline-model/--baseline-base-url`. Ollama is
  called through its native `/api/chat` with `num_ctx` sized to each prompt,
  because its OpenAI-compatible endpoint silently drops the start of prompts
  longer than the server default. That would cut the full transcript,
  system prompt included. The `num_ctx` used is recorded per question.
- **`naive_rag` retriever:** fastembed with the same model Campy embeds with
  (`--rag-k`, default 5). `--rag-retriever auto` falls back to BM25 with a
  warning when fastembed isn't installed; the retriever used is recorded.
- **Scope:** each suite's own turns. In an isolated run Campy's store also
  holds earlier suites' data, so Campy faces slightly more distractors.
- **Independent of the daemon.** Baselines run after the Campy suites;
  `--baselines-only` skips the daemon entirely.

Output: `baselines.<name>.<suite>` (metrics plus per-question `details`) and
`campy_vs_baselines`. The latter lists, per suite, the probes a baseline
passes that Campy fails (the actionable list), the reverse, and the probes
that pass with no memory at all.

## LoCoMo-10 (`--suite locomo10`)

**License:** the dataset is **CC BY-NC 4.0 (non-commercial)**. It is not
committed here. The first run downloads it into `data/` (gitignored), pinned
to snap-research/locomo commit `3eb6f2c5` and verified by sha256; a git fetch
is the fallback when raw.githubusercontent.com is blocked. `LOCOMO10_PATH`
points at an existing copy. The scoring code is this harness's own, written
from the official evaluation's behaviour. Check that non-commercial terms fit
your use before publishing results.

**Protocol:**
- **Ingestion:** each turn is written as role `user` with the session date and
  speaker in its text: `[1:56 pm on 8 May, 2023] Caroline: …`, plus
  `[shares an image: <caption>]` for image turns. `notify_turn` has no
  timestamp parameter, and temporal questions need the dates. Consolidation
  settles once per conversation, before its questions.
- **Temporal questions:** category 2 questions get LoCoMo's "answer with an
  approximate date" instruction, reworded for a memory context. Campy and
  every baseline receive the same text.
- **Per question:** `compile_context` (retrieval diagnostic), then `ask`.

**Metrics** (all systems scored identically; `details` holds every answer):

| Metric | What it is |
|---|---|
| `judge_accuracy` | **Headline.** An LLM judge grades categories 1–4 CORRECT/WRONG against the gold answer (the "J" memory-system papers report). Campy and all baselines are judged in one pass by the same judge. The prompt is this harness's own, so it's comparable across runs of this harness with the same judge model, not directly with published numbers. The judge defaults to the baselines' LLM; `--judge-model` sets a stronger one (recommended). `--judge none` skips it. |
| `f1` | The official token F1 (categories 1–4): category 1 averages comma-separated parts, category 3 uses the text before `;`. It matched the official implementation on 7,418 answer pairs. It stems only when `nltk` is installed (`f1_stemmed`); `pip install nltk` for paper-comparable F1. |
| `adversarial_abstention` | Category 5 (the event never happened): the share of answers that decline. The `_strict` variant uses only the official phrases ("not mentioned", "no information available"). The main one also accepts other declines, e.g. Campy's "No relevant context was found in memory". The official protocol offers two options; here every system gets the bare question. |
| `evidence_recall` | The share of the turns the gold answer cites that retrieval surfaced. For Campy: `compile_context`'s items matched back to turn text. For `naive_rag`: exact. `full_context` is 1.0 by construction. This separates "retrieval missed it" from "the LLM misread it". |
| `by_category` | All of the above per category: multi_hop, temporal, open_domain, single_hop, adversarial (names from the official evaluation code's comments). |

**Baselines:** `full_context` is the question's own conversation, about 32k
tokens as formatted (Ollama gets `num_ctx` 65536), and slow on a local 8B
model. `naive_rag` retrieves over all turns written so far, the same corpus
Campy's store holds.

**Subsets:** `--locomo10-conversations N`, `--locomo10-max-questions N` (taken
round-robin across categories, so a subset stays balanced) and
`--locomo10-categories 1,2,3,4`. `--smoke` defaults to 1 conversation and 25
questions. `--compare` warns when the subset or judge model differs.

## Result files and provenance

Each result file contains:

- `provenance`: scorer version, harness and hippocampy commits (plus dirty
  flag), the daemon's LLM and embedding models, a sha256 of each suite's
  fixtures, and the command line.
- `suites.<name>.details`: every probe's question, answer, verdict and reason
  (MemBench also records a summary of the `compile_context` bundle;
  `--trace-context` adds that to LoCoMo at the cost of one extra call per
  probe).

The daemon resolves its config from its own working directory, which the
harness can't see, so the model fields are **best-effort** and labelled with
their `source`. Set these environment variables to make provenance exact:

| Variable | Purpose |
|---|---|
| `CAMPY_BENCH_CONFIG` | Path to the config file the daemon actually loaded |
| `CAMPY_REPO_PATH` | hippocampy checkout (otherwise inferred from `CAMPY_MCP_CMD`) |
| `ARC_AGI_REPO` | Sibling ARC_AGI checkout for the ARC bridge |

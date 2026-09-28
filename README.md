# Campy Benchmarks Harness

Independent evaluation and benchmarking suite for HippoCampy.

## Architecture
This repository is an isolated consumer harness that tests HippoCampy strictly over standard Model Context Protocol (MCP) transport (`CAMPY_MCP_CMD`). It contains zero internal engine imports, preserving the clean layer boundaries defined in `hippocampy/docs/ecosystem-rules.md`.

## Benchmark Suites
1. **LoCoMo (`locomo/`):** Long-context multi-session conversational recall and dynamic constraint updates (deprecation).
2. **MemoryGym (`memory_gym/`):** 2D Grid RL/spatial persistence across extended step horizons (NeurIPS 2023).
3. **MemBench (`membench/`):** Multi-Session Chat (MSC) persona memory and contradictory belief resolution.
4. **ARC Bridge (`arc_bridge/`):** Integrates memory-transfer diagnostics from the sibling `ARC_AGI` repo.

## Quickstart

```bash
# Set pointer to Campy MCP server
export CAMPY_MCP_CMD="/Users/djshelton/Desktop/GitProjects/hippocampy/.venv/bin/python -m campy.adapters.mcp_server"

# Scorer + isolation self-tests (no daemon needed; run before trusting a score)
python check_scorers.py
python check_suite_order_independence.py

# Run smoke test
python run_all.py --smoke

# Full run; writes results/<utc>-<harness sha>.json with per-probe details
python run_all.py --baseline

# Recommended: a throwaway daemon with its own empty store (see "Isolated mode")
python run_all.py --baseline --isolated

# Score reference systems next to Campy (same LLM, same judge; see "Baselines")
python run_all.py --baseline --isolated --baselines all

# Compare against an earlier result file (warns when runs are not like-for-like)
python run_all.py --baseline --compare results/<earlier>.json
```

## Scoring (scorer v2)

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

The judge is still lexical and has known blind spots (e.g. "we use Docker
Swarm, not Kubernetes" passes). An LLM judge is the planned replacement.

Headline metrics: LoCoMo `accuracy` / `deprecation_accuracy`, MemBench
`accuracy` / `contradiction_score`. LoCoMo `f1` is token overlap with a short
gold string and understates full-sentence answers. MemBench `avg_latency_ms`
is `ask` alone (`compile_context` is `avg_compile_latency_ms`). ARC scores are
measured values: a retrieval miss is 0, not the old hard-coded 0.85/0.9.

## Isolated mode (`--isolated`)

Without it, a run uses your personal daemon and `~/.campy` store: benchmark
turns land in your memory, earlier runs' identical turns contaminate later
runs, and scores depend on whatever else is in the store. `--isolated`
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

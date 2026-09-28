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

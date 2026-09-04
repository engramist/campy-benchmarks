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

# Run smoke test
python run_all.py --smoke

# Run full baseline capture
python run_all.py --baseline --out baseline_snapshot.json
```

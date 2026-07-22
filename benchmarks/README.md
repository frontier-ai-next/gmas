# gMAS benchmark suite

Every benchmark track has one directory, one purpose, and one canonical entry point. Paper results are documented under `docs/benchmarks/`; this directory contains the code and reproduction protocol.

## Layout

| Directory | Purpose | Canonical entry point | Result status |
| --- | --- | --- | --- |
| `topology/` | Matched gMAS/LangGraph topology comparison on reasoning datasets | `python -m benchmarks.topology.run` | Reported in the paper |
| `ablations/` | Paired gMAS runtime-control ablations | `python -m benchmarks.ablations.run` | Reported in the paper |
| `gaia/` | Matched GAIA tool-use comparison and smoke checks | `python -m benchmarks.gaia.run` | Reported in the paper |
| `agentprune/` | AgentPrune on the MultiAgentBench research-idea track | `python -m benchmarks.agentprune.run` | Reported in the paper |
| `terminal_bench/` | Single-agent versus conditional gMAS team on Terminal-Bench 2.0 | Harbor import paths in its README | Harness complete; full matched result pending |
| `exploratory/` | Older or exploratory harnesses retained for follow-up work | Track-specific | Not a source for paper results |

## Setup

Run from the repository root:

```bash
uv sync --extra benchmarks
```

Copy `benchmarks/.env.example` to `benchmarks/.env` and set the shared OpenAI-compatible endpoint:

```text
LLM_API_KEY=...
LLM_BASE_URL=https://your-endpoint.example/v1
LLM_MODEL=your-model
```

The ablation track accepts additional `BENCH_*` variables for strong/weak model assignments, worker counts, and statistical settings.

## Output contract

Write generated artifacts to `benchmark_logs/` or `benchmarks_logs/`. These directories are ignored by Git. A publishable result must preserve:

- the task IDs, split, topology, model alias, prompts, and runner settings;
- raw per-system latency, provider token usage, calls, predictions, and errors;
- checkpoint or resume state for long runs;
- aggregate reducers and paired statistical tests;
- enough trace data to reconstruct which agents and tools executed.

Do not copy an aggregate into the documentation without retaining its corresponding run artifacts outside the repository.

## Quick validation

```bash
uv run pytest tests/test_gaia_benchmark_fairness.py tests/test_terminal_bench_benchmark.py -q
uv run python -m benchmarks.gaia.run --dry-run --max-samples 3
```

GAIA browser variants also require `uv run playwright install chromium`.

## Reported results

See the documentation pages for [matched topology](../docs/benchmarks/topology.md), [runtime ablations](../docs/benchmarks/ablations.md), [GAIA](../docs/benchmarks/gaia.md), and [AgentPrune](../docs/benchmarks/agentprune.md). Each page keeps the metric definition and caveat next to the number.

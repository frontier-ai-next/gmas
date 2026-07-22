# gMAS Terminal-Bench comparison

This benchmark compares two gMAS configurations on the 89 tasks in Terminal-Bench 2.0:

- `GMASSingleAgent`: one autonomous agent with the complete terminal tool set;
- `GASMASAgent`: a conditional multi-agent graph with planning, execution, debugging, and verification roles.

Both variants use the same OpenAI-compatible model endpoint, tool implementations, task dataset, timeout, and Harbor
environment. The agents write token usage, execution order, message traces, tool outputs, condition checks, and elapsed time
to the Harbor agent context.

## Requirements

- Linux or WSL2;
- Docker with a running daemon (`docker info` must succeed);
- Python 3.12 or 3.13 and `uv`;
- access to an OpenAI-compatible chat-completions endpoint with structured-output support.

Harbor is pinned to `0.20.0` in the commands below. Run every command from the repository root.

## Configuration

Set the model connection before starting a run:

```bash
export LLM_API_KEY="..."
export LLM_BASE_URL="https://your-endpoint.example/v1"
export LLM_MODEL="your-model-name"
```

`LLM_BASE_URL` is optional when the standard OpenAI endpoint is used. Do not commit credentials or generated Harbor
artifacts; output paths below are covered by the repository's `benchmarks_logs/` ignore rule.

## Smoke test

Run the same representative task with both variants before launching the complete benchmark:

```bash
uv run --with "harbor==0.20.0" harbor run \
  -d terminal-bench@2.0 \
  --task-name break-filter-js-from-html \
  --agent-import-path benchmarks.terminal_bench.gmas_agent:GMASSingleAgent \
  --job-name gmas-single-smoke \
  -o benchmarks_logs/terminal_bench/single-smoke

uv run --with "harbor==0.20.0" harbor run \
  -d terminal-bench@2.0 \
  --task-name break-filter-js-from-html \
  --agent-import-path benchmarks.terminal_bench.gmas_mas_agent:GASMASAgent \
  --job-name gmas-mas-smoke \
  -o benchmarks_logs/terminal_bench/mas-smoke
```

## Full comparison

Use identical concurrency and attempt counts so that the two variants are directly comparable:

```bash
uv run --with "harbor==0.20.0" harbor run \
  -d terminal-bench@2.0 \
  --agent-import-path benchmarks.terminal_bench.gmas_agent:GMASSingleAgent \
  --n-concurrent 1 \
  --n-attempts 1 \
  --job-name gmas-single-tbench2 \
  -o benchmarks_logs/terminal_bench/single

uv run --with "harbor==0.20.0" harbor run \
  -d terminal-bench@2.0 \
  --agent-import-path benchmarks.terminal_bench.gmas_mas_agent:GASMASAgent \
  --n-concurrent 1 \
  --n-attempts 1 \
  --job-name gmas-mas-tbench2 \
  -o benchmarks_logs/terminal_bench/mas
```

## Reporting results

Report the following values for each variant from the completed Harbor jobs:

| Metric | Definition |
| --- | --- |
| Passed tasks | Tasks with a successful Harbor verifier result |
| Pass rate | Passed tasks divided by 89 |
| Total tokens | Sum of `total_tokens` from agent context metadata |
| Median task time | Median `total_time` across completed tasks |
| Condition checks | Sum of recorded conditional-edge or completion checks |

Aggregate scores are intentionally not estimated from partial runs. Commit a result table only after both variants have
completed the same task set with the exact configuration above; keep the raw Harbor job directories in
`benchmarks_logs/terminal_bench/`.

## Implementation

- `gmas_agent.py` defines the single-agent baseline.
- `gmas_mas_agent.py` builds the conditional MAS graph.
- `agents_config.json` contains the MAS roles and conditional edges.
- `tools.py` adapts Harbor's container environment to a gMAS `ToolRegistry`.
- `configuration.py` validates the checked-in graph configuration before a run starts.
- `llm.py` provides the shared OpenAI-compatible caller and token accounting.

# Reproducibility contract

A benchmark result is only reviewable when its configuration and route through the system can be reconstructed. For gMAS evaluations, preserve the following with every run.

## Required run artifact

- serialized graph before and after execution;
- prompt and role definitions;
- model and endpoint identifiers, without credentials;
- decoding parameters and concurrency limits;
- random seeds where applicable;
- provider token-usage fields;
- normalized predictions and reference answers;
- JSONL or equivalent execution traces;
- dataset name, version, split, and sample identifiers;
- topology template and metric reducer;
- baseline and gMAS runner configuration.

Tool-enabled benchmarks must also record tool availability, search/browser settings, exact tool budgets, answer-normalization logic, and infrastructure-error classification.

## Common setup

```bash
uv sync --extra benchmarks
Copy-Item benchmarks/.env.example benchmarks/.env  # PowerShell
```

On POSIX shells, use `cp benchmarks/.env.example benchmarks/.env`. Fill in an OpenAI-compatible endpoint through `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL`. The ablation track also accepts `BENCH_*` overrides for strong/weak models and experiment controls.

Generated outputs belong under `benchmark_logs/` or `benchmarks_logs/`; both paths are excluded from Git to avoid committing credentials, large provider payloads, or transient browser artifacts.

## Entry points

| Track | Command |
| --- | --- |
| Matched topology | `uv run python -m benchmarks.topology.run --help` |
| Runtime ablations | `uv run python -m benchmarks.ablations.run --help` |
| GAIA | `uv run python -m benchmarks.gaia.run --help` |
| AgentPrune | `uv run python -m benchmarks.agentprune.run` |
| Terminal-Bench | See `benchmarks/terminal_bench/README.md` |

## Reporting checklist

1. Run a smoke test before the complete benchmark.
2. Keep both compared systems on the same task set and operational budget.
3. Include failures in the accuracy denominator unless a documented infrastructure rule excludes them.
4. Report latency, tokens, and quality separately; do not collapse them into “faster” or “better.”
5. Label non-significant accuracy changes as observed effects.
6. Keep exploratory harnesses out of paper result tables until they have a matched protocol and a complete run.

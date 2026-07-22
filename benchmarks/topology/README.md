# Matched topology benchmark

Compares gMAS and LangGraph with matched roles, prompts, models, datasets, stop conditions, and concurrency limits.

```bash
uv sync --extra benchmarks
uv run python -m benchmarks.topology.run --datasets big-bench-hard,gsm8k,mmlu-pro --runs 1
```

Use `--max-samples` for a smoke run and `--no-resume` only when intentionally discarding an existing checkpoint. The harness writes per-run logs, checkpoints, JSON output, and aggregate rows to the configured `--log-dir`.

Published aggregate tables and limitations: `docs/benchmarks/topology.md`.

# Runtime-control ablations

Runs six paired gMAS comparisons on BBH: early stop, disabled-node handling, reachability filtering, adaptive topology, hidden channels, and multi-model routing.

```bash
uv sync --extra benchmarks
uv run python -m benchmarks.ablations.run --max-samples 3 --workers 1
```

The primary endpoint uses `LLM_*`. Optional strong/weak model assignments and experiment controls use `BENCH_*`; see `benchmarks/.env.example`.

Outputs are written to `benchmark_logs/ablations/`. Published aggregate tables and significance notes: `docs/benchmarks/ablations.md`.

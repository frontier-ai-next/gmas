# GAIA tool-use benchmark

Compares gMAS and LangGraph on the same GAIA validation tasks under symmetric wall-time and tool-call budgets. File, web, and deep-browser paths are instrumented explicitly, and errors remain in the main accuracy denominator.

```bash
uv sync --extra benchmarks
uv run playwright install chromium
uv run python -m benchmarks.gaia.smoke
uv run python -m benchmarks.gaia.run --dry-run --max-samples 3
uv run python -m benchmarks.gaia.run --levels 1,2,3 --parallel 4
```

Use `--probe-search` to check snippets-only search without making LLM calls. Generated JSON and Markdown reports include overall, per-level, and pairwise common-task summaries.

Published result and limitations: `docs/benchmarks/gaia.md`.

# GAIA tool-use benchmark

This track compares gMAS and LangGraph on the GAIA validation split with the
same tool budget. It is separate from the closed-book topology benchmark because
search and browser behavior account for much of the runtime.

## Protocol

- GAIA validation levels 1–3, 165 tasks.
- The same task set, model, answer normalization, search backend, and five-tool budget.
- Exact final-answer correctness; failures stay incorrect in the main denominator.
- Client latency, provider tokens, LLM calls, tool calls, and budget exhaustion are recorded.
- Entry point: `python -m benchmarks.gaia.run`.

## Results

| System | Tool setting | Accuracy | Latency (s) | Tokens | LLM calls | Tool calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LangGraph | Simple search | 22.4% | 10.78 | 4,986 | 3.52 | 2.32 |
| gMAS | Simple search | 26.7% | 16.26 | 2,714 | 3.16 | 2.13 |
| LangGraph | Deep search | 22.4% | 14.23 | 7,017 | 3.37 | 2.12 |
| gMAS | Deep search | 29.1% | 16.96 | 4,696 | 3.40 | 2.36 |

The LangGraph baseline exhausted the five-tool budget on 34 simple-search and 30 deep-search tasks in this run; gMAS did not exhaust it.

## Interpretation

In these runs, gMAS is more accurate and uses fewer tokens, but takes longer.
The result is specific to this configuration and is not a latency claim.

GAIA is sensitive to search availability, provider behavior, browser versions,
timeouts, and transient network failures. Keep the raw JSON/Markdown output and
record infrastructure exclusions for each rerun.

## Smoke checks

Before a paid or long run:

```bash
uv sync --extra benchmarks
uv run playwright install chromium
uv run python -m benchmarks.gaia.smoke
uv run python -m benchmarks.gaia.run --dry-run --max-samples 3
uv run python -m benchmarks.gaia.run --probe-search
```

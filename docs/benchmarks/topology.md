# Matched topology comparison

This experiment runs gMAS and a LangGraph baseline with the same roles, prompts,
solver models, provider endpoints, decoding parameters, stop conditions, and
concurrency limits.

## Scope

- Datasets: BIG-Bench Hard, GSM8K, and MMLU-Pro.
- Topologies: single agent, chain-3, fan-in, and fan-out.
- Coverage: 68 matched model–dataset–topology configurations across six model aliases from GPT-OSS, Qwen, and Gemma.
- Metrics: normalized accuracy, client-side wall latency, provider token usage, and LLM calls.
- Entry point: `python -m benchmarks.topology.run`.

## Aggregate by topology

| Topology | gMAS / LangGraph latency (s) | Latency Δ | gMAS / LangGraph tokens | Token Δ | Accuracy Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single | 3.96 / 4.13 | -4.0% | 484 / 491 | -1.5% | +0.0 pp |
| Chain-3 | 16.07 / 15.91 | +1.0% | 2,702 / 2,620 | +3.1% | +2.5 pp |
| Fan-in | 20.32 / 22.08 | -8.0% | 4,573 / 4,495 | +1.7% | +3.0 pp |
| Fan-out | 34.60 / 38.16 | -9.3% | 9,410 / 9,416 | -0.1% | +2.9 pp |

Fan-in and fan-out contain the most removable or reorderable work and show the
largest latency reductions. Tokens do not follow the same pattern: chain-3 and
fan-in use slightly more tokens under gMAS in this aggregate.

## Aggregate by dataset

GPT-OSS rows are excluded from this panel because their dataset coverage is incomplete.

| Dataset | gMAS / LangGraph latency (s) | Latency Δ | gMAS / LangGraph tokens | Token Δ | gMAS / LangGraph accuracy | Accuracy Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BBH | 20.23 / 22.18 | -8.8% | 4,461 / 4,424 | +0.8% | 83.9% / 83.6% | +0.3 pp |
| GSM8K | 18.81 / 19.89 | -5.4% | 4,033 / 3,932 | +2.6% | 84.0% / 79.4% | +4.6 pp |
| MMLU-Pro | 17.18 / 18.14 | -5.3% | 4,383 / 4,410 | -0.6% | 68.2% / 66.8% | +1.5 pp |

## Aggregate by model

| Model alias | gMAS / LangGraph latency (s) | Latency Δ | gMAS / LangGraph tokens | Token Δ | gMAS / LangGraph accuracy | Accuracy Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3.5-35-A3B | 23.83 / 25.55 | -6.7% | 4,598 / 4,606 | -0.2% | 83.1% / 81.4% | +1.7 pp |
| qwen3.5-9b | 13.81 / 14.80 | -6.7% | 4,611 / 4,583 | +0.6% | 80.4% / 77.4% | +3.0 pp |
| qwen3.5-4b | 16.25 / 17.34 | -6.3% | 4,540 / 4,539 | +0.0% | 79.1% / 78.7% | +0.4 pp |
| gemma-12b | 21.06 / 22.59 | -6.8% | 3,420 / 3,294 | +3.8% | 72.2% / 68.8% | +3.4 pp |

## Interpretation

In this matched setup, topology and plan controls reduce latency for the more
complex graph shapes. The aggregate accuracy deltas are observations, not a
general accuracy claim; individual model–topology cells include unfavorable
cases.

For the controls responsible for the savings, continue to the [runtime-control ablations](ablations.md).

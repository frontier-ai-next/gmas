# AgentPrune on MultiAgentBench

This track tests learned topology pruning on collaborative research-idea
generation. A ten-agent research-lab graph is trained on three tasks, then
evaluated on the same 97 held-out tasks before and after pruning.

Entry point: `python -m benchmarks.agentprune.run`.

## Scoring

An LLM judge scores innovation, safety, and feasibility on a 1–5 scale. The optimization target penalizes time and tokens:

```text
score = (innovation + safety + feasibility - token_penalty - time_penalty) / 10
token_penalty = total_tokens / 8,000
time_penalty = total_time / 60
```

## Results

The pruned column averages 14 independently pruned graph realizations.

| Metric | Before pruning | After pruning | Change |
| --- | ---: | ---: | ---: |
| Tasks | 97 | 97 | matched |
| Cost-penalized score ↑ | .735±.208 | .822±.200 | +.087 |
| Innovation ↑ | 3.30±.90 | 3.18±.94 | -.12 |
| Safety ↑ | 4.50±.90 | 4.57±.79 | +.07 |
| Feasibility ↑ | 3.00±.80 | 2.92±.82 | -.08 |
| Time (s) ↓ | 78.2±13.5 | 53.0±27.4 | -32.2% |
| Tokens ↓ | 17.2k±3.2k | 12.6k±8.1k | -27.0% |

Across the 14 pruned realizations, the reported mean score change is +0.087 with a 95% confidence interval of [+0.023, +0.150].

## Interpretation

Pruning raises the cost-penalized score on this held-out set. Innovation and
feasibility decrease slightly, while safety increases. Transfer to other graph
types, domains, reward functions, or training splits was not tested.

The checked-in benchmark data and full protocol live under `benchmarks/agentprune/`.

# Benchmarks and reported results

These pages record the evaluation snapshot from the 2026 gMAS manuscript. The
paper tracks and the independent Terminal-Bench harness are listed separately:

| Track | Question | Scope | Details |
| --- | --- | --- | --- |
| Matched topology | How do gMAS and LangGraph behave under the same roles, models, prompts, and graph shapes? | BBH, GSM8K, MMLU-Pro; 68 matched model–dataset–topology configurations | [Results](topology.md) |
| Runtime controls | Which gMAS controls actually change cost or observed accuracy? | Six paired ablations on BBH | [Results](ablations.md) |
| GAIA tool use | What happens under a fixed five-tool budget? | GAIA validation, levels 1–3, 165 tasks | [Results](gaia.md) |
| AgentPrune | Does learned graph pruning improve the cost-penalized research-idea score? | MultiAgentBench research-idea track; 3 train and 97 held-out tasks | [Results](agentprune.md) |
| Terminal-Bench | How does a conditional gMAS team compare with a single gMAS agent on terminal tasks? | Terminal-Bench 2.0, 89 tasks | Harness available; no complete matched result is reported yet |

## Result snapshot

The values below were transcribed from the manuscript generated on 21 July
2026. They describe those runs only; model, provider, task, and graph choices
can change the outcome.

- In the matched topology aggregate, gMAS latency was **8.0% lower for fan-in** and **9.3% lower for fan-out**. Chain-3 latency was 1.0% higher. Aggregate token changes were small and mixed.
- Early stopping was the strongest direct cost control in the ablation: **52.6% fewer tokens**, **45.0% lower latency**, and **3.0 fewer executed agents** on average, with a non-significant observed accuracy change.
- On GAIA, gMAS used fewer tokens and achieved higher accuracy in both reported search settings, while taking longer. This is a correctness/budget result, not a latency win.
- AgentPrune increased the cost-penalized mean score from **0.735 to 0.822** on the fixed held-out set. The mean change across 14 pruned realizations was **+0.087**, with a reported 95% confidence interval of **[+0.023, +0.150]**.

## How to read the numbers

Negative latency or token deltas mean lower operational cost for gMAS. Accuracy deltas are percentage points, not percentages. Unless a statistical test is shown, an accuracy difference is an observed effect only.

The runs do not support a blanket claim that gMAS is faster. They show that:

1. topology controls can avoid work when a graph contains optional or unreachable roles;
2. the largest savings come from explicit stopping and filtering controls;
3. aggregate token use can increase in some topologies;
4. tool-enabled runs may trade latency for accuracy and lower token consumption;
5. learned pruning remains task-, reward-, and configuration-dependent.

See [reproducibility](reproducibility.md) for the run-artifact contract and [`benchmarks/`](https://github.com/frontier-ai-next/gmas/tree/main/benchmarks) for executable entry points.

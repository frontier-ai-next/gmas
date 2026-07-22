# AgentPrune benchmark

Evaluates AgentPrune on a ten-agent research-lab graph with deliberately adversarial roles. The topology is trained on three research-idea tasks and evaluated on the same 97 held-out tasks before and after pruning.

## Run

From the repository root:

```bash
uv sync --extra benchmarks
uv run python -m benchmarks.agentprune.run
```

Set `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` for an OpenAI-compatible endpoint before starting. The complete run evaluates one unpruned graph and 14 independently pruned graph realizations. It is expensive; validate the endpoint and model's structured-output support first.

## Checked-in data

| File | Contents |
| --- | --- |
| `tasks.json` | 100 research-idea tasks: 3 train and 97 test |
| `agents.json` | Ten role definitions, including domain experts and adversarial roles |
| `less_agents.json` | Reduced graph configuration used for focused checks |
| `config.py` | Model, retry, scoring, and pruning constants |
| `run.py` | Graph construction, pruning loop, evaluation, and aggregate logging |

Task domains include LLM/NLP, computer vision, graph neural networks, federated learning, and robotics.

## Protocol

1. Build the unpruned graph with a coordinator, managers, domain experts, adversarial roles, and a final synthesizer.
2. Evaluate the fixed 97-task test split.
3. Train AgentPrune on the three training tasks.
4. Evaluate the resulting pruned graph on the same test split.
5. Repeat the pruning realization 14 times and report the distribution, not one favorable graph.

The LLM judge scores innovation, safety, and feasibility. The optimization target also penalizes tokens and wall time. Results therefore need to report the component scores beside the combined score.

## Output

The run appends aggregate rows to `benchmarks/agentprune/log.csv`; that file is ignored by Git. Preserve the external model identity, endpoint type, decoding settings, raw judge outputs, and per-realization graph in the run artifact.

Published result and limitations: `docs/benchmarks/agentprune.md`.

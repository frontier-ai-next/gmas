# Running Agents

Execute agents on a graph using the MACPRunner.

## Basic Execution

The simplest way to run agents is a synchronous round:

```python
from gmas.core import AgentProfile
from gmas.builder import build_property_graph
from gmas.execution import MACPRunner

agents = [
    AgentProfile(agent_id="researcher", display_name="Researcher"),
    AgentProfile(agent_id="writer", display_name="Writer"),
]
graph = build_property_graph(
    agents,
    workflow_edges=[("researcher", "writer")],
    query="Explain quantum entanglement",
)

def llm_caller(prompt: str) -> str:
    # Your LLM API call here
    return "response text"

runner = MACPRunner(llm_caller=llm_caller)
result = runner.run_round(graph)
```

## Execution Result

`run_round()` returns a `MACPResult` named tuple:

```python
result = runner.run_round(graph)

result.final_answer          # str – final agent's output
result.final_agent_id        # str – ID of the agent that produced the answer
result.execution_order       # list[str] – order agents ran in
result.messages              # dict[str, str] – {agent_id: response}
result.total_tokens          # int – estimated token usage
result.total_time            # float – wall-clock seconds
result.agent_states          # dict | None – updated agent state histories
result.step_results          # dict | None – per-step details
result.errors                # list[ExecutionError] | None – any errors
result.pruned_agents         # list[str] | None – agents skipped by pruning
result.early_stopped         # bool – whether early-stop triggered
result.topology_modifications  # int – number of topology changes
result.budget_summary        # dict | None – budget consumption info
result.metrics               # ExecutionMetrics | None – collected metrics
```

## Async Execution

Use `arun_round()` for async LLM callers:

```python
runner = MACPRunner(async_llm_caller=async_llm_caller)
result = await runner.arun_round(graph)
```

## Execution Strategies

The runner supports two strategies controlled by `RunnerConfig.adaptive`:

### Simple (default)

Agents execute sequentially in topological order. No conditional edges, no dynamic topology changes.

```python
from gmas.execution import RunnerConfig

config = RunnerConfig(adaptive=False)
runner = MACPRunner(llm_caller=llm_caller, config=config)
result = runner.run_round(graph)
```

### Adaptive

Enables conditional edges, early stopping, pruning, fallback agents, and dynamic topology hooks.

```python
config = RunnerConfig(adaptive=True)
runner = MACPRunner(llm_caller=llm_caller, config=config)
result = runner.run_round(graph)
```

## Specifying Start/End Agents

By default, the runner infers start and end agents from the graph topology. You can override:

```python
result = runner.run_round(
    graph,
    final_agent_id="writer",      # force specific final agent
    start_agent_id="researcher",  # force specific start agent
)
```

## Updating Agent States

By default, the runner updates agent states after each round (appending responses to `agent.state`). Disable with:

```python
result = runner.run_round(graph, update_states=False)
```

## Filtering Unreachable Agents

In graphs with disconnected components, filter out agents unreachable from the start node:

```python
result = runner.run_round(graph, filter_unreachable=True)
```

## Execution with Hidden States

Transfer hidden state vectors between agents during execution:

```python
result = runner.run_round_with_hidden(
    graph,
    final_agent_id="writer",
    hidden_encoder=my_encoder,  # optional: encode responses to vectors
)
# result.hidden_states contains dict[agent_id, HiddenState]
```

## Execution Flow

1. **Initialize** – Set up callbacks, budgets, and memory
2. **Build execution order** – Topological sort of the graph
3. **For each agent** (in order):
   - Build prompt (persona + incoming messages + memory + state)
   - Call LLM
   - Save response to messages dict
   - Update memory (if enabled)
   - Check early-stop conditions
   - Apply topology hooks (if adaptive)
4. **Finalize** – Collect results, update metrics, notify callbacks

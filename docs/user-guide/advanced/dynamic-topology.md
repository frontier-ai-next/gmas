# Dynamic Topology

Modify the graph structure during execution.

## Overview

Dynamic topology allows the agent graph to change at runtime — agents can be added, removed, skipped, or rerouted based on intermediate results. This is a key differentiator of gMAS.

## TopologyAction

A `TopologyAction` describes modifications to apply after an agent executes:

| Field | Type | Description |
|-------|------|-------------|
| `early_stop` | `bool` | Stop execution immediately |
| `early_stop_reason` | `str \| None` | Reason for early stop |
| `add_edges` | `list[tuple[str, str, float]]` | Edges to add (source, target, weight) |
| `remove_edges` | `list[tuple[str, str]]` | Edges to remove |
| `skip_agents` | `list[str]` | Agents to skip entirely |
| `force_agents` | `list[str]` | Force-execute previously skipped agents |
| `condition_skip_agents` | `list[str]` | Skip agents via conditional edge logic |
| `condition_unskip_agents` | `list[str]` | Unskip conditionally skipped agents |
| `insert_chains` | `list[tuple[str, str]]` | Insert agent chains after a node |
| `new_end_agent` | `str \| None` | Change the final agent |
| `trigger_rebuild` | `bool` | Force plan rebuild |

## Topology Hooks

Register functions that return `TopologyAction` after each agent step:

```python
from gmas.execution import MACPRunner, RunnerConfig, TopologyAction, StepContext

def my_hook(ctx: StepContext, graph) -> TopologyAction | None:
    """Skip the reviewer if the writer is confident."""
    if ctx.agent_id == "writer" and "CONFIDENT" in (ctx.response or ""):
        return TopologyAction(skip_agents=["reviewer"])
    return None

config = RunnerConfig(
    adaptive=True,
    enable_dynamic_topology=True,
    topology_hooks=[my_hook],
)

runner = MACPRunner(llm_caller=llm_caller, config=config)
result = runner.run_round(graph)
```

## Async Topology Hooks

For hooks that need async operations:

```python
async def async_hook(ctx: StepContext, graph) -> TopologyAction | None:
    # Async logic here
    return TopologyAction(skip_agents=["some_agent"])

config = RunnerConfig(
    adaptive=True,
    enable_dynamic_topology=True,
    async_topology_hooks=[async_hook],
)
```

## Early Stopping

Stop execution based on conditions:

```python
from gmas.execution import EarlyStopCondition

# Stop when an agent outputs a specific keyword
config = RunnerConfig(
    adaptive=True,
    early_stop_conditions=[
        EarlyStopCondition.on_keyword("FINAL_ANSWER"),
        EarlyStopCondition.on_token_limit(max_tokens=5000),
        EarlyStopCondition.on_agent_count(max_agents=5),
    ],
)
```

### Combining Conditions

```python
# Stop when ANY condition is met
condition = EarlyStopCondition.combine_any([
    EarlyStopCondition.on_keyword("DONE"),
    EarlyStopCondition.on_token_limit(10000),
])

# Stop when ALL conditions are met
condition = EarlyStopCondition.combine_all([
    EarlyStopCondition.on_keyword("REVIEWED"),
    EarlyStopCondition.on_agent_count(min_agents_executed=3),
])
```

### Custom Conditions

```python
def check_quality(ctx: StepContext) -> bool:
    return ctx.response is not None and len(ctx.response) > 100

condition = EarlyStopCondition.on_custom(
    condition=check_quality,
    reason="Sufficient output quality",
    after_agents=["researcher"],  # only check after these agents
)
```

## Conditional Edges

Define edges that are only traversed when a condition is met:

```python
from gmas.core import AgentProfile
from gmas.builder import build_property_graph

graph = build_property_graph(agents, workflow_edges=[
    ("router", "analyst"),
    ("router", "creative"),
], query="...")

# Only traverse to creative if the router decides
graph.set_edge_condition("router", "creative", condition="has_keyword:creative")
graph.set_edge_condition("router", "analyst", condition="has_keyword:analytical")
```

## Graph Modification Methods

Direct graph modifications at runtime:

```python
# Add a new agent mid-execution
new_agent = AgentProfile(agent_id="fact_checker", display_name="Fact Checker")
graph.add_node(new_agent, connections_to=["writer"])

# Add/remove edges
graph.add_edge("fact_checker", "writer", weight=0.9)
graph.remove_edge("researcher", "writer")

# Disable nodes (present but not executed)
graph.disable("draft_agent")

# Re-enable
graph.enable("draft_agent")

# Set execution boundaries
graph.set_start_node("researcher")
graph.set_end_node("writer")
```

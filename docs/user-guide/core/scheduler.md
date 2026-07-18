# Scheduler

Determines execution order for agents in the graph.

## Execution Order

The scheduler computes a topological ordering of agents so that each agent only runs after its predecessors:

```python
from gmas.execution import build_execution_order

order = build_execution_order(graph.A_com, graph.agent_ids)
print(order)  # ['researcher', 'analyst', 'writer']
```

Agents with no incoming edges run first. If the graph contains cycles, the scheduler uses SCC (Strongly Connected Components) decomposition to find a valid ordering.

## Parallel Groups

Find agents that can run concurrently (same topological level, no dependencies between them):

```python
from gmas.execution import get_parallel_groups

groups = get_parallel_groups(graph)
for i, group in enumerate(groups):
    print(f"Level {i}: {group}")
# Level 0: ['researcher']
# Level 1: ['analyst', 'fact_checker']
# Level 2: ['writer']
```

## Adaptive Scheduler

For complex graphs with conditional edges, pruning, and dynamic topology:

```python
from gmas.execution import AdaptiveScheduler, PruningConfig

pruning_config = PruningConfig(
    min_reliability=0.3,
    max_latency_ms=5000,
)

scheduler = AdaptiveScheduler(pruning_config=pruning_config)
plan = scheduler.create_plan(graph)

print(f"Steps: {len(plan.steps)}")
for step in plan.steps:
    print(f"  {step.step_id}: {step.agent_ids} (deps: {step.dependency_ids})")
```

### ExecutionPlan

The plan contains `ExecutionStep` objects:

```python
plan = scheduler.create_plan(graph)

# Inspect steps
for step in plan.steps:
    step.step_id           # unique identifier
    step.agent_ids         # list of agent IDs in this step
    step.dependency_ids    # step IDs that must complete first

# Query plan state
plan.is_step_resolved(step_id)   # has this step completed?
plan.find_pending_step()          # next unresolved step
plan.apply_condition_skip(agent_id)  # skip an agent conditionally
```

### Pruning

The scheduler can remove unreliable or slow agents before execution:

```python
from gmas.execution import PruningConfig

pruning = PruningConfig(
    min_reliability=0.5,    # skip agents below this reliability score
    max_latency_ms=3000,    # skip agents slower than this
    max_pruned_fraction=0.3,  # don't prune more than 30% of agents
)

scheduler = AdaptiveScheduler(pruning_config=pruning)
plan = scheduler.create_plan(graph)
```

## Topological Sort with SCC

When cycles exist, the scheduler:

1. Finds all strongly connected components (SCCs)
2. Collapses each SCC into a meta-node
3. Topologically sorts the DAG of meta-nodes
4. Agents within an SCC are executed in the order they were added

```python
# This graph has a cycle: A -> B -> C -> A
# The scheduler will detect and handle it
order = build_execution_order(graph.A_com, graph.agent_ids)
```

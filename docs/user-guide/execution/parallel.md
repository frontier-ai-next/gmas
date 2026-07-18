# Parallel Execution

Run independent agents concurrently for faster execution.

## How It Works

When the graph has agents at the same topological level (no dependencies between them), the runner can execute them in parallel. This is especially useful for fan-out patterns:

```text
         researcher
         /        \
    analyst      fact_checker
         \        /
          writer
```

In this graph, `analyst` and `fact_checker` can run in parallel after `researcher` finishes.

## Enabling Parallel Execution

Parallel execution is enabled by default:

```python
from gmas.execution import MACPRunner, RunnerConfig

config = RunnerConfig(
    enable_parallel=True,
    max_parallel_size=5,  # max agents in a single parallel group
)

runner = MACPRunner(async_llm_caller=async_llm_caller, config=config)
result = await runner.arun_round(graph)
```

!!! note
    Parallel execution requires an **async** LLM caller (`async_llm_caller`). Sync callers run groups sequentially.

## Finding Parallel Groups

You can inspect parallel groups before execution:

```python
from gmas.execution import get_parallel_groups

groups = get_parallel_groups(graph)
for i, group in enumerate(groups):
    print(f"Group {i}: {group}")
# Group 0: ['researcher']
# Group 1: ['analyst', 'fact_checker']
# Group 2: ['writer']
```

## Parallel Events

When streaming, parallel execution emits specific events:

```python
for event in runner.astream(graph):
    match event.event_type:
        case StreamEventType.PARALLEL_START:
            print(f"Starting parallel group {event.group_index}: {event.agent_ids}")
        case StreamEventType.PARALLEL_END:
            print(f"Group done. Success: {event.successful}, Failed: {event.failed}")
```

## Execution Order with Parallelism

Even with parallel execution enabled, `result.execution_order` still reflects the logical execution order. Agents within a parallel group are listed in the order they were added to the graph, not by completion time.

## Controlling Parallelism

Limit the maximum number of concurrent agents:

```python
config = RunnerConfig(
    enable_parallel=True,
    max_parallel_size=3,  # at most 3 agents in parallel
)
```

Disable parallel execution entirely:

```python
config = RunnerConfig(enable_parallel=False)
```

## Error Handling in Parallel Groups

When one agent in a parallel group fails:

- The error is recorded in `result.errors`
- Other agents in the group continue executing
- Failed agents appear in `ParallelEndEvent.failed`
- The `ErrorPolicy` determines whether to stop or continue

```python
from gmas.execution import ErrorPolicy

config = RunnerConfig(
    enable_parallel=True,
    error_policy=ErrorPolicy.CONTINUE,  # continue despite failures
)
```

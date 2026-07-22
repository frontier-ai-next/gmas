# Callbacks

Track and respond to execution lifecycle events.

## Base Callback Handler

Create a custom handler by subclassing `BaseCallbackHandler`:

```python
from gmas.callbacks import BaseCallbackHandler

class MyHandler(BaseCallbackHandler):
    def on_run_start(self, graph, config, **kwargs):
        print("Execution started")

    def on_agent_end(self, agent_id, output, *, agent_name="", **kwargs):
        print(f"Agent {agent_name} ({agent_id}): {output[:100]}")

    def on_agent_error(self, error, *, agent_id, **kwargs):
        print(f"Agent {agent_id} failed: {error}")

    def on_tool_start(self, tool_name, inputs, **kwargs):
        print(f"Tool {tool_name} called with {inputs}")

    def on_tool_end(self, tool_name, output, **kwargs):
        print(f"Tool {tool_name} returned: {output[:100]}")
```

## Lifecycle hooks

`BaseCallbackHandler` provides no-op methods for all events. Override only what you need:

### Run Lifecycle
| Method | When Called |
|--------|------------|
| `on_run_start(graph, config)` | Execution begins |
| `on_run_end(output, **kwargs)` | Execution finishes |

### Agent Lifecycle
| Method | When Called |
|--------|------------|
| `on_agent_start(agent_id, agent_name, step_index, prompt)` | Agent begins processing |
| `on_agent_end(agent_id, output, agent_name, ...)` | Agent produces output |
| `on_agent_error(error, agent_id, ...)` | Agent fails |
| `on_retry(agent_id, attempt, max_attempts, ...)` | Agent is being retried |

### LLM Events
| Method | When Called |
|--------|------------|
| `on_llm_new_token(token, agent_id, ...)` | Token streamed (streaming mode) |

### Tool Events
| Method | When Called |
|--------|------------|
| `on_tool_start(tool_name, inputs, ...)` | Tool execution begins |
| `on_tool_end(tool_name, output, ...)` | Tool execution succeeds |
| `on_tool_error(tool_name, error, ...)` | Tool execution fails |

### Topology & Planning
| Method | When Called |
|--------|------------|
| `on_plan_created(plan, **kwargs)` | Execution plan created |
| `on_topology_changed(reason, **kwargs)` | Graph structure modified |
| `on_prune(agent_id, reason, **kwargs)` | Agent pruned from plan |
| `on_fallback(failed_agent_id, fallback_agent_id, ...)` | Fallback agent activated |

### Parallel Events
| Method | When Called |
|--------|------------|
| `on_parallel_start(agent_ids, group_index, ...)` | Parallel group starts |
| `on_parallel_end(agent_ids, group_index, ...)` | Parallel group ends |

### Memory Events
| Method | When Called |
|--------|------------|
| `on_memory_read(agent_id, entries_count, ...)` | Memory entries read |
| `on_memory_write(agent_id, key, value_size, ...)` | Memory entry written |

### Budget Events
| Method | When Called |
|--------|------------|
| `on_budget_warning(budget_type, current, limit, ...)` | Budget threshold crossed |
| `on_budget_exceeded(budget_type, current, limit, ...)` | Budget limit exceeded |

## Async Callback Handler

For async operations in callbacks, use `AsyncCallbackHandler`:

```python
from gmas.callbacks import AsyncCallbackHandler

class AsyncHandler(AsyncCallbackHandler):
    async def on_agent_end(self, agent_id, output, **kwargs):
        await save_to_database(agent_id, output)

    async def on_run_end(self, output, **kwargs):
        await notify_webhook(output)
```

## Using Callbacks

Attach handlers to the runner:

```python
from gmas.execution import MACPRunner

runner = MACPRunner(
    llm_caller=llm_caller,
    callbacks=[MyHandler(), AsyncHandler()],
)
result = runner.run_round(graph)
```

Or per-run:

```python
result = runner.run_round(graph, callbacks=[MyHandler()])
```

## CallbackManager

The `CallbackManager` dispatches events to all registered handlers:

```python
from gmas.callbacks import CallbackManager

manager = CallbackManager(handlers=[MyHandler()])
run_id = manager.on_run_start(graph, config)
manager.on_agent_start(agent_id="researcher", agent_name="Researcher", step_index=0, prompt="...")
manager.on_run_end(output="final answer")
```

### AsyncCallbackManager

```python
from gmas.callbacks import AsyncCallbackManager

manager = AsyncCallbackManager(handlers=[AsyncHandler()])
await manager.on_agent_end(agent_id="researcher", output="result")
```

## Built-in Handlers

### StdoutCallbackHandler

Print formatted events to the console:

```python
from gmas.callbacks import StdoutCallbackHandler

handler = StdoutCallbackHandler()
```

### MetricsCallbackHandler

Collect execution metrics:

```python
from gmas.callbacks import MetricsCallbackHandler

handler = MetricsCallbackHandler()
# After execution, access collected metrics
```

### FileCallbackHandler

Write events to a JSON lines file:

```python
from gmas.callbacks import FileCallbackHandler

handler = FileCallbackHandler(filepath="logs/run.jsonl")
```

## Typed Events

Events are also emitted as typed Pydantic models:

```python
from gmas.callbacks.events import (
    AgentStartEvent,
    AgentEndEvent,
    AgentErrorEvent,
    TokenEvent,
    ToolStartEvent,
    ToolEndEvent,
    TopologyChangedEvent,
    PruneEvent,
    FallbackEvent,
    ParallelStartEvent,
    ParallelEndEvent,
    MemoryReadEvent,
    MemoryWriteEvent,
    BudgetWarningEvent,
    BudgetExceededEvent,
)
```

Each event has `event_type`, `run_id`, `timestamp`, `tags`, and `metadata` fields plus event-specific data.

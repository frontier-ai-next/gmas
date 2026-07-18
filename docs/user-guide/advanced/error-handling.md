# Error Handling

Configure how the runner handles errors during execution.

## Error Policy

The `ErrorPolicy` enum controls error behavior:

| Policy | Behavior |
|--------|----------|
| `RAISE` | Raise the exception immediately (default) |
| `CONTINUE` | Skip the failed agent, continue execution |
| `RETRY` | Retry the agent before deciding |

```python
from gmas.execution import MACPRunner, RunnerConfig, ErrorPolicy

config = RunnerConfig(error_policy=ErrorPolicy.CONTINUE)
runner = MACPRunner(llm_caller=llm_caller, config=config)
```

## Retry Configuration

When using `ErrorPolicy.RETRY`:

```python
config = RunnerConfig(
    error_policy=ErrorPolicy.RETRY,
    max_retries=3,
    retry_delay=1.0,     # initial delay in seconds
    retry_backoff=2.0,   # multiplier for exponential backoff
)
```

Retry delays follow exponential backoff: 1s, 2s, 4s, ...

## Execution Errors

Errors are captured in `ExecutionError` objects stored in `result.errors`:

```python
result = runner.run_round(graph)

if result.errors:
    for err in result.errors:
        print(f"Agent {err.agent_id}: {err.error_type} - {err.message}")
```

## Error Events

In streaming mode, errors produce `AGENT_ERROR` events:

```python
from gmas.execution import StreamEventType

for event in runner.stream(graph):
    if event.event_type == StreamEventType.AGENT_ERROR:
        print(f"Error in {event.agent_id}: {event.error_message}")
        if event.will_retry:
            print(f"  Retrying (attempt {event.attempt}/{event.max_attempts})")
```

## Callback Error Handling

Callbacks receive error events:

```python
from gmas.callbacks import BaseCallbackHandler

class ErrorHandler(BaseCallbackHandler):
    def on_agent_error(self, error, *, agent_id, **kwargs):
        print(f"Agent {agent_id} failed: {error}")

    def on_tool_error(self, error, *, tool_name, **kwargs):
        print(f"Tool {tool_name} failed: {error}")
```

## Budget Errors

When budget limits are exceeded:

```python
from gmas.execution import BudgetConfig

budget = BudgetConfig(
    total_token_limit=10000,
    time_limit_seconds=300,
)

runner = MACPRunner(llm_caller=llm_caller, budget_config=budget)
```

Budget events fire during execution:

```python
for event in runner.stream(graph):
    if event.event_type == StreamEventType.BUDGET_WARNING:
        print(f"Budget warning: {event.budget_type} at {event.current}/{event.limit}")
    if event.event_type == StreamEventType.BUDGET_EXCEEDED:
        print(f"Budget exceeded: {event.budget_type}")
```

## Fallback Agents

When an agent fails, a fallback agent can take over:

Define fallback edges in the graph and the adaptive runner will automatically use them when a primary agent fails. Check `result.fallback_count` after execution.

## Timeout

Set a per-agent execution timeout:

```python
config = RunnerConfig(timeout=30)  # seconds per agent

runner = MACPRunner(llm_caller=llm_caller, config=config)
```

If an agent exceeds the timeout, it's treated as an error and handled according to the error policy.

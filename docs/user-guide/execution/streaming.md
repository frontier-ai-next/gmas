# Streaming

Get real-time output as agents execute using the streaming API.

## Basic Streaming

```python
from gmas.execution import MACPRunner, StreamEventType

runner = MACPRunner(llm_caller=llm_caller)

for event in runner.stream(graph):
    match event.event_type:
        case StreamEventType.RUN_START:
            print(f"Started: {event.execution_order}")
        case StreamEventType.AGENT_START:
            print(f"[{event.agent_name}] Thinking...")
        case StreamEventType.AGENT_OUTPUT:
            print(f"[{event.agent_name}] {event.content}")
        case StreamEventType.AGENT_ERROR:
            print(f"Error: {event.error_message} (retry={event.will_retry})")
        case StreamEventType.RUN_END:
            print(f"Done in {event.total_time:.2f}s")
```

## Stream Event Types

| Event | Description | Key Fields |
|-------|-------------|------------|
| `RUN_START` | Execution begins | `execution_order`, `num_agents`, `query` |
| `AGENT_START` | Agent begins processing | `agent_id`, `agent_name`, `step_index`, `predecessors` |
| `AGENT_OUTPUT` | Agent produced output | `agent_id`, `content`, `tokens_used`, `duration_ms`, `is_final` |
| `AGENT_ERROR` | Agent execution error | `agent_id`, `error_message`, `will_retry`, `attempt` |
| `TOKEN` | Single token streamed | `agent_id`, `token`, `token_index`, `is_first`, `is_last` |
| `TOPOLOGY_CHANGED` | Graph structure modified | `reason`, `old_remaining`, `new_remaining` |
| `PRUNE` | Agent was pruned | `agent_id`, `reason` |
| `FALLBACK` | Fallback agent activated | `failed_agent_id`, `fallback_agent_id` |
| `PARALLEL_START` | Parallel group begins | `agent_ids`, `group_index` |
| `PARALLEL_END` | Parallel group ends | `agent_ids`, `successful`, `failed` |
| `MEMORY_WRITE` | Memory entry written | `agent_id`, `key`, `value_size` |
| `MEMORY_READ` | Memory entries read | `agent_id`, `entries_count` |
| `BUDGET_WARNING` | Budget threshold crossed | `budget_type`, `current`, `limit` |
| `BUDGET_EXCEEDED` | Budget limit exceeded | `budget_type`, `current`, `limit` |
| `RUN_END` | Execution complete | `success`, `final_answer`, `total_tokens`, `total_time` |

## StreamBuffer

Collect streaming events and access the final result:

```python
from gmas.execution import StreamBuffer

buffer = StreamBuffer()

for event in runner.stream(graph):
    buffer.add(event)
    # Real-time processing...

# After streaming completes:
print(buffer.final_answer)
print(buffer.agent_outputs)  # dict[agent_id, output]
```

## Helper Functions

```python
from gmas.execution import stream_to_string, print_stream

# Get the final answer as a string
answer = stream_to_string(runner.stream(graph))

# Print events with formatted output to stdout
print_stream(runner.stream(graph))
```

## Async Streaming

```python
async for event in runner.astream(graph):
    match event.event_type:
        case StreamEventType.AGENT_OUTPUT:
            print(f"{event.agent_name}: {event.content}")
        case StreamEventType.RUN_END:
            print(f"Completed: {event.success}")
```

## Token-Level Streaming

Enable token-by-token streaming (requires a streaming LLM caller):

```python
from gmas.execution import RunnerConfig

config = RunnerConfig(enable_token_streaming=True)

runner = MACPRunner(
    streaming_llm_caller=my_streaming_caller,  # Callable[[str], Iterator[str]]
    config=config,
)

for event in runner.stream(graph):
    if event.event_type == StreamEventType.TOKEN:
        print(event.token, end="", flush=True)
```

## Stream + Result Together

Get both streaming events and the final `MACPResult`:

```python
events, result = runner.stream_to_result(graph)

for event in events:
    print(event)

print(result.final_answer)
```

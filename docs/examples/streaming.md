# Streaming Example

Execute agents with real-time output.

## Basic Streaming

```python
from gmas.core import AgentProfile
from gmas.builder import build_property_graph
from gmas.execution import MACPRunner, StreamEventType

agents = [
    AgentProfile(agent_id="researcher", display_name="Researcher"),
    AgentProfile(agent_id="writer", display_name="Writer"),
]
graph = build_property_graph(
    agents,
    workflow_edges=[("researcher", "writer")],
    query="Explain quantum entanglement in simple terms",
)

def llm_caller(prompt: str) -> str:
    # Your LLM API call here
    return "response text"

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
            print(f"Error: {event.error_message}")
        case StreamEventType.RUN_END:
            print(f"Done in {event.total_time:.2f}s")
```

## Stream Buffer

Collect events and access the final result:

```python
from gmas.execution import StreamBuffer

buffer = StreamBuffer()

for event in runner.stream(graph):
    buffer.add(event)
    # Real-time processing...

print(f"Final answer: {buffer.final_answer}")
print(f"Agent outputs: {buffer.agent_outputs}")
```

## Helper Functions

```python
from gmas.execution import print_stream, stream_to_string

# Print events with formatting
print_stream(runner.stream(graph))

# Get final answer as string
answer = stream_to_string(runner.stream(graph))
```

## Handling All Event Types

```python
for event in runner.stream(graph):
    match event.event_type:
        case StreamEventType.RUN_START:
            print(f"Agents: {event.execution_order}")
        case StreamEventType.AGENT_START:
            print(f"  -> {event.agent_name} (step {event.step_index})")
        case StreamEventType.AGENT_OUTPUT:
            print(f"  <- {event.agent_name}: {event.content[:100]}")
        case StreamEventType.AGENT_ERROR:
            print(f"  !! {event.agent_id}: {event.error_message}")
            if event.will_retry:
                print(f"     Retrying ({event.attempt}/{event.max_attempts})")
        case StreamEventType.PRUNE:
            print(f"  Pruned: {event.agent_id} ({event.reason})")
        case StreamEventType.FALLBACK:
            print(f"  Fallback: {event.failed_agent_id} -> {event.fallback_agent_id}")
        case StreamEventType.TOPOLOGY_CHANGED:
            print(f"  Topology changed: {event.reason}")
        case StreamEventType.BUDGET_WARNING:
            print(f"  Budget: {event.budget_type} at {event.current}/{event.limit}")
        case StreamEventType.RUN_END:
            success = "OK" if event.success else "FAILED"
            print(f"  [{success}] {event.total_time:.2f}s, {event.total_tokens} tokens")
```

## Async Streaming

```python
import asyncio

async def run():
    async for event in runner.astream(graph):
        match event.event_type:
            case StreamEventType.AGENT_OUTPUT:
                print(f"{event.agent_name}: {event.content}")
            case StreamEventType.RUN_END:
                print(f"Completed in {event.total_time:.2f}s")

asyncio.run(run())
```

## Async with StreamBuffer

```python
async def run_with_buffer():
    from gmas.execution import StreamBuffer

    buffer = StreamBuffer()
    async for event in runner.astream(graph):
        buffer.add(event)

    print(f"Final: {buffer.final_answer}")

asyncio.run(run_with_buffer())
```

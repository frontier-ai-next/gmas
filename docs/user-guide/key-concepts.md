# Key Concepts

## RoleGraph

The core data structure — a directed graph based on rustworkx with adjacency matrices, PyG conversion, and dynamic topology support.

```python
from gmas.core import RoleGraph, AgentProfile
from gmas.builder import build_property_graph

agents = [AgentProfile(agent_id="agent1", display_name="Agent 1")]
graph = build_property_graph(agents, query="Task")

# Access graph properties
print(graph.num_nodes)  # Number of nodes
print(graph.num_edges)  # Number of edges
print(graph.A_com)      # Adjacency matrix as torch.Tensor
```

## AgentProfile

Represents an individual agent with local state, tools, and optional LLM configuration.

```python
from gmas.core import AgentProfile

agent = AgentProfile(
    agent_id="unique_id",
    display_name="Human Readable Name",
    description="What this agent does",
    persona="You are a helpful assistant",
    tools=["tool1", "tool2"],
)
```

Key attributes:
- `agent_id` - Unique identifier
- `display_name` - Human-readable name
- `description` - Functional description
- `persona` - System prompt personality
- `state` - Local agent state (decentralized memory)
- `embedding` - Encoded representation
- `tools` - Available tools
- `llm_config` - Per-agent LLM configuration

## TaskNode

Represents the task/query for the agent system. Automatically added by `build_property_graph`.

```python
from gmas.core import TaskNode

task = TaskNode(query="What is the capital of France?")
```

## Execution Flow

1. **Build Graph** - Create agents and define connections via `build_property_graph`
2. **Schedule** - Determine execution order (topological sort)
3. **Execute** - Run agents in order via `MACPRunner`, passing messages between them
4. **Result** - Get `MACPResult` with final answer, metrics, and agent states

## Graph Topology

Edges define information flow between agents:

```python
from gmas.builder import build_property_graph

# Linear chain: A -> B -> C
edges = [("agent_a", "agent_b"), ("agent_b", "agent_c")]

# Parallel: A -> B, A -> C
edges = [("agent_a", "agent_b"), ("agent_a", "agent_c")]

# Star: All connect to center
edges = [("agent_1", "center"), ("agent_2", "center")]

# Complex: multi-level
edges = [
    ("researcher", "analyst"),
    ("researcher", "fact_checker"),
    ("analyst", "writer"),
    ("fact_checker", "writer"),
]
```

## Dynamic Topology

The graph can be changed while a run is in progress:

```python
# Add new agent
new_agent = AgentProfile(agent_id="new", display_name="New Agent")
graph.add_node(new_agent, connections_to=["existing_agent"])

# Add edge
graph.add_edge("agent_a", "agent_b", weight=0.8)

# Remove edge
graph.remove_edge("agent_a", "agent_b")

# Disable nodes (present but not executed)
graph.disable("draft_agent")
```

See [Dynamic Topology](advanced/dynamic-topology.md) for topology hooks and conditional edges.

## Memory

Each agent has its own decentralized state:

```python
# Access agent state
agent = graph.get_agent_by_id("agent_id")
current_state = agent.state

# Update state (returns new instance)
agent = agent.append_state({"role": "user", "content": "query"})
```

See [Memory System](memory.md) for the full memory system including shared pools.

## Tools

Agents can use tools during execution:

```python
from gmas.core import AgentProfile

agent = AgentProfile(
    agent_id="researcher",
    display_name="Researcher",
    tools=["web_search", "code_interpreter"],
)
```

See [Tools](tools.md) for all available tools and custom tool creation.

## Callbacks

Track execution lifecycle events:

```python
from gmas.callbacks import BaseCallbackHandler

class MyHandler(BaseCallbackHandler):
    def on_agent_end(self, agent_id, output, **kwargs):
        print(f"{agent_id}: {output}")
```

See [Callbacks](callbacks.md) for the full callback interface.

## Streaming

Get real-time output during execution:

```python
from gmas.execution import StreamEventType

for event in runner.stream(graph):
    if event.event_type == StreamEventType.AGENT_OUTPUT:
        print(event.content)
```

See [Streaming](execution/streaming.md) for all event types and usage patterns.

## Next Steps

- [RoleGraph](core/rolegraph.md) - Deep dive into graph operations
- [AgentProfile](core/agent-profile.md) - Agent configuration
- [MACPRunner](core/macp-runner.md) - Execution engine
- [Running Agents](execution/running.md) - Execution details
- [Tools](tools.md) - Tool system
- [Callbacks](callbacks.md) - Event handling

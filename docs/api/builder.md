# Builder API

## Builder Module

```python
from gmas.builder import (
    # Main builder
    GraphBuilder,
    build_property_graph,
    build_from_schema,
    build_from_adjacency,

    # Configuration
    BuilderConfig,
    AutoBuilderConfig,
    EmbeddingBuilderConfig,

    # Auto builder
    AutoGraphBuilder,
    EmbeddingGraphBuilder,

    # Utilities
    default_edges,
    default_sequence,
)
```

## GraphBuilder

Fluent builder for constructing agent graphs step by step.

### Methods

```python
builder = GraphBuilder()

# Add agents
builder.add_agent(agent: AgentProfile) -> Self

# Add edges between agents
builder.add_edge(source: str, target: str, **kwargs) -> Self

# Add a sequence of agents (chain: A -> B -> C -> ...)
builder.add_sequence(agent_ids: list[str]) -> Self

# Build the final graph
builder.build(query: str) -> RoleGraph
```

### Example

```python
from gmas.builder import GraphBuilder
from gmas.core import AgentProfile

builder = GraphBuilder()
builder.add_agent(AgentProfile(agent_id="researcher", display_name="Researcher"))
builder.add_agent(AgentProfile(agent_id="writer", display_name="Writer"))
builder.add_edge("researcher", "writer")
graph = builder.build(query="Explain quantum computing")
```

## build_property_graph

Quick graph building function — the most common way to create a graph.

```python
graph = build_property_graph(
    agents: list[AgentProfile],
    workflow_edges: list[tuple[str, str]] | None = None,
    query: str = "",
    include_task_node: bool = True,
) -> RoleGraph
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `list[AgentProfile]` | required | List of agents |
| `workflow_edges` | `list[tuple[str, str]] \| None` | `None` | Edge connections |
| `query` | `str` | `""` | Task query |
| `include_task_node` | `bool` | `True` | Add a virtual task node |

```python
from gmas.builder import build_property_graph

graph = build_property_graph(
    agents=[agent1, agent2, agent3],
    workflow_edges=[("agent1", "agent2"), ("agent2", "agent3")],
    query="What is machine learning?",
    include_task_node=True,
)
```

## build_from_schema

Build a graph from a `GraphSchema` definition.

```python
from gmas.builder import build_from_schema
from gmas.core import GraphSchema, AgentNodeSchema

schema = GraphSchema(
    name="my_graph",
    nodes={
        "researcher": AgentNodeSchema(id="researcher", display_name="Researcher"),
        "writer": AgentNodeSchema(id="writer", display_name="Writer"),
    },
    edges=[
        WorkflowEdgeSchema(source="researcher", target="writer"),
    ],
)

graph = build_from_schema(schema, query="...")
```

## build_from_adjacency

Build a graph from an adjacency matrix.

```python
from gmas.builder import build_from_adjacency
import torch

agents = [agent1, agent2, agent3]
A_com = torch.tensor([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0],
])

graph = build_from_adjacency(agents, A_com, query="...")
```

## AutoGraphBuilder

Automatically build a graph from a task description.

```python
from gmas.builder import AutoGraphBuilder, AutoBuilderConfig

config = AutoBuilderConfig(
    max_agents=5,
    link_threshold=0.7,
)

auto_builder = AutoGraphBuilder(config)
graph = auto_builder.build(query="Your task here")
```

### AutoBuilderConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_agents` | `int` | `5` | Maximum number of agents |
| `link_threshold` | `float` | `0.7` | Threshold for auto-linking |

## EmbeddingGraphBuilder

Build graph using agent embeddings to determine connections.

```python
from gmas.builder import EmbeddingGraphBuilder, EmbeddingBuilderConfig

config = EmbeddingBuilderConfig(
    link_strategy=LinkStrategy.THRESHOLD,
    threshold=0.8,
)

builder = EmbeddingGraphBuilder(config)
graph = builder.build(agents, query="...")
```

### EmbeddingBuilderConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `link_strategy` | `LinkStrategy` | `THRESHOLD` | How to determine edges |
| `threshold` | `float` | `0.8` | Similarity threshold |

### LinkStrategy

| Member | Description |
|--------|-------------|
| `THRESHOLD` | Link agents above similarity threshold |
| `TOP_K` | Link each agent to K most similar |
| `FULLY_CONNECTED` | Connect all agents |

## Utility Functions

### default_edges

Create default fully-connected edges for a list of agents:

```python
edges = default_edges(agent_ids)
# [("agent_0", "agent_1"), ("agent_1", "agent_2"), ...]
```

### default_sequence

Create a sequential chain of agent IDs:

```python
edges = default_sequence(agent_ids)
# [("agent_0", "agent_1"), ("agent_1", "agent_2"), ...]
```

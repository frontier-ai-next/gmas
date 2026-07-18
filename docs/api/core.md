# Core API Reference

## Core Module

```python
from gmas.core import (
    # Graph
    RoleGraph,
    GraphIntegrityError,
    StateStorage,

    # Agents
    AgentProfile,
    AgentLLMConfig,
    TaskNode,

    # Builder
    build_property_graph,
    GraphBuilder,
    BuilderConfig,

    # Schema
    GraphSchema,
    AgentNodeSchema,
    TaskNodeSchema,
    EdgeType,
    NodeType,

    # Encoder
    NodeEncoder,

    # Algorithms
    GraphAlgorithms,
    CentralityType,

    # Metrics
    MetricsTracker,
    NodeMetrics,
    EdgeMetrics,

    # Events
    EventBus,
    EventHandler,
    EventType,

    # Visualization
    GraphVisualizer,
    to_mermaid,
    to_dot,
    to_ascii,
    print_graph,
)
```

## RoleGraph

Main graph class for multi-agent systems, built on `rustworkx.PyDiGraph`.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `agents` | `list[Any]` | `[]` | Agent objects |
| `node_ids` | `list[str]` | `[]` | Node identifiers |
| `role_connections` | `dict[str, list[str]]` | `{}` | Agent connections |
| `task_node` | `str \| None` | `None` | Task node ID |
| `query` | `str \| None` | `None` | Task query |
| `answer` | `str \| None` | `None` | Stored answer |
| `graph` | `rx.PyDiGraph` | `PyDiGraph()` | Underlying rustworkx graph |
| `A_com` | `torch.Tensor` | `zeros(0,0)` | Adjacency matrix |
| `S_tilde` | `torch.Tensor \| None` | `None` | Score matrix |
| `p_matrix` | `torch.Tensor \| None` | `None` | Probability matrix |
| `start_node` | `str \| None` | `None` | Start node ID |
| `end_node` | `str \| None` | `None` | End node ID |
| `disabled_nodes` | `set[str]` | `set()` | Disabled node IDs |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_nodes` | `int` | Total nodes |
| `num_edges` | `int` | Total edges |
| `agent_ids` | `list[str]` | Agent IDs |
| `A_com` | `torch.Tensor` | Adjacency matrix |
| `edge_index` | `torch.Tensor` | PyG format (2 x E) |
| `edge_attr` | `torch.Tensor` | Edge features (E x 4) |
| `embeddings` | `torch.Tensor` | Node embeddings |
| `role_sequence` | `list[str]` | Execution sequence |
| `has_conditional_edges` | `bool` | Has conditional routing |
| `conditional_edges` | `list[tuple[str, str]]` | Conditional edge list |

### Methods

```python
# Node operations
graph.add_node(agent, connections_from=None, connections_to=None) -> bool
graph.remove_node(node_id, policy=StateMigrationPolicy.DISCARD) -> Any | None
graph.replace_node(node_id, new_agent, policy=StateMigrationPolicy.COPY) -> Any | None
graph.get_agent_by_id(agent_id) -> Any | None
graph.get_node_index(node_id) -> int | None

# Edge operations
graph.add_edge(source_id, target_id, weight=1.0, **edge_attrs) -> bool
graph.remove_edge(source_id, target_id) -> bool
graph.get_neighbors(node_id, direction="out") -> list[str]

# Communication
graph.update_communication(A_com, s_tilde=None, p_matrix=None) -> None
graph.update_task(new_query) -> None

# Conditional edges
graph.set_edge_condition(source, target, condition) -> bool
graph.get_edge_condition(source, target) -> Any | None
graph.remove_edge_condition(source, target) -> bool

# Execution bounds
graph.set_start_node(node_id) -> bool
graph.set_end_node(node_id) -> bool
graph.set_execution_bounds(start_node, end_node) -> bool

# Disable/enable nodes
graph.disable(node_ids) -> int
graph.enable(node_ids=None) -> int
graph.is_enabled(node_id) -> bool

# Reachability
graph.get_reachable_from(source_id, threshold=0.5) -> set[str]
graph.get_nodes_reaching(target_id, threshold=0.5) -> set[str]
graph.get_relevant_nodes(start_node=None, end_node=None) -> set[str]
graph.get_isolated_nodes(start_node=None, end_node=None) -> set[str]
graph.get_optimized_execution_order(...) -> list[str]

# Schema validation
graph.validate_agent_input(agent_id, data) -> Any
graph.validate_agent_output(agent_id, data) -> Any
graph.has_input_schema(agent_id) -> bool
graph.has_output_schema(agent_id) -> bool

# PyG conversion
graph.to_pyg_data(node_features=None, edge_features=None,
                  include_embeddings=True, include_default_edge_attr=True) -> Data

# Serialization
graph.to_dict() -> dict
graph.subgraph(node_ids) -> RoleGraph

# Integrity
graph.verify_integrity(raise_on_error=True) -> list[str]
graph.is_consistent() -> bool
```

## GraphIntegrityError

Raised when graph integrity checks fail (duplicate nodes, invalid edges, etc.).

## StateStorage

Runtime-checkable protocol for external state persistence:

```python
class StateStorage(Protocol):
    def save(self, node_id: str, state: dict) -> None: ...
    def load(self, node_id: str) -> dict | None: ...
    def delete(self, node_id: str) -> None: ...
```

## AgentProfile

Agent definition (frozen Pydantic model). All mutations return new instances.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `agent_id` | `str` | required | Unique identifier |
| `display_name` | `str` | required | Human-readable name |
| `persona` | `str` | `""` | Agent personality/role |
| `description` | `str` | `""` | Functional description |
| `llm_backbone` | `str \| None` | `None` | LLM backbone identifier |
| `llm_config` | `AgentLLMConfig \| None` | `None` | Per-agent LLM config |
| `tools` | `list[Any]` | `[]` | Tool names or objects |
| `embedding` | `torch.Tensor \| None` | `None` | Encoded representation |
| `state` | `list[dict[str, Any]]` | `[]` | Message history |
| `hidden_state` | `torch.Tensor \| None` | `None` | Hidden state vector |
| `input_schema` | `Any \| None` | `None` | Input validation schema |
| `output_schema` | `Any \| None` | `None` | Output validation schema |

### Methods

```python
agent.with_state(state) -> AgentProfile
agent.append_state(message) -> AgentProfile
agent.clear_state() -> AgentProfile
agent.with_embedding(tensor) -> AgentProfile
agent.with_hidden_state(tensor) -> AgentProfile
agent.with_llm_config(config) -> AgentProfile
agent.has_tools() -> bool
agent.get_tool_names() -> list[str]
agent.get_tool_objects() -> list[BaseTool]
agent.to_text() -> str
agent.to_dict() -> dict
agent.get_model_name() -> str | None
agent.has_custom_llm() -> bool
```

## AgentLLMConfig

LLM configuration for an individual agent.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str \| None` | `None` | Model name |
| `base_url` | `str \| None` | `None` | API base URL |
| `api_key` | `str \| None` | `None` | API key |
| `max_tokens` | `int \| None` | `None` | Max output tokens |
| `temperature` | `float \| None` | `None` | Sampling temperature |
| `timeout` | `float \| None` | `None` | Request timeout |
| `top_p` | `float \| None` | `None` | Top-p sampling |
| `stop_sequences` | `list[str] \| None` | `None` | Stop sequences |
| `extra_params` | `dict[str, Any]` | `{}` | Additional parameters |

```python
config = AgentLLMConfig(model_name="gpt-4", temperature=0.7)
config.resolve_api_key()  # str | None
config.is_configured()    # bool
config.to_generation_params()  # dict
```

## TaskNode

Virtual task node encoding the query.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `agent_id` | `str` | `"__task__"` | Node ID |
| `query` | `str` | required | Task query |
| `description` | `str` | auto | Node description |
| `type` | `str` | `"task"` | Node type |
| `embedding` | `torch.Tensor \| None` | `None` | Encoded representation |

## NodeType

| Member | Description |
|--------|-------------|
| `AGENT` | Agent node |
| `TASK` | Task node |
| `SUBGRAPH` | Subgraph node |
| `TOOL` | Tool node |
| `CUSTOM` | Custom node type |

## EdgeType

| Member | Description |
|--------|-------------|
| `WORKFLOW` | Workflow edge |
| `TASK_CONTEXT` | Task context edge |
| `TASK_UPDATE` | Task update edge |
| `DEPENDENCY` | Dependency edge |
| `FEEDBACK` | Feedback edge |
| `FALLBACK` | Fallback edge |
| `CUSTOM` | Custom edge type |

## GraphAlgorithms

Graph analysis algorithms.

```python
algo = GraphAlgorithms(graph)
algo.compute_centrality(CentralityType.PAGERANK) -> dict[str, float]
algo.detect_communities() -> list[list[str]]
algo.find_shortest_path(source, target) -> list[str]
algo.find_cycles() -> list[list[str]]
```

## MetricsTracker

Tracks execution metrics per node and edge.

```python
tracker = MetricsTracker(history_size=1000, ema_alpha=0.1)
tracker.record_node_execution(node_id, success=True, latency_ms=100, ...)
tracker.record_edge_transition(source_id, target_id, success=True, ...)
tracker.get_node_metrics(node_id) -> NodeMetrics
tracker.get_edge_metrics(source_id, target_id) -> EdgeMetrics
tracker.suggest_pruning(reliability_threshold=0.5) -> list[str]
tracker.get_routing_weights() -> torch.Tensor
```

## NodeEncoder

Encodes agent descriptions into embeddings using sentence-transformers.

```python
encoder = NodeEncoder(model_name="all-MiniLM-L6-v2")
embedding = encoder.encode(agent) -> torch.Tensor
embeddings = encoder.encode_batch(agents) -> torch.Tensor
```

## GraphVisualizer

Visualizes agent graphs.

```python
from gmas.core import to_mermaid, to_dot, to_ascii, print_graph

mermaid_str = to_mermaid(graph)
dot_str = to_dot(graph)
ascii_str = to_ascii(graph)
print_graph(graph)  # prints to stdout
```

## StateMigrationPolicy

| Member | Description |
|--------|-------------|
| `DISCARD` | Discard state on node removal |
| `COPY` | Copy state to replacement |
| `ARCHIVE` | Archive state externally |

## Utility Functions

```python
# Build graph from configuration
build_property_graph(agents, workflow_edges=None, query="",
                     include_task_node=True) -> RoleGraph

# Visualization
to_mermaid(graph) -> str
to_dot(graph) -> str
to_ascii(graph) -> str
print_graph(graph)

# Graph conversion
graph.to_pyg_data() -> Data  # PyTorch Geometric

# Extract agents from dict
extract_agent_profiles(agents_data) -> list[AgentProfile]
```

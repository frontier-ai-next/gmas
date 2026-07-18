# GNN Routing Example

Use graph neural networks for intelligent routing.

## Setup

```python
from gmas.core import AgentProfile
from gmas.builder import build_property_graph
from gmas.execution import MACPRunner, create_openai_caller
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
```

## Create Graph

```python
agents = [
    AgentProfile(agent_id=f"agent_{i}", display_name=f"Agent {i}")
    for i in range(5)
]

graph = build_property_graph(
    agents,
    workflow_edges=[
        ("agent_0", "agent_1"),
        ("agent_0", "agent_2"),
        ("agent_1", "agent_3"),
        ("agent_2", "agent_3"),
        ("agent_3", "agent_4"),
    ],
    query="Example query",
)
```

## Convert to PyG

```python
pyg_data = graph.to_pyg_data()
print(pyg_data)
# Data(x=[6, 768], edge_index=[2, 10])  # 5 agents + 1 task node
```

## Define GNN Router

```python
class GNNRouter(torch.nn.Module):
    def __init__(self, in_channels: int = 768, hidden_channels: int = 128):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def predict_adjacency(self, x, edge_index, num_nodes: int) -> torch.Tensor:
        """Compute pairwise agent similarity as new adjacency."""
        node_embs = self.forward(x, edge_index)
        # Remove task node (first row/col) for agent-only matrix
        agent_embs = node_embs[1:]  # skip task node
        scores = torch.mm(agent_embs, agent_embs.T)
        return torch.sigmoid(scores)

model = GNNRouter()
```

## Update Graph with GNN Predictions

```python
# Generate new adjacency from GNN
num_agents = len(agents)
new_adjacency = model.predict_adjacency(
    pyg_data.x,
    pyg_data.edge_index,
    num_nodes=num_agents,
)

# Optional: create score and probability matrices
scores = new_adjacency.clone()
probabilities = F.softmax(new_adjacency.flatten(), dim=0).reshape(new_adjacency.shape)

# Update the graph's routing
graph.update_communication(
    A_com=new_adjacency,
    s_tilde=scores,
    p_matrix=probabilities,
)

print(f"New adjacency matrix:\n{graph.A_com}")
```

## Execute with GNN-Routed Graph

```python
# Define an LLM caller
def llm_caller(prompt: str) -> str:
    # Your LLM API call here
    return "response"

runner = MACPRunner(llm_caller=llm_caller)
result = runner.run_round(graph)

print(f"Execution order: {result.execution_order}")
print(f"Final answer: {result.final_answer}")
```

## Complete Pipeline

Putting it all together:

```python
# 1. Create agents and graph
agents = [
    AgentProfile(agent_id=f"agent_{i}", display_name=f"Agent {i}")
    for i in range(5)
]
graph = build_property_graph(
    agents,
    workflow_edges=[
        ("agent_0", "agent_1"),
        ("agent_0", "agent_2"),
        ("agent_1", "agent_3"),
        ("agent_2", "agent_3"),
        ("agent_3", "agent_4"),
    ],
    query="Analyze the benefits of renewable energy",
)

# 2. Convert to PyG
pyg_data = graph.to_pyg_data()

# 3. Run GNN to get new routing
model = GNNRouter()
new_adjacency = model.predict_adjacency(
    pyg_data.x, pyg_data.edge_index, num_nodes=len(agents)
)

# 4. Update graph
graph.update_communication(A_com=new_adjacency)

# 5. Execute
runner = MACPRunner(llm_caller=llm_caller)
result = runner.run_round(graph)
print(result.final_answer)
```

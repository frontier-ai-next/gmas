# GNN Routing

Use Graph Neural Networks for intelligent agent routing.

## Overview

gMAS provides PyTorch Geometric integration that enables GNN-based routing. Instead of static edges, a GNN can learn which agents should communicate based on the task and intermediate results.

## Converting to PyG Format

```python
from gmas.core import AgentProfile
from gmas.builder import build_property_graph

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

pyg_data = graph.to_pyg_data()
print(pyg_data)
# Data(x=[6, 768], edge_index=[2, 10])  # 5 agents + 1 task node
```

### Custom Node/Edge Features

```python
# Add custom features to the PyG data
node_features = {
    "trust_score": torch.tensor([0.9, 0.8, 0.95, 0.7, 0.85, 0.0]),
}
edge_features = {
    "reliability": torch.tensor([0.9, 0.8, 0.85, 0.7, 0.95,
                                  0.9, 0.8, 0.85, 0.7, 0.95]),
}

pyg_data = graph.to_pyg_data(
    node_features=node_features,
    edge_features=edge_features,
    include_embeddings=True,
    include_default_edge_attr=True,
)
```

## Training a GNN Router

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GNNRouter(torch.nn.Module):
    def __init__(self, in_channels: int = 768, hidden_channels: int = 128):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def predict_adjacency(self, x, edge_index, num_nodes):
        node_embs = self.forward(x, edge_index)
        # Compute pairwise scores
        scores = torch.mm(node_embs, node_embs.T)
        return torch.sigmoid(scores)

model = GNNRouter()
```

## Updating Graph Communication

Use GNN output to update the graph's adjacency matrix:

```python
# Get predictions from the GNN
new_A = model.predict_adjacency(pyg_data.x, pyg_data.edge_index, num_nodes=5)

# Update the graph with new routing
graph.update_communication(
    A_com=new_A,        # New adjacency matrix
    s_tilde=None,       # Optional score matrix
    p_matrix=None,      # Optional probability matrix
)

# Now run execution with the GNN-routed graph
result = runner.run_round(graph)
```

## GNN Routing Module

gMAS provides built-in GNN models:

```python
from gmas.core.gnn import GCN, GAT, GraphSAGE

# GCN routing
model = GCN(in_channels=768, hidden_channels=128, out_channels=64)

# GAT routing (with attention)
model = GAT(in_channels=768, hidden_channels=128, out_channels=64, heads=4)

# GraphSAGE routing
model = GraphSAGE(in_channels=768, hidden_channels=128, out_channels=64)
```

## Node Encoder

Encode agent descriptions into embeddings for GNN input:

```python
from gmas.core import NodeEncoder

encoder = NodeEncoder(model_name="all-MiniLM-L6-v2")

# Encode single agent
embedding = encoder.encode(agent)

# Encode batch
embeddings = encoder.encode_batch(agents)

# Assign embeddings to agents
for agent, emb in zip(agents, embeddings):
    agent = agent.with_embedding(emb)
```

## Full GNN Routing Pipeline

```python
# 1. Create and encode agents
encoder = NodeEncoder()
agents = [agent.with_embedding(emb) for agent, emb in
          zip(agents, encoder.encode_batch(agents))]

# 2. Build graph
graph = build_property_graph(agents, workflow_edges=edges, query="...")

# 3. Convert to PyG
pyg_data = graph.to_pyg_data()

# 4. Run GNN
new_A = model.predict_adjacency(pyg_data.x, pyg_data.edge_index, len(agents))

# 5. Update routing
graph.update_communication(A_com=new_A)

# 6. Execute
result = runner.run_round(graph)
```

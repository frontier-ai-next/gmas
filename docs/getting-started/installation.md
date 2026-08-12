# Installation

## Requirements

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) (recommended)
- PyTorch 2.11 or higher
- rustworkx 0.17.1 or higher

## Basic Installation

### Using uv (Recommended)

```bash
uv pip install frontier-ai-gmas
```

### Using pip

```bash
python -m pip install frontier-ai-gmas
```

The package is published on PyPI as `frontier-ai-gmas` and imported in Python
as `gmas`:

```python
import gmas

print(gmas.__version__)
```

Core dependencies include:

- rustworkx (graph operations)
- pydantic (data validation)
- torch (tensor operations)
- loguru (logging)
- openai (LLM client)

## Optional Extras

### Web Search

```bash
uv pip install frontier-ai-gmas[web-search]   # DuckDuckGo
uv pip install frontier-ai-gmas[web-fast]     # Fast HTML parsing
uv pip install frontier-ai-gmas[selenium]     # Browser automation
```

### Machine Learning

```bash
uv pip install frontier-ai-gmas[embeddings]   # sentence-transformers
uv pip install frontier-ai-gmas[pyg]          # PyTorch Geometric
```

### Visualization

```bash
uv pip install frontier-ai-gmas[viz]          # graphviz, matplotlib
```

### All Extras

```bash
uv pip install frontier-ai-gmas[all]
```

## Development Installation

Clone and install with uv:

```bash
git clone https://github.com/frontier-ai-next/gmas.git
cd gmas
uv sync
```

Install pre-commit hooks:

```bash
uv run prek install
```

## Verify Installation

```python
import gmas

from gmas.core import AgentProfile
from gmas.builder import build_property_graph

agents = [AgentProfile(agent_id="test", display_name="Test")]
graph = build_property_graph(agents, query="Test")
print(f"gMAS {gmas.__version__}")
print(f"Success! Graph has {graph.num_nodes} nodes")
```

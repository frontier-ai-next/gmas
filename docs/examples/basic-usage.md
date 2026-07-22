# Basic Usage Example

A simple multi-agent pipeline.

## Setup

```python
from gmas.core import AgentProfile
from gmas.builder import build_property_graph
from gmas.execution import MACPRunner
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ.get("LLM_BASE_URL"),
)
```

## Create Agents

```python
agents = [
    AgentProfile(
        agent_id="researcher",
        display_name="Researcher",
        description="Gathers information",
    ),
    AgentProfile(
        agent_id="writer",
        display_name="Writer",
        description="Writes final answer",
    ),
]
```

## Build Graph

```python
graph = build_property_graph(
    agents,
    workflow_edges=[("researcher", "writer")],
    query="What is quantum computing?",
)
```

## Execute

```python
def llm_caller(prompt: str) -> str:
    response = client.responses.create(
        model=os.environ["LLM_MODEL"],
        input=prompt,
    )
    return response.output_text

runner = MACPRunner(llm_caller=llm_caller)
result = runner.run_round(graph)

print(result.final_answer)
```

## Output

```
Execution order: ['researcher', 'writer']
Time: 3.45s
Answer: [The final answer from the writer agent]
```

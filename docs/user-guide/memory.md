# Memory System

Each agent maintains its own decentralized state, with optional shared memory.

## Agent State

Every `AgentProfile` has a `state` field — a list of message dictionaries:

```python
from gmas.core import AgentProfile

agent = AgentProfile(agent_id="researcher", display_name="Researcher")

# Read state (list of dicts)
current_state = agent.state  # []

# Append state immutably
agent = agent.append_state({"role": "user", "content": "query"})
agent = agent.append_state({"role": "assistant", "content": "response"})

# Replace state entirely
agent = agent.with_state([{"context": "new data"}])

# Clear state
agent = agent.clear_state()
```

Since `AgentProfile` is a frozen Pydantic model, all mutations return a **new** instance.

## Memory in Execution

Enable memory to have the runner automatically save and inject agent context:

```python
from gmas.execution import MACPRunner, RunnerConfig, MemoryConfig

config = RunnerConfig(
    enable_memory=True,
    memory_config=MemoryConfig(
        working_max_entries=10,     # short-term memory limit
        long_term_max_entries=50,   # long-term memory limit
    ),
    memory_context_limit=5,        # include last 5 entries in prompt
)

runner = MACPRunner(llm_caller=llm_caller, config=config)
result = runner.run_round(graph)
```

### How Memory Works

1. Before an agent runs, the runner fetches its recent memory entries and includes them in the prompt
2. After the agent responds, the response is saved to the agent's working memory
3. If incoming agents shared data, their responses are also recorded
4. The `memory_context_limit` controls how many entries are injected into the prompt

### Accessing Memory After Execution

```python
# Get an agent's memory
memory = runner.get_agent_memory("researcher")
if memory:
    print(memory.working)     # working memory entries
    print(memory.long_term)   # long-term entries
```

## SharedMemoryPool

Share data between agents through a centralized pool:

```python
from gmas.execution import SharedMemoryPool

pool = SharedMemoryPool()

# Write data (any agent can write)
pool.write("research_findings", "Key data about quantum computing")

# Read data (any agent can read)
value = pool.read("research_findings")

# Use with runner
runner = MACPRunner(
    llm_caller=llm_caller,
    memory_pool=pool,
)
```

The shared pool is useful for:

- Sharing context that isn't passed via graph edges
- Accumulating data from multiple agents
- Providing global configuration or state to all agents

## Memory + Graph Flow

In a typical pipeline, memory complements the graph edge communication:

```text
researcher ──(edge)──> analyst ──(edge)──> writer

 researcher writes to shared pool
 analyst reads from shared pool + gets researcher's output via edge
 writer reads from shared pool + gets analyst's output via edge
```

Edge communication passes the immediate predecessor's response. Memory provides accumulated context across the entire run.

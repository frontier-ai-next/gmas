# Multi-Model Support

Use different LLM models for different agents.

## Per-Agent LLM Configuration

Each agent can have its own LLM configuration:

```python
from gmas.core import AgentProfile, AgentLLMConfig

agents = [
    AgentProfile(
        agent_id="researcher",
        display_name="Researcher",
        llm_config=AgentLLMConfig(
            model_name="gpt-4",
            temperature=0.3,
            max_tokens=2000,
        ),
    ),
    AgentProfile(
        agent_id="writer",
        display_name="Writer",
        llm_config=AgentLLMConfig(
            model_name="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000,
        ),
    ),
]
```

## LLMCallerFactory

The `LLMCallerFactory` creates callers dynamically based on agent configurations:

```python
from gmas.execution import MACPRunner, LLMCallerFactory

# Create factory with OpenAI defaults
factory = LLMCallerFactory.create_openai_factory(
    default_api_key="sk-...",
    default_model="gpt-4o-mini",
    default_temperature=0.7,
    default_max_tokens=2000,
)

# Per-agent caller overrides via dict
runner = MACPRunner(
    llm_callers={
        "researcher": gpt4_caller,
        "writer": claude_caller,
        "reviewer": local_llm_caller,
    },
    llm_factory=factory,
)
```

## Caller Resolution Priority

When executing an agent, the runner resolves the LLM caller in this order:

1. **`llm_callers[agent_id]`** — explicit per-agent mapping
2. **`llm_factory`** — factory builds a caller from the agent's `AgentLLMConfig`
3. **`llm_caller`** — default fallback caller

## Structured Prompts

For chat-style LLMs (OpenAI, Anthropic), use structured callers that send message arrays instead of flat strings:

```python
from gmas.execution import MACPRunner, create_openai_structured_caller

runner = MACPRunner(
    structured_llm_caller=create_openai_structured_caller(
        api_key="sk-...",
        model="gpt-4o",
    ),
)
```

This sends properly formatted `[{role: "system", ...}, {role: "user", ...}]` messages.

## Async Multi-Model

Combine async callers with per-agent mapping:

```python
runner = MACPRunner(
    async_llm_callers={
        "researcher": async_gpt4_caller,
        "writer": async_claude_caller,
    },
)

result = await runner.arun_round(graph)
```

## Factory Configuration

`LLMCallerFactory.create_openai_factory()` parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_api_key` | `str \| None` | `None` | Fallback API key |
| `default_model` | `str` | `"gpt-4"` | Default model name |
| `default_base_url` | `str` | `"https://api.openai.com/v1"` | API base URL |
| `default_temperature` | `float` | `0.7` | Default temperature |
| `default_max_tokens` | `int` | `2000` | Default max tokens |

Each agent's `AgentLLMConfig` overrides these defaults for that agent.

## OpenAI Caller Helpers

Module-level helper functions for creating callers:

```python
from gmas.execution import (
    create_openai_caller,
    create_openai_structured_caller,
    create_openai_async_structured_caller,
)

# Simple string-in/string-out caller
caller = create_openai_caller(api_key="sk-...", model="gpt-4o")

# Structured (chat messages) caller
structured = create_openai_structured_caller(api_key="sk-...", model="gpt-4o")

# Async structured caller
async_structured = create_openai_async_structured_caller(
    api_key="sk-...",
    model="gpt-4o",
)
```

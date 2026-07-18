# Execution API

## Execution Module

```python
from gmas.execution import (
    # Runner
    MACPRunner,
    MACPResult,
    RunnerConfig,

    # Scheduling
    build_execution_order,
    get_parallel_groups,
    AdaptiveScheduler,

    # Streaming
    StreamEventType,
    StreamEvent,
    StreamBuffer,
    stream_to_string,
    print_stream,

    # Budget
    BudgetConfig,
    BudgetTracker,

    # Errors
    ErrorPolicy,
    ExecutionError,
    BudgetExceededError,

    # Memory
    MemoryConfig,
    SharedMemoryPool,
    AgentMemory,

    # State
    HiddenState,
    StepContext,
    TopologyAction,
    TopologyHook,
    EarlyStopCondition,

    # LLM
    LLMCallerFactory,
    LLMCallerProtocol,
    AsyncLLMCallerProtocol,
    StructuredLLMCallerProtocol,
    create_openai_caller,
    create_openai_structured_caller,
    create_openai_async_structured_caller,

    # Callbacks
    CallbackManager,
    AsyncCallbackManager,
)
```

## MACPRunner

Main execution engine, assembled from mixins: `RunnerStreamMixin`, `RunnerBatchMixin`, `RunnerExecutionMixin`, `RunnerTopologyMixin`, `RunnerCoreMixin`.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_caller` | `Callable[[str], str] \| None` | `None` | Synchronous LLM caller |
| `async_llm_caller` | `Callable[[str], Awaitable[str]] \| None` | `None` | Async LLM caller |
| `streaming_llm_caller` | `Callable[[str], Iterator[str]] \| None` | `None` | Streaming LLM caller |
| `async_streaming_llm_caller` | `Callable[[str], AsyncIterator[str]] \| None` | `None` | Async streaming caller |
| `token_counter` | `Callable[[str], int] \| None` | `None` | Custom token counter |
| `config` | `RunnerConfig \| None` | `None` | Configuration |
| `timeout` | `int` | `60` | Per-agent timeout (seconds) |
| `memory_pool` | `SharedMemoryPool \| None` | `None` | Shared memory pool |
| `llm_callers` | `dict[str, Callable] \| None` | `None` | Per-agent caller mapping |
| `async_llm_callers` | `dict[str, Callable] \| None` | `None` | Per-agent async caller mapping |
| `llm_factory` | `LLMCallerFactory \| None` | `None` | Factory for multi-model |
| `tool_registry` | `Any \| None` | `None` | Tool registry |
| `structured_llm_caller` | `StructuredLLMCallerProtocol \| None` | `None` | Structured (chat) caller |
| `async_structured_llm_caller` | `AsyncStructuredLLMCallerProtocol \| None` | `None` | Async structured caller |

### Methods

```python
# Synchronous execution
result = runner.run_round(graph, final_agent_id=None, start_agent_id=None,
                          *, update_states=None, filter_unreachable=False,
                          callbacks=None) -> MACPResult

# Async execution
result = await runner.arun_round(graph, ...) -> MACPResult

# Streaming
for event in runner.stream(graph, final_agent_id=None, *, update_states=None):
    ...

# Async streaming
async for event in runner.astream(graph, ...):
    ...

# Hidden state transfer
result = runner.run_round_with_hidden(graph, final_agent_id=None,
                                       hidden_encoder=None) -> MACPResult

# Stream + result
events, result = runner.stream_to_result(graph, ...)

# Access agent memory
memory = runner.get_agent_memory("agent_id") -> AgentMemory | None
```

## MACPResult

Result of a round execution (NamedTuple):

| Field | Type | Description |
|-------|------|-------------|
| `messages` | `dict[str, str]` | Agent responses by ID |
| `final_answer` | `str` | Final agent's output |
| `final_agent_id` | `str` | ID of final agent |
| `execution_order` | `list[str]` | Order agents ran in |
| `agent_states` | `dict \| None` | Updated agent state histories |
| `step_results` | `dict[str, StepResult] \| None` | Per-step results |
| `step_results_by_step` | `dict[str, StepResult] \| None` | Results by step ID |
| `messages_by_step` | `dict[str, str] \| None` | Messages by step ID |
| `total_tokens` | `int` | Estimated token usage |
| `total_time` | `float` | Wall-clock seconds |
| `topology_changed_count` | `int` | Topology changes during run |
| `fallback_count` | `int` | Fallback activations |
| `pruned_agents` | `list[str] \| None` | Pruned agent IDs |
| `errors` | `list[ExecutionError] \| None` | Errors encountered |
| `hidden_states` | `dict[str, HiddenState] \| None` | Agent hidden states |
| `metrics` | `ExecutionMetrics \| None` | Collected metrics |
| `budget_summary` | `dict \| None` | Budget consumption |
| `early_stopped` | `bool` | Whether early-stop triggered |
| `early_stop_reason` | `str \| None` | Reason for early stop |
| `topology_modifications` | `int` | Number of modifications |

## RunnerConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `timeout` | `float` | `60.0` | Per-agent timeout (seconds) |
| `adaptive` | `bool` | `False` | Enable adaptive execution |
| `enable_parallel` | `bool` | `True` | Enable parallel execution |
| `max_parallel_size` | `int` | `5` | Max agents in parallel group |
| `max_retries` | `int` | `2` | Max retry attempts |
| `retry_delay` | `float` | `1.0` | Initial retry delay (seconds) |
| `retry_backoff` | `float` | `2.0` | Exponential backoff multiplier |
| `update_states` | `bool` | `True` | Update agent states after run |
| `routing_policy` | `RoutingPolicy` | `TOPOLOGICAL` | Execution routing policy |
| `pruning_config` | `PruningConfig \| None` | `None` | Pruning configuration |
| `enable_hidden_channels` | `bool` | `False` | Enable hidden state transfer |
| `hidden_combine_strategy` | `str` | `"mean"` | How to combine hidden states |
| `pass_embeddings` | `bool` | `True` | Pass embeddings in hidden state |
| `error_policy` | `ErrorPolicy` | `RAISE` | Error handling policy |
| `budget_config` | `BudgetConfig \| None` | `None` | Budget limits |
| `enable_memory` | `bool` | `False` | Enable memory system |
| `memory_config` | `MemoryConfig \| None` | `None` | Memory configuration |
| `memory_context_limit` | `int` | `5` | Memory entries in prompt |
| `enable_token_streaming` | `bool` | `False` | Enable token-level streaming |
| `enable_dynamic_topology` | `bool` | `False` | Enable topology modifications |
| `topology_hooks` | `list` | `[]` | Sync topology hooks |
| `async_topology_hooks` | `list` | `[]` | Async topology hooks |
| `early_stop_conditions` | `list` | `[]` | Early stop conditions |
| `max_tool_iterations` | `int` | `3` | Max tool-calling loops |
| `max_loop_iterations` | `int` | `5` | Max adaptive loop iterations |

## StreamEventType

All streaming event types:

| Member | Value | Description |
|--------|-------|-------------|
| `RUN_START` | `"run_start"` | Execution started |
| `RUN_END` | `"run_end"` | Execution finished |
| `AGENT_START` | `"agent_start"` | Agent began processing |
| `AGENT_OUTPUT` | `"agent_output"` | Agent produced output |
| `AGENT_ERROR` | `"agent_error"` | Agent error |
| `TOKEN` | `"token"` | Single token streamed |
| `TOPOLOGY_CHANGED` | `"topology_changed"` | Graph modified |
| `PRUNE` | `"prune"` | Agent pruned |
| `FALLBACK` | `"fallback"` | Fallback activated |
| `PARALLEL_START` | `"parallel_start"` | Parallel group started |
| `PARALLEL_END` | `"parallel_end"` | Parallel group ended |
| `MEMORY_WRITE` | `"memory_write"` | Memory entry written |
| `MEMORY_READ` | `"memory_read"` | Memory entries read |
| `BUDGET_WARNING` | `"budget_warning"` | Budget threshold crossed |
| `BUDGET_EXCEEDED` | `"budget_exceeded"` | Budget limit exceeded |

## ErrorPolicy

| Member | Behavior |
|--------|----------|
| `RAISE` | Raise exception immediately |
| `CONTINUE` | Skip failed agent, continue |
| `RETRY` | Retry before deciding |

## TopologyAction

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `early_stop` | `bool` | `False` | Stop execution |
| `early_stop_reason` | `str \| None` | `None` | Reason |
| `add_edges` | `list[tuple[str, str, float]]` | `[]` | Edges to add |
| `remove_edges` | `list[tuple[str, str]]` | `[]` | Edges to remove |
| `skip_agents` | `list[str]` | `[]` | Agents to skip |
| `force_agents` | `list[str]` | `[]` | Agents to force-execute |
| `condition_skip_agents` | `list[str]` | `[]` | Conditionally skip |
| `condition_unskip_agents` | `list[str]` | `[]` | Conditionally unskip |
| `insert_chains` | `list[tuple[str, str]]` | `[]` | Insert agent chains |
| `new_end_agent` | `str \| None` | `None` | Change final agent |
| `trigger_rebuild` | `bool` | `False` | Force plan rebuild |

## EarlyStopCondition

Factory methods for creating stop conditions:

```python
EarlyStopCondition.on_keyword(keyword, reason=None, *, case_sensitive=False)
EarlyStopCondition.on_token_limit(max_tokens, reason=None)
EarlyStopCondition.on_agent_count(max_agents, reason=None)
EarlyStopCondition.on_metadata(key, value=None, comparator=None, reason=None)
EarlyStopCondition.on_custom(condition, reason="Custom condition met", **kwargs)
EarlyStopCondition.combine_any(conditions, reason="One of conditions met")
EarlyStopCondition.combine_all(conditions, reason="All conditions met")
```

## LLMCallerFactory

```python
factory = LLMCallerFactory.create_openai_factory(
    default_api_key=None,
    default_model="gpt-4",
    default_base_url="https://api.openai.com/v1",
    default_temperature=0.7,
    default_max_tokens=2000,
)

# Get caller for an agent
caller = factory.get_caller(agent.llm_config, _agent_id=agent.agent_id)
async_caller = factory.get_async_caller(agent.llm_config, _agent_id=agent.agent_id)
```

## Budget

```python
budget = BudgetConfig(
    total_token_limit=10000,
    max_prompt_length=4000,
    time_limit_seconds=300,
)
```

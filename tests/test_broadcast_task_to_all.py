"""Regression tests for broadcast_task_to_all=False."""

import pytest

from gmas.builder.graph_builder import BuilderConfig, GraphBuilder
from gmas.execution.runner import MACPRunner, RunnerConfig
from gmas.execution.runner.state import StepContext, TopologyAction
from gmas.execution.streaming import AgentStartEvent

TASK_QUERY = "-=QUERY=-"
TASK_LINE = f"Task: {TASK_QUERY}"

AGENT_PROMPT_MARKERS: dict[str, str] = {
    "coordinator": "You are Coordinator",
    "expert": "You are Domain expert",
    "summarizer": "You are Summarizer",
}


def _build_selective_task_graph():
    builder = GraphBuilder(config=BuilderConfig(check_cycles=False))
    builder.add_agent("coordinator", persona="Coordinator", description="Coordinate the task.")
    builder.add_agent("expert", persona="Domain expert", description="Provide expert analysis.")
    builder.add_agent("summarizer", persona="Summarizer", description="Summarize results.")
    builder.add_workflow_edge("coordinator", "expert")
    builder.add_workflow_edge("coordinator", "summarizer")
    builder.add_workflow_edge("expert", "summarizer")
    builder.set_start_node("coordinator")
    builder.set_end_node("summarizer")
    builder.add_task(query=TASK_QUERY)
    builder.connect_task_to_agents(agent_ids=["coordinator"])
    return builder.build()


def _runner_config(*, adaptive: bool, enable_parallel: bool = False) -> RunnerConfig:
    return RunnerConfig(
        adaptive=adaptive,
        broadcast_task_to_all=False,
        max_loop_iterations=1,
        update_states=False,
        enable_memory=False,
        enable_parallel=enable_parallel,
        prompt_preview_length=400,
        timeout=60.0,
    )


class _PromptCapture:
    def __init__(self) -> None:
        self.prompts: dict[str, str] = {}

    def _identify_agent(self, prompt: str) -> str | None:
        for agent_id, marker in AGENT_PROMPT_MARKERS.items():
            if marker in prompt:
                return agent_id
        return None

    def sync_caller(self, prompt: str) -> str:
        agent_id = self._identify_agent(prompt)
        if agent_id is not None:
            self.prompts[agent_id] = prompt
        return "mock"

    async def async_caller(self, prompt: str) -> str:
        return self.sync_caller(prompt)


def _assert_selective_task_broadcast(prompts: dict[str, str]) -> None:
    expected = {"coordinator": True, "expert": False, "summarizer": False}
    for agent_id, should_include in expected.items():
        assert agent_id in prompts, f"no prompt captured for {agent_id!r}, got: {list(prompts)}"
        has_task = TASK_LINE in prompts[agent_id]
        assert has_task is should_include, (
            f"agent {agent_id!r}: expected Task line={should_include}, prompt excerpt:\n{prompts[agent_id][:400]}"
        )


class TestBroadcastTaskHelpers:
    def test_should_include_query_for_agent_uses_live_graph(self):
        graph = _build_selective_task_graph()
        runner = MACPRunner(config=RunnerConfig(broadcast_task_to_all=False))
        assert runner._should_include_query_for_agent(graph, "coordinator") is True
        assert runner._should_include_query_for_agent(graph, "expert") is False

        task_node = graph.task_node
        assert task_node is not None
        graph.add_edge(task_node, "expert", 1.0)
        assert runner._should_include_query_for_agent(graph, "expert") is True
        graph.remove_edge(task_node, "expert")
        assert runner._should_include_query_for_agent(graph, "expert") is False

    def test_node_position_cache_invalidates_after_node_removal(self):
        builder = GraphBuilder(config=BuilderConfig(check_cycles=False))
        builder.add_agent("first", persona="First", description="Run first.")
        builder.add_agent("target", persona="Target", description="Handle the task.")
        builder.add_task(query=TASK_QUERY)
        builder.connect_task_to_agents(agent_ids=["target"])
        graph = builder.build()
        runner = MACPRunner(config=RunnerConfig(broadcast_task_to_all=False))

        assert runner._should_include_query_for_agent(graph, "target") is True
        graph.remove_node("first")
        assert runner._should_include_query_for_agent(graph, "target") is True

    def test_broadcast_true_does_not_require_a_task_edge(self):
        graph = _build_selective_task_graph()
        runner = MACPRunner(config=RunnerConfig(broadcast_task_to_all=True))
        assert runner._should_include_query_for_agent(graph, "expert") is True

    def test_should_include_query_for_agent_requires_direct_task_to_agent_edge(self):
        graph = _build_selective_task_graph()
        task_node = graph.task_node
        assert task_node is not None
        runner = MACPRunner(config=RunnerConfig(broadcast_task_to_all=False))

        assert runner._should_include_query_for_agent(graph, "expert") is False

        graph.add_edge("expert", task_node, 1.0)
        assert runner._should_include_query_for_agent(graph, "expert") is False

        graph.add_edge(task_node, "expert", 1.0)
        assert runner._should_include_query_for_agent(graph, "expert") is True


@pytest.mark.parametrize("adaptive", [False, True], ids=["simple", "adaptive"])
def test_sync_run_respects_broadcast_task_to_all_false(adaptive: bool):
    capture = _PromptCapture()
    graph = _build_selective_task_graph()
    runner = MACPRunner(llm_caller=capture.sync_caller, config=_runner_config(adaptive=adaptive))
    result = runner.run_round(graph)
    assert set(result.execution_order) == {"coordinator", "expert", "summarizer"}
    _assert_selective_task_broadcast(capture.prompts)


@pytest.mark.parametrize("adaptive", [False, True], ids=["simple", "adaptive"])
@pytest.mark.asyncio
async def test_async_run_respects_broadcast_task_to_all_false(adaptive: bool):
    capture = _PromptCapture()
    graph = _build_selective_task_graph()
    runner = MACPRunner(async_llm_caller=capture.async_caller, config=_runner_config(adaptive=adaptive))
    result = await runner.arun_round(graph)
    assert set(result.execution_order) == {"coordinator", "expert", "summarizer"}
    _assert_selective_task_broadcast(capture.prompts)


def test_structured_caller_messages_respect_broadcast_task_to_all_false():
    captured_messages: dict[str, list[dict[str, str]]] = {}

    def structured_caller(messages: list[dict[str, str]]) -> str:
        joined = "\n".join(message.get("content", "") for message in messages)
        for agent_id, marker in AGENT_PROMPT_MARKERS.items():
            if marker in joined:
                captured_messages[agent_id] = messages
                break
        return "mock"

    graph = _build_selective_task_graph()
    runner = MACPRunner(
        structured_llm_caller=structured_caller,
        config=_runner_config(adaptive=True),
    )
    result = runner.run_round(graph)
    assert set(result.execution_order) == {"coordinator", "expert", "summarizer"}

    prompt_text = {
        agent_id: "\n".join(message.get("content", "") for message in messages)
        for agent_id, messages in captured_messages.items()
    }
    _assert_selective_task_broadcast(prompt_text)


@pytest.mark.asyncio
async def test_async_structured_caller_messages_respect_broadcast_task_to_all_false():
    captured_messages: dict[str, list[dict[str, str]]] = {}

    async def structured_caller(messages: list[dict[str, str]]) -> str:
        joined = "\n".join(message.get("content", "") for message in messages)
        for agent_id, marker in AGENT_PROMPT_MARKERS.items():
            if marker in joined:
                captured_messages[agent_id] = messages
                break
        return "mock"

    graph = _build_selective_task_graph()
    runner = MACPRunner(
        async_structured_llm_caller=structured_caller,
        config=_runner_config(adaptive=True),
    )
    result = await runner.arun_round(graph)
    assert set(result.execution_order) == {"coordinator", "expert", "summarizer"}

    prompt_text = {
        agent_id: "\n".join(message.get("content", "") for message in messages)
        for agent_id, messages in captured_messages.items()
    }
    _assert_selective_task_broadcast(prompt_text)


@pytest.mark.asyncio
async def test_async_adaptive_parallel_group_respects_broadcast():
    capture = _PromptCapture()
    graph = _build_selective_task_graph()
    runner = MACPRunner(
        async_llm_caller=capture.async_caller,
        config=_runner_config(adaptive=True, enable_parallel=True),
    )
    result = await runner.arun_round(graph)
    assert "coordinator" in result.execution_order
    assert {"expert", "summarizer"} <= set(result.execution_order)
    _assert_selective_task_broadcast(capture.prompts)


@pytest.mark.parametrize("adaptive", [False, True], ids=["simple", "adaptive"])
def test_stream_prompt_preview_respects_broadcast(adaptive: bool):
    graph = _build_selective_task_graph()
    capture = _PromptCapture()
    runner = MACPRunner(llm_caller=capture.sync_caller, config=_runner_config(adaptive=adaptive))

    previews: dict[str, str] = {}
    for event in runner.stream(graph):
        if isinstance(event, AgentStartEvent):
            previews[event.agent_id] = event.prompt_preview

    assert set(previews) >= {"coordinator", "expert", "summarizer"}
    assert TASK_LINE in previews["coordinator"] or TASK_QUERY in previews["coordinator"]
    for agent_id in ("expert", "summarizer"):
        assert TASK_LINE not in previews[agent_id], f"{agent_id} stream preview must not contain task"
    _assert_selective_task_broadcast(capture.prompts)


@pytest.mark.parametrize("adaptive", [False, True], ids=["simple", "adaptive"])
@pytest.mark.asyncio
async def test_astream_prompt_preview_respects_broadcast(adaptive: bool):
    graph = _build_selective_task_graph()
    capture = _PromptCapture()
    runner = MACPRunner(async_llm_caller=capture.async_caller, config=_runner_config(adaptive=adaptive))

    previews: dict[str, str] = {}
    async for event in runner.astream(graph):
        if isinstance(event, AgentStartEvent):
            previews[event.agent_id] = event.prompt_preview

    assert TASK_LINE in previews["coordinator"]
    for agent_id in ("expert", "summarizer"):
        assert TASK_LINE not in previews[agent_id]
    _assert_selective_task_broadcast(capture.prompts)


@pytest.mark.asyncio
async def test_astream_uses_task_edge_added_by_topology_hook():
    graph = _build_selective_task_graph()
    task_node = graph.task_node
    assert task_node is not None

    async def hook(ctx: StepContext, _role_graph):
        if ctx.agent_id == "coordinator":
            return TopologyAction(add_edges=[(task_node, "expert", 1.0)])
        return None

    capture = _PromptCapture()
    runner = MACPRunner(
        async_llm_caller=capture.async_caller,
        config=RunnerConfig(
            adaptive=True,
            broadcast_task_to_all=False,
            enable_dynamic_topology=True,
            async_topology_hooks=[hook],
            update_states=False,
            enable_memory=False,
        ),
    )

    async for _event in runner.astream(graph):
        pass

    assert TASK_LINE in capture.prompts["expert"]


def test_broadcast_task_to_all_true_sends_query_to_all():
    prompts: dict[str, str] = {}

    def caller(prompt: str) -> str:
        for agent_id, marker in AGENT_PROMPT_MARKERS.items():
            if marker in prompt:
                prompts[agent_id] = prompt
        return "mock"

    graph = _build_selective_task_graph()
    runner = MACPRunner(
        llm_caller=caller,
        config=RunnerConfig(adaptive=True, broadcast_task_to_all=True, max_loop_iterations=1),
    )
    runner.run_round(graph)
    for agent_id in AGENT_PROMPT_MARKERS:
        assert TASK_LINE in prompts[agent_id], f"{agent_id} should receive task when broadcast=True"


@pytest.mark.parametrize("adaptive", [False, True], ids=["simple", "adaptive"])
def test_dynamic_topology_add_task_edge_includes_query_on_next_agent(adaptive: bool):
    graph = _build_selective_task_graph()
    task_node = graph.task_node
    assert task_node is not None

    def hook(ctx: StepContext, _role_graph):
        if ctx.agent_id == "coordinator":
            return TopologyAction(add_edges=[(task_node, "expert", 1.0)])
        return None

    capture = _PromptCapture()
    runner = MACPRunner(
        llm_caller=capture.sync_caller,
        config=RunnerConfig(
            adaptive=adaptive,
            broadcast_task_to_all=False,
            enable_dynamic_topology=True,
            topology_hooks=[hook],
            update_states=False,
            enable_memory=False,
        ),
    )
    runner.run_round(graph)
    assert TASK_LINE in capture.prompts["coordinator"]
    assert TASK_LINE in capture.prompts["expert"], "expert should receive task after runtime edge add"


@pytest.mark.parametrize("adaptive", [False, True], ids=["simple", "adaptive"])
def test_dynamic_topology_remove_task_edge_excludes_query(adaptive: bool):
    builder = GraphBuilder(config=BuilderConfig(check_cycles=False))
    builder.add_agent("coordinator", persona="Coordinator", description="Coordinate the task.")
    builder.add_agent("expert", persona="Domain expert", description="Provide expert analysis.")
    builder.add_workflow_edge("coordinator", "expert")
    builder.set_start_node("coordinator")
    builder.set_end_node("expert")
    builder.add_task(query=TASK_QUERY)
    builder.connect_task_to_agents(agent_ids=["coordinator", "expert"])
    graph = builder.build()
    task_node = graph.task_node
    assert task_node is not None

    def hook(ctx: StepContext, _role_graph):
        if ctx.agent_id == "coordinator":
            return TopologyAction(remove_edges=[(task_node, "expert")])
        return None

    capture = _PromptCapture()
    runner = MACPRunner(
        llm_caller=capture.sync_caller,
        config=RunnerConfig(
            adaptive=adaptive,
            broadcast_task_to_all=False,
            enable_dynamic_topology=True,
            topology_hooks=[hook],
            update_states=False,
            enable_memory=False,
        ),
    )
    runner.run_round(graph)
    assert TASK_LINE in capture.prompts["coordinator"]
    assert TASK_LINE not in capture.prompts["expert"]


@pytest.mark.asyncio
async def test_async_dynamic_topology_add_task_edge_includes_query():
    graph = _build_selective_task_graph()
    task_node = graph.task_node
    assert task_node is not None

    async def hook(ctx: StepContext, _role_graph):
        if ctx.agent_id == "coordinator":
            return TopologyAction(add_edges=[(task_node, "expert", 1.0)])
        return None

    capture = _PromptCapture()
    runner = MACPRunner(
        async_llm_caller=capture.async_caller,
        config=RunnerConfig(
            adaptive=True,
            broadcast_task_to_all=False,
            enable_dynamic_topology=True,
            async_topology_hooks=[hook],
            update_states=False,
            enable_memory=False,
        ),
    )
    await runner.arun_round(graph)
    assert TASK_LINE in capture.prompts["expert"]


@pytest.mark.asyncio
async def test_async_dynamic_topology_remove_task_edge_excludes_query():
    builder = GraphBuilder(config=BuilderConfig(check_cycles=False))
    builder.add_agent("coordinator", persona="Coordinator", description="Coordinate the task.")
    builder.add_agent("expert", persona="Domain expert", description="Provide expert analysis.")
    builder.add_workflow_edge("coordinator", "expert")
    builder.set_start_node("coordinator")
    builder.set_end_node("expert")
    builder.add_task(query=TASK_QUERY)
    builder.connect_task_to_agents(agent_ids=["coordinator", "expert"])
    graph = builder.build()
    task_node = graph.task_node
    assert task_node is not None

    async def hook(ctx: StepContext, _role_graph):
        if ctx.agent_id == "coordinator":
            return TopologyAction(remove_edges=[(task_node, "expert")])
        return None

    capture = _PromptCapture()
    runner = MACPRunner(
        async_llm_caller=capture.async_caller,
        config=RunnerConfig(
            adaptive=True,
            broadcast_task_to_all=False,
            enable_dynamic_topology=True,
            async_topology_hooks=[hook],
            update_states=False,
            enable_memory=False,
        ),
    )
    await runner.arun_round(graph)
    assert TASK_LINE in capture.prompts["coordinator"]
    assert TASK_LINE not in capture.prompts["expert"]


def test_broadcast_task_to_all_e2e_example_runs_without_external_llm():
    from examples.broadcast_task_to_all_e2e import assert_selective_broadcast, run_demo

    result = run_demo(live=False, adaptive=True)
    assert result["execution_order"] == ["coordinator", "expert", "summarizer"]
    assert_selective_broadcast(result)

"""Tests for MACPRunner tool-calling paths (execution mixin)."""

from typing import ClassVar

import pytest

from gmas.core.agent import AgentProfile
from gmas.execution.runner import MACPRunner, RunnerConfig
from gmas.execution.runner.prompting import StructuredPrompt
from gmas.execution.usage import LLMCallResult, LLMUsage
from gmas.tools.base import BaseTool, ToolResult
from gmas.tools.llm_integration import LLMResponse, LLMToolCall


class _EchoTool(BaseTool):
    @property
    def name(self) -> str:
        return "echo_tool"

    @property
    def description(self) -> str:
        return "Echoes input."

    @property
    def parameters_schema(self) -> dict:
        return {"type": "object", "properties": {"msg": {"type": "string"}}}

    def execute(self, **kwargs):
        return ToolResult(tool_name=self.name, success=True, output=kwargs.get("msg", ""))


class TestRunAgentWithToolsSync:
    def test_tool_round_then_text_answer(self):
        """LLM requests a tool once, then returns plain text."""
        agent = AgentProfile(agent_id="a0", display_name="A", tools=[_EchoTool()])
        phase = {"n": 0}

        def caller(prompt, tools=None):
            if tools:
                if phase["n"] == 0:
                    phase["n"] += 1
                    return LLMResponse(
                        content="",
                        tool_calls=[
                            LLMToolCall(id="1", name="echo_tool", arguments={"msg": "hi"}),
                        ],
                    )
                return LLMResponse(content="final answer", tool_calls=[])
            return LLMResponse(content="final-from-plain", tool_calls=[])

        runner = MACPRunner(config=RunnerConfig(max_tool_iterations=4))
        text, usage = runner._run_agent_with_tools(caller, "user task", agent)
        assert "final answer" in text
        assert usage.total_tokens >= 0

    def test_provider_usage_is_used_for_tool_loop(self):
        agent = AgentProfile(agent_id="a0", display_name="A", tools=[_EchoTool()])

        def caller(prompt, tools=None):
            if tools:
                return LLMResponse(
                    content="final answer",
                    tool_calls=[],
                    usage=LLMUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
                )
            return LLMCallResult(content="unused", usage=LLMUsage(total_tokens=1))

        runner = MACPRunner(config=RunnerConfig(max_tool_iterations=2))
        _text, usage = runner._run_agent_with_tools(caller, "user task", agent)
        assert usage.total_tokens == 120
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 20

    def test_caller_returns_string_with_tools_exits_early(self):
        agent = AgentProfile(agent_id="a0", display_name="A", tools=[_EchoTool()])

        def caller(prompt, tools=None):
            if tools:
                return "shortcut-string"
            return "no-tools"

        runner = MACPRunner()
        text, usage = runner._run_agent_with_tools(caller, "x", agent)
        assert text == "shortcut-string"
        assert usage.total_tokens > 0

    def test_provider_usage_on_string_result_is_authoritative(self):
        agent = AgentProfile(agent_id="a0", display_name="A", tools=[_EchoTool()])

        class ProviderText(str):  # noqa: SLOT000
            usage: ClassVar[dict[str, int]] = {
                "prompt_tokens": 70,
                "completion_tokens": 30,
                "total_tokens": 100,
            }

        def caller(prompt, tools=None):
            return ProviderText("short")

        runner = MACPRunner(token_counter=lambda _value: 1)
        text, usage = runner._run_agent_with_tools(caller, "prompt", agent)
        assert text == "short"
        assert usage.prompt_tokens == 70
        assert usage.completion_tokens == 30
        assert usage.total_tokens == 100

    def test_provider_usage_is_preserved_for_structured_prompt(self):
        agent = AgentProfile(agent_id="a0", display_name="A")
        prompt = StructuredPrompt("flat prompt", [{"role": "user", "content": "structured prompt"}])

        def caller(_prompt):
            return LLMCallResult(
                content="answer",
                usage=LLMUsage(prompt_tokens=70, completion_tokens=30, total_tokens=100),
            )

        runner = MACPRunner(token_counter=lambda _value: 1)
        text, usage = runner._run_agent_with_tools(caller, prompt, agent)

        assert text == "answer"
        assert usage.prompt_tokens == 70
        assert usage.completion_tokens == 30
        assert usage.total_tokens == 100

    def test_explicit_structured_caller_preserves_provider_usage(self):
        agent = AgentProfile(agent_id="a0", display_name="A")
        prompt = StructuredPrompt("flat prompt", [{"role": "user", "content": "structured prompt"}])

        def structured_caller(_messages):
            return LLMCallResult(
                content="answer",
                usage=LLMUsage(prompt_tokens=60, completion_tokens=40, total_tokens=100),
            )

        runner = MACPRunner(
            structured_llm_caller=structured_caller,
            token_counter=lambda _value: 1,
        )
        text, usage = runner._run_agent_with_tools(runner.llm_caller, prompt, agent)

        assert text == "answer"
        assert usage.prompt_tokens == 60
        assert usage.completion_tokens == 40
        assert usage.total_tokens == 100

    def test_caller_without_tools_param_plain_prompt(self):
        agent = AgentProfile(agent_id="a0", display_name="A", tools=[_EchoTool()])

        def plain_caller(prompt: str):
            return "no-tools-param"

        runner = MACPRunner()
        text, _toks = runner._run_agent_with_tools(plain_caller, "hello", agent)
        assert text == "no-tools-param"

    def test_structured_prompt_non_structured_caller_branch(self):
        agent = AgentProfile(agent_id="a0", display_name="A", tools=[_EchoTool()])
        sp = StructuredPrompt("flat", [{"role": "user", "content": "u"}])
        calls = [0]

        def caller(prompt, tools=None):
            calls[0] += 1
            if tools and calls[0] == 1:
                return LLMResponse(
                    content="",
                    tool_calls=[LLMToolCall(id="1", name="echo_tool", arguments={"msg": "x"})],
                )
            return LLMResponse(content="done", tool_calls=[])

        runner = MACPRunner(config=RunnerConfig(max_tool_iterations=3))
        text, _ = runner._run_agent_with_tools(caller, sp, agent)
        assert text == "done"

    def test_duplicate_tool_calls_all_cached_breaks_loop(self):
        agent = AgentProfile(agent_id="a0", display_name="A", tools=[_EchoTool()])
        tc = LLMToolCall(id="1", name="echo_tool", arguments={"msg": "same"})

        def caller(prompt, tools=None):
            return LLMResponse(content="", tool_calls=[tc])

        runner = MACPRunner(config=RunnerConfig(max_tool_iterations=3))
        text, _ = runner._run_agent_with_tools(caller, "p", agent)
        assert isinstance(text, str)


class TestRunAgentWithToolsAsync:
    @pytest.mark.asyncio
    async def test_async_tool_round_then_text(self):
        agent = AgentProfile(agent_id="a0", display_name="A", tools=[_EchoTool()])
        phase = {"n": 0}

        async def async_caller(prompt, tools=None):
            if tools:
                if phase["n"] == 0:
                    phase["n"] += 1
                    return LLMResponse(
                        content="",
                        tool_calls=[
                            LLMToolCall(id="1", name="echo_tool", arguments={"msg": "async-hi"}),
                        ],
                    )
                return LLMResponse(content="async final", tool_calls=[])
            return LLMResponse(content="plain", tool_calls=[])

        runner = MACPRunner(config=RunnerConfig(max_tool_iterations=4))
        text, usage = await runner._run_agent_with_tools_async(async_caller, "task", agent)
        assert "async final" in text
        assert usage.total_tokens >= 0

    @pytest.mark.asyncio
    async def test_async_caller_returns_string_with_tools(self):
        agent = AgentProfile(agent_id="a0", display_name="A", tools=[_EchoTool()])

        async def async_caller(prompt, tools=None):
            if tools:
                return "async-shortcut"
            return "x"

        runner = MACPRunner()
        text, _ = await runner._run_agent_with_tools_async(async_caller, "z", agent)
        assert text == "async-shortcut"

    @pytest.mark.asyncio
    async def test_async_provider_usage_on_string_result_is_authoritative(self):
        agent = AgentProfile(agent_id="a0", display_name="A", tools=[_EchoTool()])

        class ProviderText(str):  # noqa: SLOT000
            usage: ClassVar[dict[str, int]] = {"input_tokens": 80, "output_tokens": 20}

        async def async_caller(prompt, tools=None):
            return ProviderText("short")

        runner = MACPRunner(token_counter=lambda _value: 1)
        text, usage = await runner._run_agent_with_tools_async(async_caller, "prompt", agent)
        assert text == "short"
        assert usage.prompt_tokens == 80
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 100

    @pytest.mark.asyncio
    async def test_async_provider_usage_is_preserved_for_structured_prompt(self):
        agent = AgentProfile(agent_id="a0", display_name="A")
        prompt = StructuredPrompt("flat prompt", [{"role": "user", "content": "structured prompt"}])

        async def async_caller(_prompt):
            return LLMCallResult(
                content="answer",
                usage=LLMUsage(prompt_tokens=70, completion_tokens=30, total_tokens=100),
            )

        runner = MACPRunner(token_counter=lambda _value: 1)
        text, usage = await runner._run_agent_with_tools_async(async_caller, prompt, agent)

        assert text == "answer"
        assert usage.prompt_tokens == 70
        assert usage.completion_tokens == 30
        assert usage.total_tokens == 100

    @pytest.mark.asyncio
    async def test_explicit_async_structured_caller_preserves_provider_usage(self):
        agent = AgentProfile(agent_id="a0", display_name="A")
        prompt = StructuredPrompt("flat prompt", [{"role": "user", "content": "structured prompt"}])

        async def structured_caller(_messages):
            return LLMCallResult(
                content="answer",
                usage=LLMUsage(prompt_tokens=60, completion_tokens=40, total_tokens=100),
            )

        runner = MACPRunner(
            async_structured_llm_caller=structured_caller,
            token_counter=lambda _value: 1,
        )
        text, usage = await runner._run_agent_with_tools_async(None, prompt, agent)

        assert text == "answer"
        assert usage.prompt_tokens == 60
        assert usage.completion_tokens == 40
        assert usage.total_tokens == 100

    @pytest.mark.asyncio
    async def test_async_plain_caller_no_tools_param(self):
        agent = AgentProfile(agent_id="a0", display_name="A", tools=[_EchoTool()])

        async def plain(p: str):
            return "async-plain"

        runner = MACPRunner()
        text, _ = await runner._run_agent_with_tools_async(plain, "q", agent)
        assert text == "async-plain"

    @pytest.mark.asyncio
    async def test_async_structured_prompt_tool_loop(self):
        agent = AgentProfile(agent_id="a0", display_name="A", tools=[_EchoTool()])
        sp = StructuredPrompt("t", [{"role": "user", "content": "c"}])
        n = [0]

        async def async_caller(prompt, tools=None):
            n[0] += 1
            if tools and n[0] == 1:
                return LLMResponse(
                    content="",
                    tool_calls=[LLMToolCall(id="1", name="echo_tool", arguments={"msg": "y"})],
                )
            return LLMResponse(content="async-done", tool_calls=[])

        runner = MACPRunner(config=RunnerConfig(max_tool_iterations=3))
        text, _ = await runner._run_agent_with_tools_async(async_caller, sp, agent)
        assert text == "async-done"

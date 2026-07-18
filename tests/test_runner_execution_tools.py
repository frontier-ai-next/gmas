"""Tests for MACPRunner tool-calling paths (execution mixin)."""

import pytest

from gmas.core.agent import AgentProfile
from gmas.execution.runner import MACPRunner, RunnerConfig
from gmas.execution.runner.prompting import StructuredPrompt
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
        text, toks = runner._run_agent_with_tools(caller, "user task", agent)
        assert "final answer" in text
        assert toks >= 0

    def test_caller_returns_string_with_tools_exits_early(self):
        agent = AgentProfile(agent_id="a0", display_name="A", tools=[_EchoTool()])

        def caller(prompt, tools=None):
            if tools:
                return "shortcut-string"
            return "no-tools"

        runner = MACPRunner()
        text, toks = runner._run_agent_with_tools(caller, "x", agent)
        assert text == "shortcut-string"
        assert toks > 0

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
        text, toks = await runner._run_agent_with_tools_async(async_caller, "task", agent)
        assert "async final" in text
        assert toks >= 0

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

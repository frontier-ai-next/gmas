"""Tests for the Terminal-Bench comparison package."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from benchmarks.terminal_bench.configuration import (
    AgentsConfiguration,
    load_agents_configuration,
    tools_for_servers,
)
from benchmarks.terminal_bench.llm import LLMSettings, parse_structured_chat_with_retry
from benchmarks.terminal_bench.tools import ToolBox


class _RecordingEnvironment:
    def __init__(self) -> None:
        self.commands: list[tuple[str, int]] = []

    async def exec(self, command: str, timeout_sec: int) -> str:
        self.commands.append((command, timeout_sec))
        return "return_code=0\nstdout=ok"


class _Decision(BaseModel):
    is_done: bool


def test_checked_in_agents_configuration_is_valid() -> None:
    config = load_agents_configuration()
    agent_ids = {agent.agent_id for agent in config.agents}

    assert "coordinator" in agent_ids
    assert "test_verifier" in agent_ids
    assert config.conditional_edges


def test_agents_configuration_rejects_dangling_edge() -> None:
    raw: dict[str, Any] = load_agents_configuration().model_dump(by_alias=True)
    raw["conditional_edges"][0]["target"] = "missing-agent"

    with pytest.raises(ValidationError, match="unknown agents"):
        AgentsConfiguration.model_validate(raw)


def test_tools_for_servers_preserves_order_and_deduplicates() -> None:
    tools = tools_for_servers(["filesystem", "rust-mcp-filesystem", "git"])

    assert tools[:2] == ["read_file", "write_file"]
    assert len(tools) == len(set(tools))
    assert "git_status" in tools


def test_tools_for_servers_rejects_unknown_server() -> None:
    with pytest.raises(ValueError, match="unknown-server"):
        tools_for_servers(["unknown-server"])


def test_llm_settings_are_loaded_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    with pytest.raises(RuntimeError, match="LLM_API_KEY, LLM_MODEL"):
        LLMSettings.from_environment()


def test_structured_chat_parser_uses_parsed_completion() -> None:
    parsed = _Decision(is_done=True)
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))])
    client = Mock(spec=OpenAI)
    client.chat.completions.parse.return_value = completion

    result = parse_structured_chat_with_retry(
        client,
        "model",
        [{"role": "user", "content": "Decide"}],
        _Decision,
    )

    assert result is parsed
    client.chat.completions.parse.assert_called_once()


def test_structured_chat_parser_rejects_missing_parsed_payload() -> None:
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=None))])
    client = Mock(spec=OpenAI)
    client.chat.completions.parse.return_value = completion

    with pytest.raises(ValueError, match="did not return"):
        parse_structured_chat_with_retry(
            client,
            "model",
            [{"role": "user", "content": "Decide"}],
            _Decision,
        )

    assert client.chat.completions.parse.call_count == 3


def test_python_file_tools_encode_path_before_building_shell_command() -> None:
    environment = _RecordingEnvironment()
    toolbox = object.__new__(ToolBox)
    toolbox.environment = environment
    toolbox.tool_outputs = []
    toolbox._memory = {}
    path = 'directory/it\'s-"quoted".py'

    outputs = [
        toolbox.write_file(path, "content"),
        toolbox.edit_file(path, "before", "after"),
        toolbox.list_directory(path),
    ]

    assert outputs == ["return_code=0\nstdout=ok"] * 3
    for command, _timeout in environment.commands:
        assert path not in command
        assert "base64.b64decode" in command


def test_git_argument_lists_cannot_inject_shell_commands() -> None:
    arguments = ToolBox._shell_arguments("file.py; touch /tmp/injected")

    assert arguments == "'file.py;' touch /tmp/injected"

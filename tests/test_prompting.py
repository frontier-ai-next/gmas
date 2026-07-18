"""Unit tests for gmas.execution.runner.prompting helpers."""

from gmas.execution.runner.prompting import (
    StructuredPrompt,
    _build_assistant_parts,
    _collect_tool_parts,
    _strip_tool_metadata,
)


class TestCollectToolParts:
    def test_empty_tail(self):
        messages = [{"role": "assistant", "content": "hi"}]
        parts, j = _collect_tool_parts(messages, start=0, per_result_limit=0)
        assert parts == []
        assert j == 0

    def test_collect_and_truncate(self):
        long = "x" * 100
        messages = [
            {"role": "assistant", "content": "a"},
            {"role": "tool", "content": long},
            {"role": "tool", "content": "short"},
        ]
        parts, j = _collect_tool_parts(messages, start=1, per_result_limit=10)
        assert len(parts) == 2
        assert parts[0].endswith("\n...(truncated)")
        assert parts[1] == "short"
        assert j == 3

    def test_skips_empty_tool_content(self):
        messages = [
            {"role": "tool", "content": ""},
            {"role": "tool", "content": "ok"},
        ]
        parts, j = _collect_tool_parts(messages, 0, 0)
        assert parts == ["ok"]
        assert j == 2


class TestBuildAssistantParts:
    def test_content_only(self):
        assert _build_assistant_parts({"role": "assistant", "content": "Hello"}) == ["Hello"]

    def test_tool_calls_with_json_args(self):
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "foo", "arguments": '{"a": 1}'}},
            ],
        }
        parts = _build_assistant_parts(msg)
        assert len(parts) == 1
        assert "[Called foo(" in parts[0]
        assert "1" in parts[0]

    def test_tool_calls_invalid_json_falls_back(self):
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "bar", "arguments": "not-json"}},
            ],
        }
        parts = _build_assistant_parts(msg)
        assert parts == ["[Called bar(not-json)]"]

    def test_missing_function_name(self):
        msg = {"role": "assistant", "tool_calls": [{"function": {}}]}
        parts = _build_assistant_parts(msg)
        assert parts[0].startswith("[Called ?(")


class TestStripToolMetadata:
    def test_system_user_passthrough(self):
        inp = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ]
        out = _strip_tool_metadata(inp)
        assert out == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ]

    def test_merges_assistant_with_following_tools(self):
        inp = [
            {"role": "assistant", "content": "Thinking"},
            {"role": "tool", "content": "result"},
        ]
        out = _strip_tool_metadata(inp)
        assert len(out) == 1
        assert out[0]["role"] == "assistant"
        assert "Thinking" in out[0]["content"]
        assert "result" in out[0]["content"]

    def test_consecutive_assistant_merge(self):
        inp = [
            {"role": "assistant", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        out = _strip_tool_metadata(inp)
        assert len(out) == 1
        assert "a" in out[0]["content"]
        assert "b" in out[0]["content"]

    def test_max_total_chars_truncates_tool_results(self):
        inp = [
            {"role": "assistant", "content": "x"},
            {"role": "tool", "content": "a" * 2000},
            {"role": "tool", "content": "b" * 2000},
        ]
        out = _strip_tool_metadata(inp, max_total_chars=1000)
        assert len(out) == 1
        body = out[0]["content"]
        assert "truncated" in body

    def test_ignores_empty_system_user(self):
        inp = [
            {"role": "system", "content": ""},
            {"role": "user", "content": ""},
        ]
        assert _strip_tool_metadata(inp) == []

    def test_unknown_roles_skipped(self):
        inp = [{"role": "tool", "content": "only"}]
        assert _strip_tool_metadata(inp) == []


class TestStructuredPrompt:
    def test_str_returns_text(self):
        p = StructuredPrompt("flat", [{"role": "user", "content": "u"}])
        assert str(p) == "flat"

    def test_repr_short_text(self):
        p = StructuredPrompt("hi", [])
        assert "hi" in repr(p)
        assert "messages=0" in repr(p)

    def test_repr_long_text_truncated(self):
        long_text = "z" * 100
        p = StructuredPrompt(long_text, [{"role": "user", "content": "x"}])
        r = repr(p)
        assert "..." in r
        assert "messages=1" in r

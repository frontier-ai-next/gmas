"""Tests for shared LLM usage extraction and runner accounting."""

from unittest.mock import MagicMock

from gmas.execution.usage import (
    LLMCallResult,
    LLMUsage,
    budget_prompt_completion,
    count_llm_call,
    extract_llm_usage,
    unwrap_llm_text,
    usage_to_observability_dict,
)
from gmas.tools.llm_integration import LLMResponse, parse_openai_response


def test_extract_llm_usage_from_call_result():
    usage = LLMUsage(prompt_tokens=11, completion_tokens=7, total_tokens=18)
    result = LLMCallResult(content="hello", usage=usage)
    assert extract_llm_usage(result) == usage


def test_extract_llm_usage_from_openai_response_object():
    response = MagicMock()
    response.usage = MagicMock(prompt_tokens=20, completion_tokens=4, total_tokens=24)
    parsed = parse_openai_response(response)
    extracted = extract_llm_usage(parsed)
    assert extracted is not None
    assert extracted.total_tokens == 24
    assert extracted.prompt_tokens == 20
    assert extracted.completion_tokens == 4


def test_count_llm_call_prefers_provider_usage():
    usage = LLMUsage(prompt_tokens=30, completion_tokens=12, total_tokens=42)
    _, measured = count_llm_call(lambda _text: 999, LLMCallResult(content="x", usage=usage))
    assert measured.total_tokens == 42
    assert measured.prompt_tokens == 30
    assert measured.completion_tokens == 12


def test_count_llm_call_falls_back_to_token_counter():
    _, measured = count_llm_call(lambda text: len(text.split()), "answer", prompt_text="one two")
    assert measured.total_tokens > 0


def test_budget_prompt_completion_uses_provider_split():
    usage = LLMUsage(prompt_tokens=100, completion_tokens=25, total_tokens=125)
    prompt, completion = budget_prompt_completion(usage)
    assert (prompt, completion) == (100, 25)


def test_usage_to_observability_dict_matches_wrapper_fields():
    payload = usage_to_observability_dict(LLMUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5))
    assert payload == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
        "input_tokens": 3,
        "output_tokens": 2,
    }


def test_unwrap_llm_text_from_call_result():
    assert unwrap_llm_text(LLMCallResult(content="done")) == "done"
    assert unwrap_llm_text(LLMResponse(content="tool path")) == "tool path"

"""
Shared LLM token usage types and extraction helpers.

These utilities align MACPRunner accounting with observability wrappers
(``GMASObservabilityCallback.wrap_async_llm``) by reading the same
provider ``usage`` payload from LLM call results.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self


@dataclass
class LLMUsage:
    """Normalized token usage for a single LLM API call or aggregated step."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        if self.total_tokens == 0 and (self.prompt_tokens or self.completion_tokens):
            self.total_tokens = self.prompt_tokens + self.completion_tokens

    def add(self, other: Self | None) -> None:
        """Add another usage record into this accumulator."""
        if other is None:
            return
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        if other.total_tokens:
            self.total_tokens += other.total_tokens
        else:
            self.total_tokens += other.prompt_tokens + other.completion_tokens

    def add_estimate(self, total: int) -> None:
        """Fallback split when the provider did not return usage metadata."""
        if total <= 0:
            return
        self.prompt_tokens += total // 2
        self.completion_tokens += total - (total // 2)
        self.total_tokens += total

    def model_dump(self, mode: str = "json") -> dict[str, int]:
        """OpenAI/Pydantic-compatible dump for observability ``_safe_llm_usage``."""
        _ = mode
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LLMCallResult:
    """Canonical LLM caller result carrying content and provider usage."""

    content: Any
    usage: LLMUsage | None = None

    @property
    def text(self) -> str:
        return response_text(self.content)


def usage_from_mapping(data: dict[str, Any]) -> LLMUsage | None:
    """Build :class:`LLMUsage` from a mapping (OpenAI, Anthropic, etc.)."""
    prompt = int(data.get("prompt_tokens") or data.get("input_tokens") or 0)
    completion = int(data.get("completion_tokens") or data.get("output_tokens") or 0)
    total = int(data.get("total_tokens") or 0)
    if not any((prompt, completion, total)):
        return None
    if total == 0:
        total = prompt + completion
    return LLMUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )


def usage_from_object(obj: Any) -> LLMUsage | None:
    """Extract usage metadata from an arbitrary provider response object."""
    if obj is None:
        return None
    if isinstance(obj, LLMUsage):
        return obj
    if isinstance(obj, dict):
        nested = obj.get("usage")
        if isinstance(nested, dict):
            return usage_from_mapping(nested)
        return usage_from_mapping(obj)

    usage_attr = getattr(obj, "usage", None)
    if usage_attr is not None:
        if isinstance(usage_attr, dict):
            return usage_from_mapping(usage_attr)
        mapping = {
            "prompt_tokens": getattr(usage_attr, "prompt_tokens", None) or getattr(usage_attr, "input_tokens", 0),
            "completion_tokens": getattr(usage_attr, "completion_tokens", None)
            or getattr(usage_attr, "output_tokens", 0),
            "total_tokens": getattr(usage_attr, "total_tokens", 0),
        }
        return usage_from_mapping(mapping)

    raw_response = getattr(obj, "raw_response", None)
    if raw_response is not None and raw_response is not obj:
        nested = usage_from_object(raw_response)
        if nested is not None:
            return nested
    return None


def extract_llm_usage(result: Any) -> LLMUsage | None:
    """Extract provider usage from any supported LLM caller result shape."""
    if isinstance(result, LLMCallResult):
        if result.usage is not None:
            return result.usage
        return usage_from_object(result.content)

    direct = usage_from_object(result)
    if direct is not None:
        return direct

    content_usage = getattr(result, "usage", None)
    if content_usage is not None and isinstance(content_usage, LLMUsage):
        return content_usage

    return usage_from_object(getattr(result, "raw_response", None))


def unwrap_llm_content(result: Any) -> Any:
    """Return the inner payload from ``LLMCallResult`` wrappers."""
    if isinstance(result, LLMCallResult):
        return result.content
    return result


def response_text(content: Any) -> str:
    """Best-effort text extraction from caller content payloads."""
    if isinstance(content, str):
        return content
    text = getattr(content, "content", None)
    if isinstance(text, str):
        return text
    return ""


def unwrap_llm_text(result: Any) -> str:
    """Return plain text from any supported caller result."""
    return response_text(unwrap_llm_content(result))


def is_llm_response(content: Any) -> bool:
    """Whether *content* looks like an tool-aware LLM response object."""
    return hasattr(content, "has_tool_calls") and hasattr(content, "content")


def count_llm_call(
    token_counter: Callable[[str], int],
    result: Any,
    *,
    prompt_text: str = "",
) -> tuple[Any, LLMUsage]:
    """Measure one LLM call, preferring provider usage over local estimates."""
    content = unwrap_llm_content(result)
    usage = extract_llm_usage(result)
    accumulated = LLMUsage()
    if usage is not None:
        accumulated.add(usage)
        return content, accumulated

    response = response_text(content)
    estimated = token_counter(prompt_text) + token_counter(response)
    accumulated.add_estimate(estimated)
    return content, accumulated


def budget_prompt_completion(usage: LLMUsage) -> tuple[int, int]:
    """Return prompt/completion split, falling back to an even estimate."""
    if usage.prompt_tokens or usage.completion_tokens:
        return usage.prompt_tokens, usage.completion_tokens
    half = usage.total_tokens // 2
    return half, usage.total_tokens - half


_OBSERVABILITY_USAGE_KEYS = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }
)


def safe_llm_usage_dict(result: Any) -> dict[str, int] | None:
    """Mirror ``gmas_observability.callback._safe_llm_usage`` for parity tests."""
    if isinstance(result, dict):
        usage = result.get("usage")
    else:
        usage = getattr(result, "usage", None)
        raw = getattr(result, "raw_response", None)
        if usage is None and raw is not None:
            usage = getattr(raw, "usage", None)

    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump(mode="json")
    if isinstance(usage, dict):
        return {
            str(key): value
            for key, value in usage.items()
            if key in _OBSERVABILITY_USAGE_KEYS and isinstance(value, int) and not isinstance(value, bool)
        }
    return None


def usage_to_observability_dict(usage: LLMUsage | None) -> dict[str, int] | None:
    """Serialize usage for observability event payloads."""
    if usage is None or usage.total_tokens == 0:
        return None
    base = safe_llm_usage_dict({"usage": usage.model_dump()})
    if base is None:
        return None
    return {
        **base,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
    }

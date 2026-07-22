"""OpenAI-compatible caller used by the Terminal-Bench agents."""

import os
from dataclasses import dataclass
from typing import Any, ClassVar

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from pydantic_core import ValidationError

from benchmarks.terminal_bench.consts import (
    CONDITION_MAX_COMPLETION_TOKENS,
    LLM_MAX_COMPLETION_TOKENS,
    LLM_TEMPERATURE,
    OPENAI_MAX_RETRIES,
    PARSE_MAX_RETRIES,
    RUNNER_TIMEOUT,
)
from gmas.tools.llm_integration import LLMResponse, parse_openai_response


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """Validated connection settings for the benchmark model."""

    api_key: str
    base_url: str | None
    model: str

    @classmethod
    def from_environment(cls) -> "LLMSettings":
        """Load required settings without reading them at module import time."""
        api_key = os.getenv("LLM_API_KEY")
        model = os.getenv("LLM_MODEL")
        if not api_key or not model:
            missing = [name for name, value in (("LLM_API_KEY", api_key), ("LLM_MODEL", model)) if not value]
            names = ", ".join(missing)
            msg = f"Missing required environment variables: {names}"
            raise RuntimeError(msg)
        return cls(
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL") or None,
            model=model,
        )

    def make_async_client(self) -> AsyncOpenAI:
        """Create the asynchronous client used for agent calls."""
        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=OPENAI_MAX_RETRIES,
            timeout=RUNNER_TIMEOUT,
        )

    def make_sync_client(self) -> OpenAI:
        """Create the synchronous client used for routing decisions."""
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=OPENAI_MAX_RETRIES,
            timeout=RUNNER_TIMEOUT,
        )


def parse_structured_chat_with_retry[ResponseModelT: BaseModel](
    client: OpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
    response_format: type[ResponseModelT],
) -> ResponseModelT:
    """Parse one structured chat response, retrying validation failures."""
    last_error: ValidationError | None = None
    for _ in range(PARSE_MAX_RETRIES):
        try:
            completion = client.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_format,
                max_completion_tokens=CONDITION_MAX_COMPLETION_TOKENS,
                temperature=LLM_TEMPERATURE,
            )
        except ValidationError as exc:
            last_error = exc
            continue

        parsed = completion.choices[0].message.parsed
        if parsed is not None:
            return parsed

    if last_error is not None:
        raise last_error
    msg = "The model did not return a parsed structured response"
    raise ValueError(msg)


class RecordingAsyncCaller:
    """Structured gMAS caller that records provider-reported token usage."""

    supports_structured: ClassVar[bool] = True

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        context_stats: list[dict[str, int]],
    ) -> None:
        self._client = client
        self._model = model
        self._context_stats = context_stats

    async def __call__(
        self,
        prompt: str | list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> str | LLMResponse:
        """Call the model and retain exact token counts when available."""
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": LLM_TEMPERATURE,
            "max_completion_tokens": LLM_MAX_COMPLETION_TOKENS,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**kwargs)
        usage = response.usage
        if usage is not None and usage.prompt_tokens:
            self._context_stats.append(
                {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
            )
        else:
            word_count = sum(len(str(message.get("content") or "").split()) for message in messages)
            self._context_stats.append({"words_in_context": word_count})

        if tools:
            return parse_openai_response(response)
        return response.choices[0].message.content or ""

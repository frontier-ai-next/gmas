#!/usr/bin/env python3
r"""
GAIA benchmark: gMAS vs LangGraph on the same GAIA tasks.

This script keeps the comparison explicit:

* File tasks are included. gMAS receives its built-in ``FileSearchTool``;
  LangGraph receives LangChain's file-management read/search/list tools.
* Web variants use snippets-only DuckDuckGo/DDGS through the same DDGS backend.
* Deep variants use Playwright: gMAS through ``WebSearchTool(deep_search="playwright")``,
  LangGraph through ``PlayWrightBrowserToolkit`` created per run and closed after use.
  Deep runs are serialized with a process-wide lock (Playwright sync API is not safe when
  multiple threads drive browsers at once); ``--parallel`` still overlaps web-only work,
  but only one deep browser run runs at a time.
* The main score is strict final-answer equality after normalization. Substring
  containment is stored only as a debug signal.
* Errors stay in the denominator: ``accuracy_all = correct / total``.
* Fair run budget: symmetric outer wall-time row cap, hard total tool-call
  budget (default 5 actual tool executions), same DDG/snippets limits, and
  explicit final-answer fallback policy (default strict/no fallback).
  Infrastructure failures (recursion cap, setup, runner timeout, search infra)
  are excluded from ``error_rate`` but remain incorrect in ``accuracy_all``.
* The JSON/Markdown reports include overall, by-GAIA-level, and pairwise common-task
  deltas so framework differences are visible at level 1/2/3 instead of being hidden
  in a single aggregate.


Required environment, loaded from ``.env`` and ``benchmarks/.env`` if present:

    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

Usage examples::
    uv run python -m benchmarks.gaia.run --dry-run --max-samples 3
    uv run python -m benchmarks.gaia.run --probe-search
    uv run python -m benchmarks.gaia.run --levels 1,2,3 --max-samples 30 --parallel 4
    uv run python -m benchmarks.gaia.run --resume --checkpoint benchmark_logs/my_run.json
"""

import argparse
import asyncio
import concurrent.futures
import contextlib
import json
import multiprocessing
import os
import queue as queue_module
import random
import re
import sys
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

from gmas.builder import GraphBuilder
from gmas.execution import MACPRunner, RunnerConfig, StreamEventType
from gmas.tools import FileSearchTool, ToolCall, ToolRegistry, ToolResult, WebSearchTool, create_openai_caller
from gmas.tools.web_search import DuckDuckGoProvider

_lc_create_agent: Any = None
PlayWrightBrowserToolkit: Any = None
FileManagementToolkit: Any = None
DuckDuckGoSearchResults: Any = None
ExtractTextTool: Any = None
NavigateTool: Any = None
DuckDuckGoSearchAPIWrapper: Any = None
BaseCallbackHandler: Any = object
UsageMetadataCallbackHandler: Any = None
AIMessage: Any = None
ChatOpenAI: Any = None
async_playwright: Any = None

try:
    from langchain.agents import create_agent as _lc_create_agent
    from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
    from langchain_community.agent_toolkits.file_management import FileManagementToolkit
    from langchain_community.tools import DuckDuckGoSearchResults
    from langchain_community.tools.playwright import ExtractTextTool
    from langchain_community.tools.playwright.navigate import NavigateTool
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    from langchain_core.callbacks import BaseCallbackHandler, UsageMetadataCallbackHandler
    from langchain_core.messages import AIMessage
    from langchain_openai import ChatOpenAI
    from playwright.async_api import async_playwright
except ImportError:  # pragma: no cover - dependency check is reported at runtime
    pass


GAIA_REPO = "gaia-benchmark/GAIA"
DDGS_BACKEND = "auto"
DEFAULT_LOG_DIR = Path("benchmark_logs")
DEFAULT_CHECKPOINT_FILENAME = "gaia_search_checkpoint.json"
OUTPUT_PREVIEW_CHARS = 8000
QUESTION_PREVIEW_CHARS = 500
DEFAULT_MAX_RESULTS = 5
DEFAULT_MAX_TOKENS = 1200
DEFAULT_MAX_TOOL_ITERATIONS = 5
DEFAULT_VARIANT_TIMEOUT_S = 360.0
DEFAULT_GMAS_RUNNER_TIMEOUT_S = 900.0
DEFAULT_FINAL_FALLBACK = "none"
FINAL_FALLBACK_MODES = ("none", "langgraph", "symmetric")
REASONING_EFFORT_ENV_VARS = ("LLM_REASONING_EFFORT", "OPENAI_REASONING_EFFORT", "GMAS_REASONING_EFFORT")
LANGGRAPH_FORCED_FINAL_MAX_TOKENS = 4096
LANGGRAPH_FORCED_FINAL_RETRY_MAX_TOKENS = 8192
LANGGRAPH_RECURSION_STEPS_PER_TOOL = 14
WEB_TOOL_OUTPUT_CHARS = 6000
DEEP_MAX_CONTENT_CHARS = 6000
DEEP_MAX_FETCH_PAGES = 3
LANGGRAPH_BROWSER_TEXT_CHARS = 10000
FILE_MAX_READ_CHARS = 80_000
PROBE_QUERIES = ("capital of France", "Python programming language", "Wikipedia encyclopedia")
TEXT_FILE_EXTENSIONS = {
    ".csv",
    ".htm",
    ".html",
    ".json",
    ".jsonl",
    ".jsonld",
    ".md",
    ".pdb",
    ".tsv",
    ".txt",
    ".xml",
}

GAIA_SYSTEM_PROMPT = (
    "You are a general AI assistant. I will ask you a question. "
    "Report your thoughts, and finish your answer with the following template: "
    "FINAL ANSWER: [YOUR FINAL ANSWER]. "
    "YOUR FINAL ANSWER should be a number OR as few words as possible "
    "OR a comma separated list of numbers and/or strings. "
    "If you are asked for a number, don't use comma to write your number "
    "neither use units such as $ or percent sign unless specified otherwise. "
    "If you are asked for a string, don't use articles, neither abbreviations "
    "(e.g. for cities), and write the digits in plain text unless specified otherwise. "
    "If you are asked for a comma separated list, apply the above rules "
    "depending of whether the element to be put in the list is a number or a string."
)


def _bench_system_prompt(max_tool_calls: int) -> str:
    return (
        GAIA_SYSTEM_PROMPT + "\n\nYou have tools for web and file inspection. Use each exact tool call at most once. "
        f"For one GAIA question, keep the search compact and use at most {max_tool_calls} tool calls total. "
        "If a tool returns an error, no results, or unreadable/binary file content, do not retry "
        "the same call; make the best possible inference and finish with FINAL ANSWER. "
        "Use browser/navigation tools only for web URLs, not for local attached files."
    )


@dataclass(frozen=True)
class BenchConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    parallel: int
    variant_timeout_s: float
    gmas_runner_timeout_s: float
    max_results: int
    max_tool_iterations: int
    final_fallback: str
    reasoning_effort: str
    log_dir: Path


@dataclass(frozen=True)
class Sample:
    index: int
    task_id: str
    level: int
    question: str
    reference: str
    file_name: str
    file_path: str


@dataclass
class RunOutput:
    output: str
    elapsed_s: float
    tokens: int = 0
    calls: int = 0
    tool_calls: int = 0
    tool_budget_limit: int = 0
    tool_budget_exhausted: bool = False
    bench_note: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class Variant:
    name: str
    framework: str
    search_type: str


VARIANTS = (
    Variant("gmas_web", "gMAS", "web"),
    Variant("gmas_deep", "gMAS", "deep"),
    Variant("langgraph_web", "langgraph", "web"),
    Variant("langgraph_deep", "langgraph", "deep"),
)

# Playwright sync API is greenlet-based and not safe under concurrent threads in one process.
# Serialize all deep (browser) runs so ``--parallel > 1`` stays safe for web-only overlap.
_PLAYWRIGHT_BENCH_LOCK = threading.Lock()
_CHECKPOINT_LOCK = threading.Lock()


def _load_local_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and not os.environ.get(key):
            os.environ[key] = value.strip().strip('"').strip("'")


def _load_env() -> None:
    root = Path(__file__).resolve().parents[2]
    _load_local_env(root / ".env")
    _load_local_env(root / "benchmarks" / ".env")
    _load_local_env(Path(__file__).resolve().parent / ".env")


def _quiet_library_noise() -> None:
    with contextlib.suppress(Exception):
        stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
        stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
        if stdout_reconfigure is not None:
            stdout_reconfigure(encoding="utf-8")
        if stderr_reconfigure is not None:
            stderr_reconfigure(encoding="utf-8")
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    with contextlib.suppress(Exception):
        from datasets import disable_progress_bars

        disable_progress_bars()
    with contextlib.suppress(Exception):
        from gmas.config.logging import setup_logging

        setup_logging(level="ERROR")


def _make_cfg(args: argparse.Namespace) -> BenchConfig:
    _load_env()
    model = args.model or os.getenv("LLM_MODEL", "")
    base_url = args.base_url if args.base_url is not None else os.getenv("LLM_BASE_URL", "")
    api_key = os.getenv(args.api_key_env, "")
    final_fallback = str(args.final_fallback).strip().lower()
    if final_fallback not in FINAL_FALLBACK_MODES:
        msg = f"Unsupported --final-fallback {args.final_fallback!r}; expected one of {FINAL_FALLBACK_MODES}"
        raise SystemExit(msg)
    return BenchConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        parallel=max(1, args.parallel),
        variant_timeout_s=args.variant_timeout,
        gmas_runner_timeout_s=args.gmas_runner_timeout,
        max_results=args.max_results,
        max_tool_iterations=args.max_tool_iterations,
        final_fallback=final_fallback,
        reasoning_effort=_resolve_reasoning_effort(args.reasoning_effort, model),
        log_dir=Path(args.log_dir),
    )


def _resolve_reasoning_effort(requested: str | None, model: str) -> str:
    explicit = (requested or "").strip().lower()
    if explicit:
        return explicit
    for name in REASONING_EFFORT_ENV_VARS:
        value = os.getenv(name, "").strip().lower()
        if value:
            return value
    return "low" if "gpt-oss" in model.lower() else ""


def _openai_caller_kwargs(cfg: BenchConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if cfg.reasoning_effort:
        kwargs["reasoning_effort"] = cfg.reasoning_effort
    return kwargs


def _langchain_model_kwargs(cfg: BenchConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if cfg.reasoning_effort:
        kwargs["reasoning_effort"] = cfg.reasoning_effort
    return kwargs


def _tool_budget(cfg: BenchConfig) -> int:
    return cfg.max_tool_iterations


@dataclass
class ToolBudget:
    """Per-run hard budget for actual tool executions."""

    max_calls: int
    calls: int = 0
    exhausted: bool = False

    def claim(self, tool_name: str) -> tuple[bool, str | None]:
        if self.calls >= self.max_calls:
            self.exhausted = True
            return False, self.exhaustion_message(tool_name)
        self.calls += 1
        return True, None

    def exhaustion_message(self, tool_name: str) -> str:
        return (
            f"Tool budget exhausted before {tool_name!r}. "
            f"Limit: {self.max_calls} total tool call(s). "
            "Use the observations already available and provide FINAL ANSWER now."
        )

    def as_diagnostics(self) -> dict[str, Any]:
        return {"max_calls": self.max_calls, "calls": self.calls, "exhausted": self.exhausted}


class BudgetedToolRegistry(ToolRegistry):
    """ToolRegistry that enforces a per-run total tool-call budget."""

    def __init__(self, budget: ToolBudget) -> None:
        super().__init__()
        self.budget = budget

    def execute(self, call: ToolCall) -> ToolResult:
        allowed, message = self.budget.claim(call.name)
        if not allowed:
            return ToolResult(tool_name=call.name, success=True, output=message or "Tool budget exhausted.")
        return super().execute(call)


def _should_force_final(cfg: BenchConfig, variant: Variant) -> bool:
    if cfg.final_fallback == "none":
        return False
    if cfg.final_fallback == "symmetric":
        return True
    return cfg.final_fallback == "langgraph" and variant.framework == "langgraph"


def _outer_timeout_sec(cfg: BenchConfig, variant: Variant) -> float | None:
    """Symmetric benchmark-level wall timeout for every framework variant."""
    _ = variant
    if cfg.variant_timeout_s <= 0:
        return None
    return cfg.variant_timeout_s


def _gmas_runner_timeout_sec(cfg: BenchConfig) -> float:
    return cfg.gmas_runner_timeout_s


def _require_llm_config(cfg: BenchConfig) -> None:
    missing = []
    if not cfg.api_key:
        missing.append("LLM_API_KEY")
    if not cfg.model:
        missing.append("LLM_MODEL")
    if missing:
        msg = f"Missing required LLM config: {', '.join(missing)}"
        raise SystemExit(msg)


def _ddg_provider(timeout: int = 15) -> DuckDuckGoProvider:
    return DuckDuckGoProvider(timeout=timeout, ddgs_backend=DDGS_BACKEND)


def _truncate_tool_text(text: str, limit: int, label: str) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return (
        text[:limit].rstrip() + f"\n\n[benchmark truncated {omitted} chars from {label}; "
        "make a narrower follow-up query if needed.]"
    )


def _cap_lc_tool_output(output: Any, tool_name: str, max_output_chars: int | None) -> str:
    text = str(output)
    if max_output_chars is None:
        return text
    return _truncate_tool_text(text, max_output_chars, f"LangGraph {tool_name}")


def _langgraph_navigation_failure_message(exc: Exception) -> str | None:
    text = str(exc)
    if "Download is starting" in text:
        return (
            "Navigation started a file download instead of an HTML page. "
            "Treat this as an unreadable browser page; use web_search snippets or another source, "
            "then provide the best possible FINAL ANSWER."
        )
    if "Page.goto" not in text and "net::" not in text:
        return None
    return (
        f"Navigation failed: {_preview_text(text, 500)}. "
        "Treat this as an unavailable browser page; use web_search snippets or another source, "
        "then provide the best possible FINAL ANSWER."
    )


class SearchOnlyWebSearchTool(WebSearchTool):
    """Benchmark web mode: snippets-only search, no URL fetch/page-reading actions."""

    @property
    def description(self) -> str:
        return (
            "Search the public web and return compact snippets only. "
            "Use a short query string. URL fetch, page reading, browser actions, "
            "and fetch_content are intentionally disabled for the web-only benchmark."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Short web search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": f"Maximum number of snippets, capped at {self._max_results}.",
                },
            },
            "required": ["query"],
        }

    def execute(  # noqa: PLR0913
        self,
        query: str = "",
        url: str = "",
        *,
        action: str = "",
        fetch_content: bool | None = None,
        max_results: int | None = None,
        wait_for_selector: str | None = None,
        selector: str = "",
        value: str = "",
        submit: bool = False,
        js_code: str = "",
        max_depth: int = 2,
        max_pages: int = 10,
        max_links: int = 50,
        url_filter: str | None = None,
        tab_index: int | None = None,
        background: bool = False,
        path: str = "",
        full_page: bool = False,
        urls: list[str] | None = None,
        cookies: list[dict[str, Any]] | None = None,
        limit: int = 100,
        clear: bool = False,
        trace_screenshots: bool = True,
        trace_snapshots: bool = True,
        trace_sources: bool = True,
        provider: str | None = None,
        intent: str | None = None,
        wait_timeout: int | None = None,
        no_cache: bool = False,
        deduplicate: bool | None = None,
        **_kwargs: Any,
    ) -> ToolResult:
        del (
            url,
            action,
            fetch_content,
            wait_for_selector,
            selector,
            value,
            submit,
            js_code,
            max_depth,
            max_pages,
            max_links,
            url_filter,
            tab_index,
            background,
            path,
            full_page,
            urls,
            cookies,
            limit,
            clear,
            trace_screenshots,
            trace_snapshots,
            trace_sources,
            provider,
            intent,
            wait_timeout,
            no_cache,
            deduplicate,
        )
        bounded_results = self._max_results
        if max_results is not None:
            with contextlib.suppress(TypeError, ValueError):
                bounded_results = max(1, min(int(max_results), self._max_results))
        return super().execute(
            query=(query or "").strip(),
            action="search",
            fetch_content=False,
            max_results=bounded_results,
        )


def _make_web_tool(max_results: int = DEFAULT_MAX_RESULTS) -> WebSearchTool:
    return SearchOnlyWebSearchTool(
        provider=_ddg_provider(),
        max_results=max_results,
        max_content_length=WEB_TOOL_OUTPUT_CHARS,
        fetch_content=False,
        max_fetch_pages=0,
        timeout=15,
        cache=True,
        cache_ttl=3600,
        cache_max_entries=2048,
    )


def _make_deep_tool(max_results: int = DEFAULT_MAX_RESULTS) -> WebSearchTool:
    tool = WebSearchTool(
        provider=_ddg_provider(),
        max_results=max_results,
        max_content_length=DEEP_MAX_CONTENT_CHARS,
        fetch_content=True,
        max_fetch_pages=DEEP_MAX_FETCH_PAGES,
        timeout=25,
        cache=True,
        cache_ttl=3600,
        cache_max_entries=2048,
        deep_search="playwright",
        browser_config={
            "headless": True,
            "browser": "chromium",
            "scroll_to_bottom": True,
            "max_scrolls": 3,
            "disable_images": True,
        },
    )
    tool.warm_up()
    return tool


class SnapshotFileSearchTool(FileSearchTool):
    """FileSearchTool variant that accepts HuggingFace snapshot symlink paths."""

    def __init__(self, base_directory: str | Path, **kwargs: Any) -> None:
        self._raw_base_directory = Path(base_directory).absolute()
        super().__init__(base_directory=base_directory, **kwargs)
        self._base_directory = self._raw_base_directory

    def _is_path_safe(self, path: Path) -> bool:
        try:
            absolute = path.absolute()
            if absolute.is_relative_to(self._raw_base_directory):
                return True
        except (ValueError, OSError):
            pass
        return super()._is_path_safe(path)


def _make_file_tool(file_root: str) -> FileSearchTool:
    return SnapshotFileSearchTool(base_directory=file_root, max_results=20, max_read_size=FILE_MAX_READ_CHARS)


def _execute_web_search(tool: WebSearchTool, arguments: dict[str, Any] | str) -> str:
    result = tool.execute(**arguments) if isinstance(arguments, dict) else tool.execute(query=arguments)
    if result and result.success:
        return result.output or "No results found."
    return f"Error: {getattr(result, 'error', None) or 'No results found.'}"


def _make_lc_ddg_tool(max_results: int = 10) -> Any:
    if DuckDuckGoSearchAPIWrapper is None or DuckDuckGoSearchResults is None:
        msg = "langchain-community DuckDuckGo search is not installed"
        raise RuntimeError(msg)
    wrapper = DuckDuckGoSearchAPIWrapper(
        backend=DDGS_BACKEND,
        region="wt-wt",
        max_results=max_results,
    )
    return DuckDuckGoSearchResults(
        api_wrapper=wrapper,
        max_results=max_results,
        handle_tool_error=True,
        handle_validation_error=True,
    )


def _lc_invoke_ddg_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> str:
    try:
        search_tool = _make_lc_ddg_tool(max_results)
    except Exception as exc:
        return f"Error: {exc}"
    try:
        result = str(search_tool.invoke((query or "").strip()))
        return _truncate_tool_text(result, WEB_TOOL_OUTPUT_CHARS, "LangGraph web_search")
    except Exception as exc:
        return f"Error: {exc}"


def _make_lc_web_tool(max_results: int, budget: ToolBudget | None = None) -> Any:
    from langchain_core.tools import tool

    search_tool = _make_lc_ddg_tool(max_results)

    @tool
    def web_search(query: str) -> str:
        """Search the public web through DuckDuckGo/DDGS using the configured backend."""
        if budget is not None:
            allowed, message = budget.claim("web_search")
            if not allowed:
                return message or "Tool budget exhausted."
        try:
            result = str(search_tool.invoke((query or "").strip()))
            if result.strip():
                return _truncate_tool_text(result, WEB_TOOL_OUTPUT_CHARS, "LangGraph web_search")
        except Exception as exc:
            return f"No web search results found ({exc}). Continue without repeating this exact search."
        return "No web search results found. Continue without repeating this exact search."

    return web_search


def _budget_lc_tool(tool_obj: Any, budget: ToolBudget | None, max_output_chars: int | None = None) -> Any:
    if budget is None:
        return tool_obj
    try:
        from langchain_core.tools import BaseTool
        from pydantic import ConfigDict, PrivateAttr
    except ImportError:
        return tool_obj

    class BudgetedLangChainTool(BaseTool):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        _inner: Any = PrivateAttr()
        _budget: ToolBudget = PrivateAttr()
        _max_output_chars: int | None = PrivateAttr()

        def __init__(self, inner: Any, run_budget: ToolBudget, output_cap: int | None) -> None:
            kwargs: dict[str, Any] = {
                "name": getattr(inner, "name", type(inner).__name__),
                "description": getattr(inner, "description", "Budgeted benchmark tool"),
            }
            args_schema = getattr(inner, "args_schema", None)
            if args_schema is not None:
                kwargs["args_schema"] = args_schema
            super().__init__(**kwargs)
            self._inner = inner
            self._budget = run_budget
            self._max_output_chars = output_cap

        def _run(self, *args: Any, **kwargs: Any) -> str:
            allowed, message = self._budget.claim(self.name)
            if not allowed:
                return message or "Tool budget exhausted."
            payload: Any = args[0] if args and not kwargs else kwargs
            if hasattr(self._inner, "invoke"):
                return _cap_lc_tool_output(self._inner.invoke(payload), self.name, self._max_output_chars)
            if hasattr(self._inner, "_run"):
                return _cap_lc_tool_output(self._inner._run(*args, **kwargs), self.name, self._max_output_chars)
            msg = f"Wrapped tool {self.name!r} is not invokable"
            raise RuntimeError(msg)

        async def _arun(self, *args: Any, **kwargs: Any) -> str:
            allowed, message = self._budget.claim(self.name)
            if not allowed:
                return message or "Tool budget exhausted."
            payload: Any = args[0] if args and not kwargs else kwargs
            if hasattr(self._inner, "ainvoke"):
                return _cap_lc_tool_output(await self._inner.ainvoke(payload), self.name, self._max_output_chars)
            if hasattr(self._inner, "_arun"):
                return _cap_lc_tool_output(await self._inner._arun(*args, **kwargs), self.name, self._max_output_chars)
            return self._run(*args, **kwargs)

    return BudgetedLangChainTool(tool_obj, budget, max_output_chars)


def _budget_lc_tools(tools: list[Any], budget: ToolBudget | None, max_output_chars: int | None = None) -> list[Any]:
    return [_budget_lc_tool(tool_obj, budget, max_output_chars=max_output_chars) for tool_obj in tools]


def _lg_dispatch(name: str, arguments: dict[str, Any], *, deep: bool = False) -> str:
    """Compatibility helper for the old smoke script."""
    if name == "web_search":
        return _lc_invoke_ddg_search(str(arguments.get("query", "")))
    return f"Error: unsupported tool {name!r}; deep={deep} is not benchmarked here"


def _download_gaia_snapshot() -> Path | None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None
    try:
        return Path(snapshot_download(repo_id=GAIA_REPO, repo_type="dataset"))
    except Exception:
        return None


def _resolve_gaia_file(snapshot_root: Path | None, file_name: str) -> str:
    if not file_name:
        return ""
    direct = Path(file_name)
    if direct.is_file():
        return str(direct.absolute())
    if snapshot_root is None:
        return ""
    for candidate in (
        snapshot_root / file_name,
        snapshot_root / "2023" / "validation" / file_name,
        snapshot_root / "2023" / "test" / file_name,
    ):
        if candidate.is_file():
            # Keep the snapshot path (with its original filename/suffix) instead
            # of resolving through HuggingFace's symlink into blobs/. The blob
            # target is suffixless, which makes text attachments look like
            # unknown binary files and makes relative file-tool use harder.
            return str(candidate.absolute())
    return ""


def _load_gaia_samples(levels: list[int], max_samples: int | None, seed: int) -> tuple[list[Sample], int]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        msg = "Install benchmark deps first: uv sync --extra benchmarks"
        raise SystemExit(msg) from exc

    samples: list[Sample] = []
    next_idx = 0
    snapshot_root = _download_gaia_snapshot()

    for level in sorted(set(levels)):
        ds = load_dataset(GAIA_REPO, f"2023_level{level}", split="validation")
        for row_idx, item in enumerate(ds):
            question = str(item.get("Question", item.get("question", "")) or "").strip()
            reference = str(item.get("Final answer", item.get("final_answer", item.get("answer", ""))) or "").strip()
            file_name = str(item.get("file_name", item.get("file_path", "")) or "").strip()
            if not question or not reference:
                continue
            file_path = _resolve_gaia_file(snapshot_root, file_name)
            samples.append(
                Sample(
                    index=next_idx,
                    task_id=str(item.get("task_id", f"gaia_L{level}_{row_idx}")),
                    level=level,
                    question=question,
                    reference=reference,
                    file_name=file_name,
                    file_path=file_path,
                ),
            )
            next_idx += 1

    if max_samples is not None and len(samples) > max_samples:
        samples = _stratified_subsample(samples, max_samples, seed=seed)

    return samples, sum(1 for sample in samples if sample.file_name)


def _stratified_subsample(samples: list[Sample], max_n: int, seed: int) -> list[Sample]:
    rng = random.Random(seed)
    by_level: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_level[sample.level].append(sample)
    for bucket in by_level.values():
        rng.shuffle(bucket)

    out: list[Sample] = []
    positions = dict.fromkeys(by_level, 0)
    while len(out) < max_n:
        progressed = False
        for level in sorted(by_level):
            pos = positions[level]
            if pos < len(by_level[level]):
                out.append(by_level[level][pos])
                positions[level] += 1
                progressed = True
                if len(out) == max_n:
                    break
        if not progressed:
            break

    return [
        Sample(
            index=i,
            task_id=s.task_id,
            level=s.level,
            question=s.question,
            reference=s.reference,
            file_name=s.file_name,
            file_path=s.file_path,
        )
        for i, s in enumerate(out)
    ]


def _normalize_answer(text: str | None) -> str:
    value = (text or "").strip()
    value = value.strip("`'\" ")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s*,\s*", ", ", value)
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = value.rstrip(".")
    return value.lower()


def _format_question(sample: Sample) -> str:
    if sample.file_path:
        if not _file_tools_enabled(sample):
            return (
                f"{sample.question}\n\n"
                f"[Attached file on disk: {sample.file_path}]\n"
                "This benchmark run registers file-reading tools only for text-like attachments. "
                "This attachment type is not registered with a parser, so make the best possible inference."
            )
        return (
            f"{sample.question}\n\n"
            f"[Attached file on disk: {sample.file_path}]\n"
            "Use the available file tool to inspect the attachment when needed. "
            f"The file tool is rooted at the attachment directory; read `{Path(sample.file_path).name}` directly."
        )
    if sample.file_name:
        return f"{sample.question}\n\n[Associated GAIA file name: {sample.file_name}; local path was not resolved.]"
    return sample.question


def _extract_final_answer(output: str | None) -> str:
    if not output:
        return ""
    text = output.strip()
    if _looks_like_tool_transcript(text):
        return ""
    patterns = (
        r"(?:^|\n)\s*FINAL\s+ANSWER\s*:\s*(.+?)(?:\n\s*\n|\Z)",
        r"(?:^|\n)\s*Final\s+Answer\s*:\s*(.+?)(?:\n\s*\n|\Z)",
        r"\\boxed\{([^}]+)\}",
        r"(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*(.+?)(?:\n\s*\n|\Z)",
    )
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL))
        if matches:
            return matches[-1].group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in reversed(lines):
        if _looks_like_tool_transcript(line):
            continue
        if re.match(r"^(FINAL\s+ANSWER|Final\s+Answer)\s*:", line, flags=re.IGNORECASE):
            continue
        if len(line) <= 200 and not line.endswith(":"):
            return line
    return lines[-1] if lines else ""


def _looks_like_tool_transcript(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return bool(
        re.match(r"^\[(?:Called\s+)?[A-Za-z_][\w-]*\s*\(", stripped)
        or re.match(r"^\[[A-Za-z_][\w-]*\s*\(\{", stripped)
        or re.match(r"^\[[A-Za-z_][\w-]*\]\s*:", stripped)
    )


def _strip_articles(text: str) -> str:
    return re.sub(r"^(?:the|a|an)\s+", "", text.strip(), flags=re.IGNORECASE)


def _normalize_list_answer(text: str) -> str:
    parts = [_normalize_answer(part) for part in text.split(",") if part.strip()]
    return ", ".join(sorted(parts))


def _numeric_equal(left: str, right: str) -> bool:
    numeric = re.compile(r"^[+-]?(?:\d[\d,]*)(?:\.\d+)?(?:/\d+)?$")
    left_clean = re.sub(r"[$%]", "", left.replace(",", "").strip())
    right_clean = re.sub(r"[$%]", "", right.replace(",", "").strip())
    if not numeric.match(left_clean) or not numeric.match(right_clean):
        return False
    try:
        return Fraction(left_clean) == Fraction(right_clean)
    except (ArithmeticError, ValueError, ZeroDivisionError):
        return False


def _answers_equal(pred_n: str, ref_n: str) -> bool:
    if not pred_n or not ref_n:
        return False
    if pred_n == ref_n or _numeric_equal(pred_n, ref_n):
        return True
    if _strip_articles(pred_n) == _strip_articles(ref_n):
        return True
    if "," in pred_n or "," in ref_n:
        return _normalize_list_answer(pred_n) == _normalize_list_answer(ref_n)
    return False


def score_gaia(output: str | None, reference: str) -> tuple[bool, str, bool]:
    prediction = _extract_final_answer(output)
    pred_n = _normalize_answer(prediction)
    ref_n = _normalize_answer(reference)
    strict = _answers_equal(pred_n, ref_n)
    contains_reference = bool(ref_n and ref_n in _normalize_answer(output or ""))
    return strict, prediction, contains_reference


def _classify_error_message(msg: str, variant: Variant) -> tuple[str, bool]:
    """Return (error_kind, exclude_from_error_rate)."""
    lower = msg.lower()
    if "recursion limit" in lower or "graphrecursionerror" in lower:
        return "recursion_cap", True
    if "maximum context length" in lower or "context length is" in lower:
        return "context_limit", True
    if "not installed" in lower or "install benchmark" in lower or "install playwright" in lower:
        return "setup", True
    if "variant wall timeout" in lower or "langgraph wall timeout" in lower:
        return "wall_timeout", True
    if variant.framework == "gMAS" and ("timeout" in lower or "timed out" in lower):
        return "runner_timeout", True
    if "all search providers failed" in lower or "network_error" in lower:
        return "search_infra", True
    if "download is starting" in lower and variant.framework == "langgraph":
        return "browser_download", True
    return "agent", False


def _error_counts_in_rate(row: dict[str, Any]) -> bool:
    return bool(row.get("error")) and not row.get("error_excluded")


def _make_openai_caller(cfg: BenchConfig) -> Any:
    return create_openai_caller(
        api_key=cfg.api_key,
        base_url=cfg.base_url or None,
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        tool_choice="auto",
        **_openai_caller_kwargs(cfg),
    )


class _CountingCaller:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls = 0
        self.call_summaries: list[dict[str, Any]] = []

    def __call__(self, prompt: str | list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> Any:
        self.calls += 1
        response = self._inner(prompt) if tools is None else self._inner(prompt, tools=tools)
        self.call_summaries.append(_summarize_llm_call(self.calls, prompt, tools, response))
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _event_text(event: Any) -> str:
    content = getattr(event, "content", "")
    if isinstance(content, str):
        return content
    if hasattr(content, "text"):
        return str(content.text)
    return str(content or "")


def _preview_text(value: Any, limit: int = 500) -> str:
    text = str(value or "")
    return text[:limit] + ("...(truncated)" if len(text) > limit else "")


def _summarize_prompt(prompt: str | list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(prompt, list):
        return {
            "kind": "messages",
            "n_messages": len(prompt),
            "messages": [
                {
                    "role": message.get("role"),
                    "content_len": len(str(message.get("content") or "")),
                    "content_preview": _preview_text(message.get("content"), 240),
                    "has_tool_calls": bool(message.get("tool_calls")),
                }
                for message in prompt[-8:]
            ],
        }
    return {"kind": "text", "content_len": len(prompt), "content_preview": _preview_text(prompt, 500)}


def _summarize_llm_call(
    call_idx: int,
    prompt: str | list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    response: Any,
) -> dict[str, Any]:
    raw = getattr(response, "raw_response", None)
    choice = None
    message = None
    with contextlib.suppress(Exception):
        if raw is None:
            raise AttributeError
        choice = raw.choices[0]
        message = choice.message
    tool_calls = getattr(response, "tool_calls", []) or []
    content = response if isinstance(response, str) else getattr(response, "content", "")
    return {
        "call": call_idx,
        "tools_enabled": bool(tools),
        "tool_names": [schema.get("function", {}).get("name") for schema in tools or []],
        "prompt": _summarize_prompt(prompt),
        "response_type": type(response).__name__,
        "finish_reason": getattr(choice, "finish_reason", None),
        "content_len": len(str(content or "")),
        "content_preview": _preview_text(content, 500),
        "tool_calls": [
            {"id": getattr(tc, "id", ""), "name": getattr(tc, "name", ""), "arguments": getattr(tc, "arguments", {})}
            for tc in tool_calls
        ],
        "raw_message_content_len": len(str(getattr(message, "content", "") or "")) if message is not None else None,
    }


def _summarize_stream_event(event: Any) -> dict[str, Any]:
    event_type = str(getattr(event, "event_type", ""))
    summary: dict[str, Any] = {"event_type": event_type}
    if event_type == StreamEventType.AGENT_OUTPUT:
        content = _event_text(event)
        summary.update(
            {
                "agent_id": getattr(event, "agent_id", ""),
                "is_final": bool(getattr(event, "is_final", False)),
                "tokens_used": int(getattr(event, "tokens_used", 0) or 0),
                "content_len": len(content),
                "content_preview": _preview_text(content, 500),
            }
        )
    elif event_type == StreamEventType.RUN_END:
        final_answer = str(getattr(event, "final_answer", "") or "")
        summary.update(
            {
                "success": bool(getattr(event, "success", False)),
                "final_agent_id": getattr(event, "final_agent_id", ""),
                "final_answer_len": len(final_answer),
                "final_answer_preview": _preview_text(final_answer, 500),
                "errors": list(getattr(event, "errors", []) or []),
            }
        )
    elif event_type == StreamEventType.AGENT_ERROR:
        summary.update(
            {
                "agent_id": getattr(event, "agent_id", ""),
                "error_type": getattr(event, "error_type", ""),
                "error_message": _preview_text(getattr(event, "error_message", ""), 500),
            }
        )
    return summary


def _file_root_for(sample: Sample) -> str:
    if sample.file_path:
        return str(Path(sample.file_path).absolute().parent)
    return str(Path.cwd())


def _file_tools_enabled(sample: Sample) -> bool:
    if not sample.file_path:
        return False
    suffix = Path(sample.file_path).suffix.lower() or Path(sample.file_name).suffix.lower()
    return suffix in TEXT_FILE_EXTENSIONS


def run_gmas(sample: Sample, cfg: BenchConfig, *, deep: bool) -> RunOutput:
    started = time.perf_counter()
    browser_deep = deep and not sample.file_path
    web_tool = _make_deep_tool(cfg.max_results) if browser_deep else _make_web_tool(cfg.max_results)
    tool_budget = ToolBudget(_tool_budget(cfg))
    registry = BudgetedToolRegistry(tool_budget)
    registry.register(web_tool)
    tool_names = ["web_search"]
    if _file_tools_enabled(sample):
        registry.register(_make_file_tool(_file_root_for(sample)))
        tool_names.append("file_search")
    caller = _CountingCaller(_make_openai_caller(cfg))
    try:
        builder = GraphBuilder()
        builder.add_agent(
            "researcher",
            "Researcher",
            _bench_system_prompt(_tool_budget(cfg)),
            "Use web_search for external facts and file_search to inspect attached GAIA files.",
            tools=tool_names,
        )
        builder.add_task(query=_format_question(sample), description="GAIA research task")
        builder.connect_task_to_agents(agent_ids=["researcher"])

        runner = MACPRunner(
            llm_caller=caller,
            config=RunnerConfig(
                max_tool_iterations=_tool_budget(cfg),
                timeout=_gmas_runner_timeout_sec(cfg),
                tool_registry=registry,
            ),
        )

        output = ""
        tokens = 0
        stream_events: list[dict[str, Any]] = []
        for event in runner.stream(builder.build()):
            stream_events.append(_summarize_stream_event(event))
            if getattr(event, "event_type", None) == StreamEventType.AGENT_OUTPUT:
                output = _event_text(event) or output
                tokens = int(getattr(event, "tokens_used", tokens) or tokens)
            elif getattr(event, "event_type", None) == StreamEventType.RUN_END:
                output = str(getattr(event, "final_answer", "") or output)
                tokens = int(getattr(event, "total_tokens", tokens) or tokens)

        diagnostics = {
            "stream_events": stream_events,
            "llm_calls": caller.call_summaries,
            "tool_names": tool_names,
            "tool_budget": tool_budget.as_diagnostics(),
            "forced_final_policy": cfg.final_fallback,
            "forced_final_used": False,
            "forced_final_error": None,
            "output_empty": not bool(output.strip()),
            "output_looks_like_tool_transcript": _looks_like_tool_transcript(output),
        }
        variant = Variant("gmas_deep" if deep else "gmas_web", "gMAS", "deep" if deep else "web")
        if _should_force_final(cfg, variant) and not output.strip():
            try:
                forced_output = _force_gmas_final_answer(sample, stream_events, cfg, diagnostics)
                if forced_output.strip():
                    output = forced_output
                    diagnostics["forced_final_used"] = True
            except Exception as exc:  # pragma: no cover - provider/runtime dependent diagnostic path
                diagnostics["forced_final_error"] = f"{type(exc).__name__}: {exc}"
        diagnostics["output_empty"] = not bool(output.strip())
        diagnostics["output_looks_like_tool_transcript"] = _looks_like_tool_transcript(output)
        forced_attempt_rows = diagnostics.get("forced_final_attempts", [])
        forced_attempts = len(forced_attempt_rows) if isinstance(forced_attempt_rows, list) else 0
        return RunOutput(
            output=output,
            elapsed_s=time.perf_counter() - started,
            tokens=tokens,
            calls=caller.calls + forced_attempts,
            tool_calls=tool_budget.calls,
            tool_budget_limit=tool_budget.max_calls,
            tool_budget_exhausted=tool_budget.exhausted,
            diagnostics=diagnostics,
        )
    finally:
        with contextlib.suppress(Exception):
            web_tool.close()


def run_gmas_web(sample: Sample, cfg: BenchConfig) -> RunOutput:
    return run_gmas(sample, cfg, deep=False)


def run_gmas_deep(sample: Sample, cfg: BenchConfig) -> RunOutput:
    return run_gmas(sample, cfg, deep=True)


class _LLMCallCounter(cast("Any", BaseCallbackHandler)):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def on_llm_end(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        self.calls += 1


def _sum_langchain_tokens(handler: Any) -> int:
    if handler is None:
        return 0
    total = 0
    for meta in getattr(handler, "usage_metadata", {}).values():
        total += int(meta.get("total_tokens", 0) or 0)
    return total


def _stringify_lc_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(block.get("text", block)) if isinstance(block, dict) else str(block) for block in content)
    return str(content or "")


def _last_ai_message_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if AIMessage is not None and isinstance(message, AIMessage):
            text = _stringify_lc_content(message.content).strip()
            if text:
                return text
    return ""


def _summarize_lc_messages(messages: list[Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for message in messages[-12:]:
        content = _stringify_lc_content(getattr(message, "content", ""))
        response_metadata = getattr(message, "response_metadata", {}) or {}
        usage_metadata = getattr(message, "usage_metadata", {}) or {}
        summaries.append(
            {
                "type": type(message).__name__,
                "content_len": len(content),
                "content_preview": _preview_text(content, 500),
                "tool_calls": [
                    {"name": call.get("name"), "args": call.get("args")}
                    for call in getattr(message, "tool_calls", []) or []
                    if isinstance(call, dict)
                ],
                "name": getattr(message, "name", None),
                "finish_reason": response_metadata.get("finish_reason"),
                "usage_metadata": usage_metadata,
            }
        )
    return summaries


def _lc_transcript_for_final_answer(messages: list[Any]) -> str:
    """Compact LangChain messages into a plain transcript for the forced-final fallback."""
    chunks: list[str] = []
    for message in messages[-14:]:
        content = _stringify_lc_content(getattr(message, "content", "")).strip()
        tool_calls = getattr(message, "tool_calls", []) or []
        msg_type = type(message).__name__
        if content:
            chunks.append(f"{msg_type}: {_preview_text(content, 2500)}")
        if tool_calls:
            compact_calls = [
                {"name": call.get("name"), "args": call.get("args")} for call in tool_calls if isinstance(call, dict)
            ]
            if compact_calls:
                chunks.append(f"{msg_type} tool_calls: {compact_calls}")
    return "\n\n".join(chunks)


def _summarize_lc_response(response: Any, *, max_tokens: int) -> dict[str, Any]:
    content = _stringify_lc_content(getattr(response, "content", ""))
    response_metadata = getattr(response, "response_metadata", {}) or {}
    usage_metadata = getattr(response, "usage_metadata", {}) or {}
    return {
        "max_tokens": max_tokens,
        "type": type(response).__name__,
        "content_len": len(content),
        "content_preview": _preview_text(content, 500),
        "finish_reason": response_metadata.get("finish_reason"),
        "usage_metadata": usage_metadata,
        "response_metadata": response_metadata,
    }


def _fallback_token_budgets(cfg: BenchConfig) -> list[int]:
    if cfg.final_fallback == "langgraph":
        return [
            max(cfg.max_tokens, LANGGRAPH_FORCED_FINAL_MAX_TOKENS),
            max(cfg.max_tokens, LANGGRAPH_FORCED_FINAL_RETRY_MAX_TOKENS),
        ]
    return [cfg.max_tokens]


def _force_langgraph_final_answer(
    model: Any,
    sample: Sample,
    messages: list[Any],
    cfg: BenchConfig,
    diagnostics: dict[str, Any] | None = None,
) -> str:
    """Ask the LLM for a final text answer when an agent terminates with empty content."""
    transcript = _lc_transcript_for_final_answer(messages)
    if not transcript.strip():
        transcript = "No tool transcript was captured. Answer from the original question alone."
    prompt = (
        "Agent stopped without a textual final answer. Use the original GAIA question and the "
        "tool observations below to give the best possible final response now. Do not call tools. "
        "Do not include hidden reasoning or tool transcript markers. Finish with exactly this template: "
        "FINAL ANSWER: [YOUR FINAL ANSWER].\n\n"
        f"Original question:\n{_format_question(sample)}\n\n"
        f"Conversation/tool transcript:\n{transcript}"
    )
    attempts: list[dict[str, Any]] = []
    budgets = _fallback_token_budgets(cfg)
    last_output = ""

    for attempt_idx, final_max_tokens in enumerate(dict.fromkeys(budgets), start=1):
        final_model = model
        if final_max_tokens != cfg.max_tokens:
            with contextlib.suppress(Exception):
                final_model = model.bind(max_tokens=final_max_tokens)
        response = final_model.invoke(
            [
                {"role": "system", "content": GAIA_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            config={"recursion_limit": _lc_recursion_limit(cfg, deep=False)},
        )
        response_summary = _summarize_lc_response(response, max_tokens=final_max_tokens)
        response_summary["attempt"] = attempt_idx
        attempts.append(response_summary)
        last_output = _stringify_lc_content(getattr(response, "content", "")).strip()
        if last_output:
            break
        if response_summary.get("finish_reason") != "length":
            break
    if diagnostics is not None:
        diagnostics["forced_final_max_tokens"] = attempts[-1]["max_tokens"] if attempts else None
        diagnostics["forced_final_response"] = attempts[-1] if attempts else None
        diagnostics["forced_final_attempts"] = attempts
    return last_output


def _force_gmas_final_answer(
    sample: Sample,
    stream_events: list[dict[str, Any]],
    cfg: BenchConfig,
    diagnostics: dict[str, Any] | None = None,
) -> str:
    transcript = json.dumps(stream_events[-14:], ensure_ascii=False, default=str)
    if not transcript.strip() or transcript == "[]":
        transcript = "No stream events were captured. Answer from the original question alone."
    prompt = (
        "gMAS stopped without a textual final answer. Use the original GAIA question and the "
        "stream/tool observations below to give the best possible final response now. Do not call tools. "
        "Do not include hidden reasoning or tool transcript markers. Finish with exactly this template: "
        "FINAL ANSWER: [YOUR FINAL ANSWER].\n\n"
        f"Original question:\n{_format_question(sample)}\n\n"
        f"gMAS stream transcript:\n{transcript}"
    )
    attempts: list[dict[str, Any]] = []
    last_output = ""
    for attempt_idx, final_max_tokens in enumerate(dict.fromkeys(_fallback_token_budgets(cfg)), start=1):
        caller = create_openai_caller(
            api_key=cfg.api_key,
            base_url=cfg.base_url or None,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=final_max_tokens,
            system_prompt=GAIA_SYSTEM_PROMPT,
            **_openai_caller_kwargs(cfg),
        )
        last_output = str(caller(prompt) or "").strip()
        attempts.append(
            {
                "attempt": attempt_idx,
                "max_tokens": final_max_tokens,
                "content_len": len(last_output),
                "content_preview": _preview_text(last_output, 500),
            }
        )
        if last_output:
            break
    if diagnostics is not None:
        diagnostics["forced_final_max_tokens"] = attempts[-1]["max_tokens"] if attempts else None
        diagnostics["forced_final_response"] = attempts[-1] if attempts else None
        diagnostics["forced_final_attempts"] = attempts
    return last_output


def _make_lc_file_tools(file_root: str) -> list[Any]:
    if FileManagementToolkit is None:
        msg = "langchain-community FileManagementToolkit is not installed"
        raise RuntimeError(msg)
    tools = FileManagementToolkit(
        root_dir=file_root,
        selected_tools=["read_file", "file_search", "list_directory"],
    ).get_tools()
    for tool in tools:
        with contextlib.suppress(Exception):
            tool.handle_tool_error = True
            tool.handle_validation_error = True
    return tools


def _lc_recursion_limit(cfg: BenchConfig, *, deep: bool) -> int:
    """High cap so LangGraph finishes tool loops instead of aborting at ~15–40 steps."""
    budget = _tool_budget(cfg)
    extra = 16 if deep else 12
    return max(72, budget * LANGGRAPH_RECURSION_STEPS_PER_TOOL + extra)


def _invoke_lc_agent(agent: Any, payload: dict[str, Any], cfg: BenchConfig, *, deep: bool) -> dict[str, Any]:
    last_state: dict[str, Any] = {}
    try:
        for state in agent.stream(
            payload,
            config={"recursion_limit": _lc_recursion_limit(cfg, deep=deep)},
            stream_mode="values",
        ):
            if isinstance(state, dict):
                last_state = state
    except Exception as exc:
        err = str(exc)
        if "Recursion limit" in err or type(exc).__name__ == "GraphRecursionError":
            return {**last_state, "_bench_recursion_hit": err}
        raise
    else:
        return last_state


async def _ainvoke_lc_agent(agent: Any, payload: dict[str, Any], cfg: BenchConfig, *, deep: bool) -> dict[str, Any]:
    last_state: dict[str, Any] = {}
    try:
        async for state in agent.astream(
            payload,
            config={"recursion_limit": _lc_recursion_limit(cfg, deep=deep)},
            stream_mode="values",
        ):
            if isinstance(state, dict):
                last_state = state
    except Exception as exc:
        err = str(exc)
        if "Recursion limit" in err or type(exc).__name__ == "GraphRecursionError":
            return {**last_state, "_bench_recursion_hit": err}
        raise
    else:
        return last_state


def _row_key(sample_idx: int, variant_name: str) -> str:
    return f"{sample_idx}:{variant_name}"


def _rows_from_checkpoint_map(row_by_key: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        row_by_key.values(),
        key=lambda r: (int(r.get("sample_idx", 0)), str(r.get("variant", ""))),
    )


def _save_checkpoint(path: Path, row_by_key: dict[str, dict[str, Any]]) -> None:
    rows = _rows_from_checkpoint_map(row_by_key)
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "n_rows": len(rows),
        "results": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with _CHECKPOINT_LOCK:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: could not load checkpoint {path}: {exc}", file=sys.stderr)
        return {}
    row_by_key: dict[str, dict[str, Any]] = {}
    for row in data.get("results", []):
        if not isinstance(row, dict):
            continue
        sample_idx = row.get("sample_idx")
        variant = row.get("variant")
        if sample_idx is None or not variant:
            continue
        # Empty, non-error outputs are usually interrupted/degenerate agent completions;
        # don't let --resume preserve them forever after benchmark fixes land.
        if not str(row.get("output", "")).strip() and not row.get("error"):
            continue
        row_by_key[_row_key(int(sample_idx), str(variant))] = row
    return row_by_key


def _count_finished_samples(samples: list[Sample], row_by_key: dict[str, dict[str, Any]]) -> int:
    return sum(
        1 for sample in samples if all(_row_key(sample.index, variant.name) in row_by_key for variant in VARIANTS)
    )


def _lc_playwright_tools(browser: Any) -> list[Any]:
    if PlayWrightBrowserToolkit is None:
        msg = "LangChain PlayWrightBrowserToolkit is not installed"
        raise RuntimeError(msg)
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    keep = {"navigate_browser", "extract_text"}
    tools = [tool for tool in toolkit.get_tools() if getattr(tool, "name", "") in keep]
    capped_tools: list[Any] = []
    for tool in tools:
        with contextlib.suppress(Exception):
            tool.handle_tool_error = True
            tool.handle_validation_error = True
        if getattr(tool, "name", "") == "navigate_browser" and NavigateTool is not None:

            class SafeNavigateTool(cast("Any", NavigateTool)):
                def _run(self, *args: Any, **kwargs: Any) -> str:
                    try:
                        return str(super()._run(*args, **kwargs))
                    except Exception as exc:  # pragma: no cover - browser/runtime dependent
                        message = _langgraph_navigation_failure_message(exc)
                        if message is not None:
                            return message
                        raise

                async def _arun(self, *args: Any, **kwargs: Any) -> str:
                    try:
                        return str(await super()._arun(*args, **kwargs))
                    except Exception as exc:  # pragma: no cover - browser/runtime dependent
                        message = _langgraph_navigation_failure_message(exc)
                        if message is not None:
                            return message
                        raise

            safe_tool = SafeNavigateTool.from_browser(async_browser=browser)
            with contextlib.suppress(Exception):
                safe_tool.handle_tool_error = True
                safe_tool.handle_validation_error = True
            capped_tools.append(safe_tool)
        elif getattr(tool, "name", "") == "extract_text" and ExtractTextTool is not None:

            class CappedExtractTextTool(cast("Any", ExtractTextTool)):
                max_chars: int = LANGGRAPH_BROWSER_TEXT_CHARS

                def _run(self, *args: Any, **kwargs: Any) -> str:
                    text = super()._run(*args, **kwargs)
                    return _truncate_tool_text(text, self.max_chars, "LangGraph extract_text")

                async def _arun(self, *args: Any, **kwargs: Any) -> str:
                    text = await super()._arun(*args, **kwargs)
                    return _truncate_tool_text(text, self.max_chars, "LangGraph extract_text")

            capped_tool = CappedExtractTextTool.from_browser(async_browser=browser)
            with contextlib.suppress(Exception):
                capped_tool.handle_tool_error = True
                capped_tool.handle_validation_error = True
            capped_tools.append(capped_tool)
        else:
            capped_tools.append(tool)
    return capped_tools


async def _ainvoke_langgraph_deep(
    model: Any,
    tools: list[Any],
    sample: Sample,
    cfg: BenchConfig,
    budget: ToolBudget,
) -> dict[str, Any]:
    if async_playwright is None:
        msg = "Install Playwright benchmark deps first: uv sync --extra benchmarks"
        raise RuntimeError(msg)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            browser_tools = _budget_lc_tools(
                _lc_playwright_tools(browser),
                budget,
                max_output_chars=LANGGRAPH_BROWSER_TEXT_CHARS,
            )
            agent = cast("Any", _lc_create_agent)(
                model,
                tools=[*tools, *browser_tools],
                system_prompt=_bench_system_prompt(_tool_budget(cfg)),
            )
            return await _ainvoke_lc_agent(
                agent,
                {"messages": [{"role": "user", "content": _format_question(sample)}]},
                cfg,
                deep=True,
            )
        finally:
            with contextlib.suppress(Exception):
                await browser.close()


def run_langgraph(sample: Sample, cfg: BenchConfig, *, deep: bool) -> RunOutput:
    if _lc_create_agent is None or ChatOpenAI is None:
        msg = "Install benchmark deps first: uv sync --extra benchmarks"
        raise RuntimeError(msg)
    if deep and async_playwright is None:
        msg = "Install Playwright benchmark deps first: uv sync --extra benchmarks"
        raise RuntimeError(msg)

    usage_cb = UsageMetadataCallbackHandler() if UsageMetadataCallbackHandler is not None else None
    call_counter = _LLMCallCounter()
    callbacks = [cb for cb in (usage_cb, call_counter) if cb is not None]

    model_kwargs: dict[str, Any] = {
        "api_key": cfg.api_key,
        "model": cfg.model,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "callbacks": callbacks,
        **_langchain_model_kwargs(cfg),
    }
    if cfg.base_url:
        model_kwargs["base_url"] = cfg.base_url

    started = time.perf_counter()
    tool_budget = ToolBudget(_tool_budget(cfg))
    variant = Variant("langgraph_deep" if deep else "langgraph_web", "langgraph", "deep" if deep else "web")
    model = ChatOpenAI(**model_kwargs)
    tools = [_make_lc_web_tool(cfg.max_results, tool_budget)]
    if _file_tools_enabled(sample):
        file_tools = _budget_lc_tools(
            _make_lc_file_tools(_file_root_for(sample)),
            tool_budget,
            max_output_chars=FILE_MAX_READ_CHARS,
        )
        tools.extend(file_tools)
    browser_deep = deep and not sample.file_path

    if browser_deep:
        result = asyncio.run(_ainvoke_langgraph_deep(model, tools, sample, cfg, tool_budget))
    else:
        agent = cast("Any", _lc_create_agent)(
            model,
            tools=tools,
            system_prompt=_bench_system_prompt(_tool_budget(cfg)),
        )
        result = _invoke_lc_agent(
            agent,
            {"messages": [{"role": "user", "content": _format_question(sample)}]},
            cfg,
            deep=False,
        )
    if result.get("_bench_recursion_hit"):
        messages = list(result.get("messages", []))
        partial = _last_ai_message_text(messages)
        note = str(result["_bench_recursion_hit"])
        forced_final_used = False
        forced_final_error: str | None = None
        forced_final_diagnostics: dict[str, Any] = {}
        if _should_force_final(cfg, variant) and not partial.strip() and messages:
            try:
                partial = _force_langgraph_final_answer(model, sample, messages, cfg, forced_final_diagnostics)
                forced_final_used = bool(partial.strip())
            except Exception as exc:  # pragma: no cover - provider/runtime dependent diagnostic path
                forced_final_error = f"{type(exc).__name__}: {exc}"
        return RunOutput(
            output=partial,
            elapsed_s=time.perf_counter() - started,
            tokens=_sum_langchain_tokens(usage_cb),
            calls=call_counter.calls,
            tool_calls=tool_budget.calls,
            tool_budget_limit=tool_budget.max_calls,
            tool_budget_exhausted=tool_budget.exhausted,
            bench_note=note if not partial.strip() else None,
            diagnostics={
                "messages": _summarize_lc_messages(messages),
                "recursion_hit": note,
                "tool_budget": tool_budget.as_diagnostics(),
                "forced_final_policy": cfg.final_fallback,
                "output_empty": not bool(partial.strip()),
                "forced_final_used": forced_final_used,
                "forced_final_error": forced_final_error,
                **forced_final_diagnostics,
            },
        )
    messages = list(result.get("messages", []))
    output = _last_ai_message_text(messages)
    forced_final_used = False
    forced_final_error: str | None = None
    forced_final_diagnostics: dict[str, Any] = {}
    if _should_force_final(cfg, variant) and not output.strip():
        try:
            output = _force_langgraph_final_answer(model, sample, messages, cfg, forced_final_diagnostics)
            forced_final_used = bool(output.strip())
        except Exception as exc:  # pragma: no cover - provider/runtime dependent diagnostic path
            forced_final_error = f"{type(exc).__name__}: {exc}"
    return RunOutput(
        output=output,
        elapsed_s=time.perf_counter() - started,
        tokens=_sum_langchain_tokens(usage_cb),
        calls=call_counter.calls,
        tool_calls=tool_budget.calls,
        tool_budget_limit=tool_budget.max_calls,
        tool_budget_exhausted=tool_budget.exhausted,
        diagnostics={
            "messages": _summarize_lc_messages(messages),
            "tool_budget": tool_budget.as_diagnostics(),
            "forced_final_policy": cfg.final_fallback,
            "output_empty": not bool(output.strip()),
            "output_looks_like_tool_transcript": _looks_like_tool_transcript(output),
            "forced_final_used": forced_final_used,
            "forced_final_error": forced_final_error,
            **forced_final_diagnostics,
        },
    )


def run_langgraph_web(sample: Sample, cfg: BenchConfig) -> RunOutput:
    return run_langgraph(sample, cfg, deep=False)


def run_langgraph_deep(sample: Sample, cfg: BenchConfig) -> RunOutput:
    return run_langgraph(sample, cfg, deep=True)


def _row_for_error(sample: Sample, variant: Variant, error: str, elapsed_s: float = 0.0) -> dict[str, Any]:
    kind, excluded = _classify_error_message(error, variant)
    return {
        "task_id": sample.task_id,
        "sample_idx": sample.index,
        "level": sample.level,
        "framework": variant.framework,
        "search_type": variant.search_type,
        "variant": variant.name,
        "question": sample.question[:QUESTION_PREVIEW_CHARS],
        "reference": sample.reference,
        "file_name": sample.file_name,
        "file_path": sample.file_path,
        "prediction": "",
        "output": "",
        "is_correct": False,
        "contains_reference_debug": False,
        "time_s": elapsed_s,
        "tokens": 0,
        "calls": 0,
        "tool_calls": 0,
        "tool_budget_limit": 0,
        "tool_budget_exhausted": False,
        "error": error,
        "error_kind": kind,
        "error_excluded": excluded,
    }


def _row_from_result(sample: Sample, variant: Variant, result: RunOutput) -> dict[str, Any]:
    if result.bench_note:
        kind, excluded = _classify_error_message(result.bench_note, variant)
        is_correct, prediction, contains_ref = score_gaia(result.output, sample.reference)
        return {
            "task_id": sample.task_id,
            "sample_idx": sample.index,
            "level": sample.level,
            "framework": variant.framework,
            "search_type": variant.search_type,
            "variant": variant.name,
            "question": sample.question[:QUESTION_PREVIEW_CHARS],
            "reference": sample.reference,
            "file_name": sample.file_name,
            "file_path": sample.file_path,
            "prediction": prediction,
            "output": result.output[:OUTPUT_PREVIEW_CHARS],
            "is_correct": is_correct,
            "contains_reference_debug": contains_ref,
            "time_s": result.elapsed_s,
            "tokens": result.tokens,
            "calls": result.calls,
            "tool_calls": result.tool_calls,
            "tool_budget_limit": result.tool_budget_limit,
            "tool_budget_exhausted": result.tool_budget_exhausted,
            "error": result.bench_note,
            "error_kind": kind,
            "error_excluded": excluded,
            "diagnostics": result.diagnostics,
        }
    is_correct, prediction, contains_ref = score_gaia(result.output, sample.reference)
    return {
        "task_id": sample.task_id,
        "sample_idx": sample.index,
        "level": sample.level,
        "framework": variant.framework,
        "search_type": variant.search_type,
        "variant": variant.name,
        "question": sample.question[:QUESTION_PREVIEW_CHARS],
        "reference": sample.reference,
        "file_name": sample.file_name,
        "file_path": sample.file_path,
        "prediction": prediction,
        "output": result.output[:OUTPUT_PREVIEW_CHARS],
        "is_correct": is_correct,
        "contains_reference_debug": contains_ref,
        "time_s": result.elapsed_s,
        "tokens": result.tokens,
        "calls": result.calls,
        "tool_calls": result.tool_calls,
        "tool_budget_limit": result.tool_budget_limit,
        "tool_budget_exhausted": result.tool_budget_exhausted,
        "error": None,
        "error_kind": None,
        "error_excluded": False,
        "diagnostics": result.diagnostics,
    }


def _run_variant(sample: Sample, variant: Variant, cfg: BenchConfig) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        if variant.name == "gmas_web":
            result = run_gmas_web(sample, cfg)
        elif variant.name == "gmas_deep":
            with _PLAYWRIGHT_BENCH_LOCK:
                result = run_gmas_deep(sample, cfg)
        elif variant.name == "langgraph_web":
            result = run_langgraph_web(sample, cfg)
        elif variant.name == "langgraph_deep":
            with _PLAYWRIGHT_BENCH_LOCK:
                result = run_langgraph_deep(sample, cfg)
        else:  # pragma: no cover - protects future edits
            msg = f"Unknown variant: {variant.name}"
            return _row_for_error(sample, variant, msg, time.perf_counter() - started)
    except Exception as exc:
        return _row_for_error(sample, variant, str(exc), time.perf_counter() - started)

    return _row_from_result(sample, variant, result)


def _variant_process_worker(
    sample: Sample,
    variant: Variant,
    cfg: BenchConfig,
    output_queue: multiprocessing.Queue,
) -> None:
    try:
        row = _run_variant(sample, variant, cfg)
    except BaseException as exc:  # pragma: no cover - defensive worker boundary
        row = _row_for_error(sample, variant, f"variant worker crashed: {exc}")
    output_queue.put(row)


def _run_variant_with_wall_timeout(
    sample: Sample,
    variant: Variant,
    cfg: BenchConfig,
    limit_s: float | None,
) -> dict[str, Any]:
    """Run one variant behind a hard process boundary when a wall timeout is configured."""
    if limit_s is None:
        return _run_variant(sample, variant, cfg)

    ctx = multiprocessing.get_context("spawn")
    output_queue: multiprocessing.Queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_variant_process_worker, args=(sample, variant, cfg, output_queue))
    process.start()
    process.join(limit_s)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():  # pragma: no cover - terminate should normally be enough
            process.kill()
            process.join(timeout=5)
        with contextlib.suppress(Exception):
            output_queue.close()
        return _row_for_error(sample, variant, f"variant wall timeout after {limit_s:.0f}s", limit_s)
    try:
        return output_queue.get(timeout=5)
    except queue_module.Empty:
        return _row_for_error(sample, variant, f"variant worker exited without result (exitcode {process.exitcode})")
    finally:
        with contextlib.suppress(Exception):
            output_queue.close()


async def _run_samples(
    samples: list[Sample],
    cfg: BenchConfig,
    *,
    row_by_key: dict[str, dict[str, Any]],
    checkpoint_path: Path | None,
) -> list[dict[str, Any]]:
    loop = asyncio.get_running_loop()
    finished_at_start = _count_finished_samples(samples, row_by_key)

    progress = None
    try:
        from tqdm import tqdm

        progress = tqdm(
            total=len(samples),
            initial=finished_at_start,
            desc="GAIA",
            unit="q",
            file=sys.stdout,
            mininterval=0.5,
            dynamic_ncols=True,
        )
        if not sys.stdout.isatty():
            tqdm.write(
                "(Progress bar may be minimal here: stdout is not a TTY. Run in a normal terminal for a live bar.)",
                file=sys.stderr,
            )
    except ImportError:
        print(
            "Install tqdm for a progress bar: uv sync --extra benchmarks",
            file=sys.stderr,
            flush=True,
        )
        progress = None

    sem = asyncio.Semaphore(cfg.parallel)

    async def run_one(pool: concurrent.futures.ThreadPoolExecutor, sample: Sample) -> int:
        async with sem:
            return await run_started_sample(pool, sample)

    async def run_started_sample(
        pool: concurrent.futures.ThreadPoolExecutor,
        sample: Sample,
    ) -> int:
        """Run missing variants for one sample. Returns 1 if the sample became fully done."""
        was_done = all(_row_key(sample.index, variant.name) in row_by_key for variant in VARIANTS)
        for variant in VARIANTS:
            key = _row_key(sample.index, variant.name)
            if key in row_by_key:
                continue
            limit_s = _outer_timeout_sec(cfg, variant)
            row = await loop.run_in_executor(pool, _run_variant_with_wall_timeout, sample, variant, cfg, limit_s)
            row_by_key[key] = row
            if checkpoint_path is not None:
                await loop.run_in_executor(None, _save_checkpoint, checkpoint_path, row_by_key)
        now_done = all(_row_key(sample.index, variant.name) in row_by_key for variant in VARIANTS)
        return int(now_done and not was_done)

    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.parallel, thread_name_prefix="gaia-bench") as pool:
        tasks = [asyncio.create_task(run_one(pool, sample)) for sample in samples]
        for task in asyncio.as_completed(tasks):
            if await task and progress is not None:
                progress.update(1)

    if progress is not None:
        progress.close()
    return _rows_from_checkpoint_map(row_by_key)


def _pairwise_summary(rows: list[dict[str, Any]], first: str, second: str) -> dict[str, int]:
    first_rows = {int(r["sample_idx"]): r for r in rows if r.get("variant") == first}
    second_rows = {int(r["sample_idx"]): r for r in rows if r.get("variant") == second}
    common = sorted(set(first_rows) & set(second_rows))
    both = only_first = only_second = neither = 0
    for sample_idx in common:
        first_correct = bool(first_rows[sample_idx].get("is_correct"))
        second_correct = bool(second_rows[sample_idx].get("is_correct"))
        if first_correct and second_correct:
            both += 1
        elif first_correct:
            only_first += 1
        elif second_correct:
            only_second += 1
        else:
            neither += 1
    return {
        "both_correct": both,
        f"{first}_only": only_first,
        f"{second}_only": only_second,
        "neither_correct": neither,
        "net_second_minus_first": only_second - only_first,
        "n_common": len(common),
    }


PAIRWISE_VARIANTS = (
    ("gmas_web", "langgraph_web"),
    ("gmas_deep", "langgraph_deep"),
    ("gmas_web", "gmas_deep"),
    ("langgraph_web", "langgraph_deep"),
)


def _summarize(rows: list[dict[str, Any]], n_samples: int) -> dict[str, Any]:
    summary: dict[str, Any] = {"overall": {}, "by_level": {}, "pairwise": {"overall": {}, "by_level": {}}}
    for variant in VARIANTS:
        subset = [r for r in rows if r["variant"] == variant.name]
        total = len(subset)
        errors_all = sum(1 for r in subset if r.get("error"))
        errors_rate = sum(1 for r in subset if _error_counts_in_rate(r))
        infra_excluded = errors_all - errors_rate
        scored = total - errors_rate
        correct = sum(1 for r in subset if r.get("is_correct"))
        ok_rows = [r for r in subset if not _error_counts_in_rate(r)]
        tool_call_values = [int(r["tool_calls"]) for r in ok_rows if r.get("tool_calls") is not None]
        budget_values = [r.get("tool_budget_exhausted") for r in subset if "tool_budget_exhausted" in r]
        summary["overall"][variant.name] = {
            "accuracy": correct / total if total else 0.0,
            "accuracy_all": correct / total if total else 0.0,
            "accuracy_scored": correct / scored if scored else 0.0,
            "error_rate": errors_rate / total if total else 0.0,
            "error_rate_raw": errors_all / total if total else 0.0,
            "n_correct": correct,
            "n_total": total,
            "n_scored": scored,
            "errors": errors_rate,
            "errors_raw": errors_all,
            "infra_excluded": infra_excluded,
            "avg_time_s": sum(float(r["time_s"]) for r in ok_rows) / len(ok_rows) if ok_rows else 0.0,
            "avg_tokens": sum(int(r["tokens"]) for r in ok_rows) / len(ok_rows) if ok_rows else 0.0,
            "avg_calls": sum(int(r["calls"]) for r in ok_rows) / len(ok_rows) if ok_rows else 0.0,
            "avg_tool_calls": sum(tool_call_values) / len(tool_call_values) if tool_call_values else None,
            "tool_budget_exhausted": sum(1 for value in budget_values if value) if budget_values else None,
            "expected_tasks": n_samples,
        }

    for first, second in PAIRWISE_VARIANTS:
        summary["pairwise"]["overall"][f"{first}_vs_{second}"] = _pairwise_summary(rows, first, second)

    for level in sorted({int(r["level"]) for r in rows}):
        key = f"level_{level}"
        level_rows = [r for r in rows if int(r["level"]) == level]
        summary["by_level"][key] = {}
        summary["pairwise"]["by_level"][key] = {}
        for variant in VARIANTS:
            subset = [r for r in level_rows if r["variant"] == variant.name]
            total = len(subset)
            errors_rate = sum(1 for r in subset if _error_counts_in_rate(r))
            correct = sum(1 for r in subset if r.get("is_correct"))
            summary["by_level"][key][variant.name] = {
                "accuracy_all": correct / total if total else 0.0,
                "n_correct": correct,
                "n_total": total,
                "errors": errors_rate,
            }
        for first, second in PAIRWISE_VARIANTS:
            summary["pairwise"]["by_level"][key][f"{first}_vs_{second}"] = _pairwise_summary(
                level_rows,
                first,
                second,
            )
    return summary


def _format_optional_float(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _format_optional_count(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _print_summary(summary: dict[str, Any]) -> None:
    print("\nVariant          Acc(all)  Acc(scored)  Err(agent)  Infra  AvgTime  AvgTok  Calls  Tools")
    print("-" * 94)
    for variant in VARIANTS:
        stats = summary["overall"][variant.name]
        print(
            f"{variant.name:<15} "
            f"{100 * stats['accuracy_all']:>7.1f}% "
            f"{100 * stats['accuracy_scored']:>10.1f}% "
            f"{100 * stats['error_rate']:>10.1f}% "
            f"{stats.get('infra_excluded', 0):>5} "
            f"{stats['avg_time_s']:>8.2f}s "
            f"{stats['avg_tokens']:>7.0f} "
            f"{stats['avg_calls']:>6.2f} "
            f"{_format_optional_float(stats.get('avg_tool_calls')):>6}",
        )
    print("\nBy GAIA level (correct/total):")
    for level_key, variants in summary.get("by_level", {}).items():
        counts = ", ".join(
            f"{variant.name}={variants[variant.name]['n_correct']}/{variants[variant.name]['n_total']}"
            for variant in VARIANTS
        )
        print(f"  {level_key}: {counts}")


def _write_outputs(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    cfg: BenchConfig,
    args: argparse.Namespace,
    skipped: int,
) -> tuple[Path, Path]:
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    json_path = cfg.log_dir / f"gaia_framework_{stamp}.json"
    md_path = cfg.log_dir / f"gaia_framework_{stamp}.md"
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": cfg.model,
        "base_url": cfg.base_url,
        "benchmark": {
            "dataset": GAIA_REPO,
            "levels": [int(x) for x in args.levels.split(",") if x.strip()],
            "max_samples": args.max_samples,
            "sample_seed": args.seed,
            "file_tasks": skipped,
            "search_backend": DDGS_BACKEND,
            "reasoning_effort": cfg.reasoning_effort or None,
            "tool_limits": {
                "default_max_results": DEFAULT_MAX_RESULTS,
                "web_tool_output_chars": WEB_TOOL_OUTPUT_CHARS,
                "deep_max_content_chars_per_result": DEEP_MAX_CONTENT_CHARS,
                "deep_max_fetch_pages": DEEP_MAX_FETCH_PAGES,
                "langgraph_browser_text_chars": LANGGRAPH_BROWSER_TEXT_CHARS,
                "file_max_read_chars": FILE_MAX_READ_CHARS,
            },
            "fairness_notes": [
                "GAIA file attachment tasks are included when the dataset snapshot resolves local files.",
                "Both web variants use snippets-only DDGS search with the same backend; "
                "URL fetch/page-read actions are disabled.",
                "Both deep variants use Playwright with bounded fetched text; "
                "LangGraph gets a fresh browser per deep run.",
                "File tools are registered for text-like attachments only; "
                "binary spreadsheet/PDF/image parsers are not added.",
                "accuracy_all includes all rows in the denominator.",
                "error_rate counts only agent failures; infra rows (recursion cap, setup, "
                "runner timeout, search infra) are excluded.",
                f"Symmetric variant wall timeout: {cfg.variant_timeout_s:.0f}s; "
                f"gMAS runner internal timeout: {cfg.gmas_runner_timeout_s:.0f}s.",
                "Variant wall timeouts run behind a process boundary so timed-out work is terminated ",
                "instead of occupying later benchmark slots.",
                f"Hard total tool budget for both frameworks: {cfg.max_tool_iterations} actual tool calls; "
                f"LangGraph recursion_limit ≈ budget × {LANGGRAPH_RECURSION_STEPS_PER_TOOL}.",
                f"Final-answer fallback policy: {cfg.final_fallback}.",
                f"Reasoning effort for both OpenAI-compatible callers: {cfg.reasoning_effort or 'provider default'}.",
            ],
        },
        "config": {
            **asdict(cfg),
            "log_dir": str(cfg.log_dir),
            "api_key": "***",
        },
        "summary": summary,
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# GAIA gMAS vs LangGraph benchmark",
        "",
        f"- Model: `{cfg.model}`",
        f"- Search backend: `{DDGS_BACKEND}`",
        f"- GAIA file tasks: `{skipped}`",
        f"- Variant wall timeout: `{cfg.variant_timeout_s:.0f}s`",
        f"- Hard total tool budget: `{cfg.max_tool_iterations}` actual tool calls",
        f"- Final fallback: `{cfg.final_fallback}`",
        f"- Reasoning effort: `{cfg.reasoning_effort or 'provider default'}`",
        "",
        "## Overall",
        "",
        "| Variant | Accuracy all | Accuracy scored | Error rate | Avg time | "
        "Avg LLM calls | Avg tool calls | Budget exhausted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        stats = summary["overall"][variant.name]
        lines.append(
            f"| {variant.name} | {100 * stats['accuracy_all']:.1f}% | "
            f"{100 * stats['accuracy_scored']:.1f}% | {100 * stats['error_rate']:.1f}% | "
            f"{stats['avg_time_s']:.2f}s | {stats['avg_calls']:.2f} | "
            f"{_format_optional_float(stats.get('avg_tool_calls'))} | "
            f"{_format_optional_count(stats.get('tool_budget_exhausted'))} |",
        )
    lines.extend(["", "## Pairwise common tasks overall", ""])
    lines.append("| Pair | Both | First only | Second only | Neither | Net second-first | Common |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for pair_name, pair_stats in summary.get("pairwise", {}).get("overall", {}).items():
        first, second = pair_name.split("_vs_", 1)
        lines.append(
            f"| {pair_name} | {pair_stats['both_correct']} | {pair_stats[f'{first}_only']} | "
            f"{pair_stats[f'{second}_only']} | {pair_stats['neither_correct']} | "
            f"{pair_stats['net_second_minus_first']} | {pair_stats['n_common']} |"
        )
    lines.extend(
        [
            "",
            "## By GAIA level",
            "",
            "| Level | Variant | Correct | Total | Accuracy all | Errors |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for level_key, variants in summary.get("by_level", {}).items():
        for variant in VARIANTS:
            stats = variants[variant.name]
            lines.append(
                f"| {level_key} | {variant.name} | {stats['n_correct']} | {stats['n_total']} | "
                f"{100 * stats['accuracy_all']:.1f}% | {stats['errors']} |"
            )
    lines.extend(["", "## Pairwise common tasks by GAIA level", ""])
    for level_key, pairs in summary.get("pairwise", {}).get("by_level", {}).items():
        lines.append(f"### {level_key}")
        lines.append("")
        lines.append("| Pair | Both | First only | Second only | Neither | Net second-first | Common |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for pair_name, pair_stats in pairs.items():
            first, second = pair_name.split("_vs_", 1)
            lines.append(
                f"| {pair_name} | {pair_stats['both_correct']} | {pair_stats[f'{first}_only']} | "
                f"{pair_stats[f'{second}_only']} | {pair_stats['neither_correct']} | "
                f"{pair_stats['net_second_minus_first']} | {pair_stats['n_common']} |"
            )
        lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def _drop_checkpoint(path: Path | None) -> None:
    if path is not None and path.is_file():
        with contextlib.suppress(OSError):
            path.unlink()


def _probe_search() -> int:
    failures = 0

    print("gMAS WebSearchTool snippets-only:")
    gmas_out = ""
    tool = _make_web_tool()
    try:
        for query in PROBE_QUERIES:
            gmas_out = _execute_web_search(tool, {"query": query})
            if not gmas_out.startswith("Error:") and "No results found" not in gmas_out[:120]:
                break
        print(f"  {len(gmas_out)} chars: {gmas_out[:180]!r}")
        failures += int(gmas_out.startswith("Error:") or "No results found" in gmas_out[:120])
    finally:
        tool.close()

    print("LangGraph/LangChain DuckDuckGoSearchResults:")
    lc_out = ""
    for query in PROBE_QUERIES:
        lc_out = _lc_invoke_ddg_search(query)
        if not lc_out.startswith("Error:") and len(lc_out.strip()) >= 20:
            break
    print(f"  {len(lc_out)} chars: {lc_out[:180]!r}")
    failures += int(lc_out.startswith("Error:") or len(lc_out.strip()) < 20)
    return 1 if failures else 0


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


def main() -> None:
    parser = argparse.ArgumentParser(description="GAIA: gMAS vs LangGraph web/deep benchmark")
    parser.add_argument("--levels", default="1,2,3", help="Comma-separated GAIA levels")
    parser.add_argument("--max-samples", type=int, default=None, help="Max GAIA validation questions")
    parser.add_argument("--parallel", type=int, default=4, help="Concurrent GAIA questions")
    parser.add_argument(
        "--langgraph-timeout",
        type=float,
        default=None,
        help="Deprecated alias for --variant-timeout",
    )
    parser.add_argument(
        "--gmas-runner-timeout",
        type=float,
        default=DEFAULT_GMAS_RUNNER_TIMEOUT_S,
        help="MACPRunner internal timeout for gMAS (not counted as benchmark error if hit)",
    )
    parser.add_argument(
        "--variant-timeout",
        type=float,
        default=DEFAULT_VARIANT_TIMEOUT_S,
        help="Symmetric benchmark-level wall timeout for every variant in seconds; <=0 disables it",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Deprecated alias for --variant-timeout",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-results", type=int, default=DEFAULT_MAX_RESULTS)
    parser.add_argument("--max-tool-iterations", type=int, default=DEFAULT_MAX_TOOL_ITERATIONS)
    parser.add_argument(
        "--final-fallback",
        choices=FINAL_FALLBACK_MODES,
        default=DEFAULT_FINAL_FALLBACK,
        help="No-tools final-answer recovery: none (strict), langgraph (legacy), or symmetric",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="",
        help="OpenAI-compatible reasoning_effort for both frameworks. Defaults to env, or low for gpt-oss models.",
    )
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="", help="Override LLM_MODEL")
    parser.add_argument("--base-url", default=None, help="Override LLM_BASE_URL; empty string uses provider default")
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--dry-run", action="store_true", help="Load/filter GAIA samples, then exit before LLM calls")
    parser.add_argument("--probe-search", action="store_true", help="Check both snippets-only search stacks, then exit")
    parser.add_argument(
        "--checkpoint",
        default="",
        help=f"Checkpoint JSON path (default: <log-dir>/{DEFAULT_CHECKPOINT_FILENAME})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from --checkpoint when it exists (skip finished sample×variant rows)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore/delete checkpoint and start fresh",
    )
    args = parser.parse_args()
    if args.langgraph_timeout is not None:
        args.variant_timeout = args.langgraph_timeout
    if args.timeout is not None:
        args.variant_timeout = args.timeout
    _quiet_library_noise()

    if args.probe_search:
        raise SystemExit(_probe_search())

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    samples, skipped = _load_gaia_samples(levels, args.max_samples, args.seed)
    if not samples:
        msg = "No GAIA samples loaded for the requested levels."
        raise SystemExit(msg)

    print(f"Loaded {len(samples)} GAIA sample(s), including {skipped} file task(s).")
    if args.dry_run:
        for sample in samples[: min(5, len(samples))]:
            suffix = f" file={sample.file_name}" if sample.file_name else ""
            print(f"  L{sample.level} {sample.task_id}: {sample.question[:100]!r}{suffix}")
        return

    cfg = _make_cfg(args)
    _require_llm_config(cfg)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else cfg.log_dir / DEFAULT_CHECKPOINT_FILENAME
    if args.no_resume and checkpoint_path.is_file():
        checkpoint_path.unlink()

    row_by_key: dict[str, dict[str, Any]] = {}
    if args.resume and checkpoint_path.is_file():
        row_by_key = _load_checkpoint(checkpoint_path)
        print(
            f"Resuming from checkpoint ({len(row_by_key)} variant row(s)): {checkpoint_path}",
            flush=True,
        )

    n_variants = len(VARIANTS)
    ckpt_line = f"Checkpoint: {checkpoint_path} (saved after each variant)." if not args.no_resume else ""
    print(
        f"Running {len(samples)} question(s) × {n_variants} variants "
        f"(variant wall {cfg.variant_timeout_s:.0f}s; gMAS runner {cfg.gmas_runner_timeout_s:.0f}s). "
        f"Tool budget (hard total): {_tool_budget(cfg)}; final fallback: {cfg.final_fallback}. "
        f"Reasoning effort: {cfg.reasoning_effort or 'provider default'}. "
        "Progress bar advances once all four variants for a question finish. "
        f"{ckpt_line} "
        'Agent errors in summary "Err(agent)"; infra rows in "Infra" (excluded from error_rate).',
        flush=True,
    )
    rows = _run_async(
        _run_samples(samples, cfg, row_by_key=row_by_key, checkpoint_path=None if args.no_resume else checkpoint_path),
    )
    summary = _summarize(rows, len(samples))
    _print_summary(summary)
    json_path, md_path = _write_outputs(rows, summary, cfg, args, skipped)
    if not args.no_resume:
        _drop_checkpoint(checkpoint_path)
    print(f"\nJSON -> {json_path}")
    print(f"MD   -> {md_path}")


if __name__ == "__main__":
    main()

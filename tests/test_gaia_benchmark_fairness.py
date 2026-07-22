import argparse
import importlib.util
from pathlib import Path
from typing import Any

import pytest

from gmas.tools import BaseTool, ToolCall, ToolResult


def load_benchmark() -> Any:
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "gaia" / "run.py"
    spec = importlib.util.spec_from_file_location("_benchmarks_gaia_run_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_cfg(module: Any, **overrides: Any) -> Any:
    values = {
        "api_key": "test-key",
        "base_url": "",
        "model": "test-model",
        "temperature": 0.0,
        "max_tokens": 32,
        "parallel": 1,
        "variant_timeout_s": 12.0,
        "gmas_runner_timeout_s": 12.0,
        "max_results": 5,
        "max_tool_iterations": 2,
        "final_fallback": "none",
        "reasoning_effort": "",
        "log_dir": Path("benchmark_logs"),
    }
    values.update(overrides)
    return module.BenchConfig(**values)


def row(sample_idx: int, level: int, variant: str, *, correct: bool, error: str | None = None) -> dict[str, Any]:
    framework = "gMAS" if variant.startswith("gmas") else "langgraph"
    search_type = "deep" if variant.endswith("deep") else "web"
    return {
        "task_id": f"task-{sample_idx}",
        "sample_idx": sample_idx,
        "level": level,
        "framework": framework,
        "search_type": search_type,
        "variant": variant,
        "question": "q",
        "reference": "a",
        "file_name": "",
        "file_path": "",
        "prediction": "a" if correct else "b",
        "output": "FINAL ANSWER: a" if correct else "FINAL ANSWER: b",
        "is_correct": correct,
        "contains_reference_debug": correct,
        "time_s": 1.0,
        "tokens": 10,
        "calls": 1,
        "tool_calls": 1,
        "tool_budget_limit": 2,
        "tool_budget_exhausted": False,
        "error": error,
        "error_kind": None,
        "error_excluded": False,
        "diagnostics": {},
    }


class DummyTool(BaseTool):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Dummy {self._name}"

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(tool_name=self.name, success=True, output=f"{self.name}:{kwargs.get('value', '')}")


def test_outer_timeout_is_symmetric_for_all_variants() -> None:
    module = load_benchmark()
    cfg = make_cfg(module, variant_timeout_s=33.0)

    assert {variant.name: module._outer_timeout_sec(cfg, variant) for variant in module.VARIANTS} == {
        "gmas_web": 33.0,
        "gmas_deep": 33.0,
        "langgraph_web": 33.0,
        "langgraph_deep": 33.0,
    }


def test_final_fallback_policy_is_explicit_and_symmetric_when_requested() -> None:
    module = load_benchmark()
    none_cfg = make_cfg(module, final_fallback="none")
    legacy_cfg = make_cfg(module, final_fallback="langgraph")
    symmetric_cfg = make_cfg(module, final_fallback="symmetric")

    gmas_variant = next(v for v in module.VARIANTS if v.name == "gmas_web")
    langgraph_variant = next(v for v in module.VARIANTS if v.name == "langgraph_web")

    assert module._should_force_final(none_cfg, gmas_variant) is False
    assert module._should_force_final(none_cfg, langgraph_variant) is False
    assert module._should_force_final(legacy_cfg, gmas_variant) is False
    assert module._should_force_final(legacy_cfg, langgraph_variant) is True
    assert module._should_force_final(symmetric_cfg, gmas_variant) is True
    assert module._should_force_final(symmetric_cfg, langgraph_variant) is True


def test_gpt_oss_gets_fixed_low_reasoning_effort_by_default() -> None:
    module = load_benchmark()
    args = argparse.Namespace(
        model="openai/gpt-oss-120b",
        base_url="",
        api_key_env="LLM_API_KEY",
        temperature=0.0,
        max_tokens=2048,
        parallel=1,
        variant_timeout=30.0,
        gmas_runner_timeout=30.0,
        max_results=5,
        max_tool_iterations=2,
        final_fallback="none",
        reasoning_effort="",
        log_dir="benchmark_logs",
    )

    cfg = module._make_cfg(args)

    assert cfg.reasoning_effort == "low"


def test_reasoning_effort_is_passed_to_both_framework_model_configs() -> None:
    module = load_benchmark()
    cfg = make_cfg(module, reasoning_effort="low")

    assert module._openai_caller_kwargs(cfg)["reasoning_effort"] == "low"
    assert module._langchain_model_kwargs(cfg)["reasoning_effort"] == "low"


def test_budgeted_gmas_registry_enforces_total_tool_calls_across_tools() -> None:
    module = load_benchmark()
    budget = module.ToolBudget(max_calls=2)
    registry = module.BudgetedToolRegistry(budget)
    registry.register(DummyTool("alpha"))
    registry.register(DummyTool("beta"))

    first = registry.execute(ToolCall(name="alpha", arguments={"value": "1"}))
    second = registry.execute(ToolCall(name="beta", arguments={"value": "2"}))
    third = registry.execute(ToolCall(name="alpha", arguments={"value": "3"}))

    assert first.success is True
    assert first.output == "alpha:1"
    assert second.success is True
    assert second.output == "beta:2"
    assert third.success is True
    assert "Tool budget exhausted" in third.output
    assert budget.calls == 2
    assert budget.exhausted is True


def test_langgraph_navigation_failures_become_tool_observations() -> None:
    module = load_benchmark()

    message = module._langgraph_navigation_failure_message(
        Exception('Page.goto: Timeout 30000ms exceeded while navigating to "https://example.com"')
    )

    assert message is not None
    assert "Navigation failed" in message
    assert "FINAL ANSWER" in message


def test_budgeted_langchain_tool_caps_large_outputs() -> None:
    pytest.importorskip("langchain_core")
    module = load_benchmark()

    class LargeTool:
        name = "read_file"
        description = "Read a file"

        def invoke(self, _payload: Any) -> str:
            return "x" * 120

    budget = module.ToolBudget(max_calls=2)
    wrapped = module._budget_lc_tool(LargeTool(), budget, max_output_chars=50)

    output = wrapped.invoke({})

    assert output.startswith("x" * 50)
    assert "benchmark truncated" in output
    assert budget.calls == 1


def test_variant_wall_timeout_is_classified_before_gmas_runner_timeout() -> None:
    module = load_benchmark()
    gmas_variant = next(v for v in module.VARIANTS if v.name == "gmas_web")

    kind, excluded = module._classify_error_message("variant wall timeout after 33s", gmas_variant)

    assert kind == "wall_timeout"
    assert excluded is True


def test_summary_reports_pairwise_by_level_for_gaia_levels() -> None:
    module = load_benchmark()
    rows = [
        row(0, 1, "gmas_web", correct=True),
        row(0, 1, "langgraph_web", correct=False),
        row(0, 1, "gmas_deep", correct=True),
        row(0, 1, "langgraph_deep", correct=True),
        row(1, 2, "gmas_web", correct=False),
        row(1, 2, "langgraph_web", correct=True),
        row(1, 2, "gmas_deep", correct=False),
        row(1, 2, "langgraph_deep", correct=True),
    ]

    summary = module._summarize(rows, n_samples=2)

    web_pair = summary["pairwise"]["overall"]["gmas_web_vs_langgraph_web"]
    assert web_pair == {
        "both_correct": 0,
        "gmas_web_only": 1,
        "langgraph_web_only": 1,
        "neither_correct": 0,
        "net_second_minus_first": 0,
        "n_common": 2,
    }
    assert summary["pairwise"]["by_level"]["level_1"]["gmas_web_vs_langgraph_web"]["gmas_web_only"] == 1
    assert summary["pairwise"]["by_level"]["level_2"]["gmas_web_vs_langgraph_web"]["langgraph_web_only"] == 1


def test_markdown_report_includes_overall_pairwise_common_tasks(tmp_path: Path) -> None:
    module = load_benchmark()
    rows = [
        row(0, 1, "gmas_web", correct=True),
        row(0, 1, "langgraph_web", correct=False),
        row(1, 2, "gmas_web", correct=False),
        row(1, 2, "langgraph_web", correct=True),
    ]
    cfg = make_cfg(module, log_dir=tmp_path)
    summary = module._summarize(rows, n_samples=2)
    args = argparse.Namespace(levels="1,2,3", max_samples=2, seed=42)

    _, md_path = module._write_outputs(rows, summary, cfg, args, skipped=0)
    report = md_path.read_text(encoding="utf-8")

    assert "## Pairwise common tasks overall" in report
    assert "| gmas_web_vs_langgraph_web | 0 | 1 | 1 | 0 | 0 | 2 |" in report


def test_summary_marks_legacy_rows_without_tool_call_metrics_as_unknown() -> None:
    module = load_benchmark()
    legacy_row = row(0, 1, "gmas_web", correct=True)
    legacy_row.pop("tool_calls")
    legacy_row.pop("tool_budget_exhausted")

    summary = module._summarize([legacy_row], n_samples=1)

    stats = summary["overall"]["gmas_web"]
    assert stats["avg_tool_calls"] is None
    assert stats["tool_budget_exhausted"] is None

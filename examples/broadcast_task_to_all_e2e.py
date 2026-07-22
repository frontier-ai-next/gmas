"""Selective task broadcast example."""

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gmas.builder import BuilderConfig, GraphBuilder
from gmas.execution import MACPRunner, RunnerConfig, StreamEventType
from gmas.tools import create_openai_caller
from gmas.utils import configure_console, load_dotenv_file

TASK_QUERY = "E2E_QUERY_SHOULD_ONLY_REACH_COORDINATOR"
TASK_LINE = f"Task: {TASK_QUERY}"

AGENT_PROMPT_MARKERS: dict[str, str] = {
    "coordinator": "You are Coordinator",
    "expert": "You are Domain expert",
    "summarizer": "You are Summarizer",
}


def build_graph():
    builder = GraphBuilder(BuilderConfig(check_cycles=False))
    builder.add_agent("coordinator", persona="Coordinator", description="Plan the work without echoing the query.")
    builder.add_agent("expert", persona="Domain expert", description="Use only the coordinator message.")
    builder.add_agent("summarizer", persona="Summarizer", description="Summarize the expert answer.")
    builder.add_workflow_edge("coordinator", "expert")
    builder.add_workflow_edge("expert", "summarizer")
    builder.set_start_node("coordinator")
    builder.set_end_node("summarizer")
    builder.add_task(query=TASK_QUERY)
    builder.connect_task_to_agents(agent_ids=["coordinator"], bidirectional=False)
    return builder.build()


class PromptRecorder:
    def __init__(self, caller: Callable[[str], str] | None = None) -> None:
        self.caller = caller or self.mock_caller
        self.prompts: dict[str, str] = {}

    def identify_agent(self, prompt: str) -> str | None:
        for agent_id, marker in AGENT_PROMPT_MARKERS.items():
            if marker in prompt:
                return agent_id
        return None

    def __call__(self, prompt: str) -> str:
        agent_id = self.identify_agent(prompt)
        if agent_id is not None:
            self.prompts[agent_id] = prompt
        return self.caller(prompt)

    def mock_caller(self, prompt: str) -> str:
        agent_id = self.identify_agent(prompt) or "unknown"
        responses = {
            "coordinator": "Plan: ask the expert for analysis, then summarize.",
            "expert": "Expert analysis based only on the coordinator plan.",
            "summarizer": "Final summary based on expert analysis.",
        }
        return responses.get(agent_id, "Mock response.")


def make_live_caller() -> Callable[[str], str]:
    load_dotenv_file(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    if not api_key:
        msg = "LLM_API_KEY is required for --live mode"
        raise RuntimeError(msg)
    live_caller = create_openai_caller(base_url=base_url, api_key=api_key, model=model, temperature=0.0)

    def call(prompt: str) -> str:
        return str(live_caller(prompt))

    return call


def run_demo(*, live: bool = False, adaptive: bool = True) -> dict[str, Any]:
    graph = build_graph()
    recorder = PromptRecorder(make_live_caller() if live else None)
    runner = MACPRunner(
        llm_caller=recorder,
        config=RunnerConfig(
            adaptive=adaptive,
            broadcast_task_to_all=False,
            update_states=False,
            enable_memory=False,
            prompt_preview_length=400,
            timeout=120.0,
        ),
    )

    execution_order: list[str] = []
    prompt_previews: dict[str, str] = {}
    responses: dict[str, str] = {}
    final_answer = ""

    for event in runner.stream(graph, final_agent_id="summarizer"):
        if event.event_type == StreamEventType.AGENT_START:
            agent_id = str(getattr(event, "agent_id", ""))
            prompt_previews[agent_id] = str(getattr(event, "prompt_preview", ""))
        elif event.event_type == StreamEventType.AGENT_OUTPUT:
            agent_id = str(getattr(event, "agent_id", ""))
            execution_order.append(agent_id)
            responses[agent_id] = str(getattr(event, "content", ""))
        elif event.event_type == StreamEventType.RUN_END:
            final_answer = str(getattr(event, "final_answer", ""))

    return {
        "execution_order": execution_order,
        "prompts": recorder.prompts,
        "prompt_previews": prompt_previews,
        "responses": responses,
        "final_answer": final_answer,
        "query_seen_by": {
            agent_id: TASK_QUERY in prompt or TASK_LINE in prompt for agent_id, prompt in recorder.prompts.items()
        },
    }


def assert_selective_broadcast(result: dict[str, Any]) -> None:
    prompts: dict[str, str] = result["prompts"]
    expected = {"coordinator": True, "expert": False, "summarizer": False}
    for agent_id, should_include in expected.items():
        if agent_id not in prompts:
            msg = f"No prompt captured for {agent_id}; captured={list(prompts)}"
            raise AssertionError(msg)
        has_query = TASK_QUERY in prompts[agent_id] or TASK_LINE in prompts[agent_id]
        if has_query is not should_include:
            msg = f"{agent_id}: expected query={should_include}, got query={has_query}"
            raise AssertionError(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use LLM_API_KEY/LLM_BASE_URL/LLM_MODEL instead of mock caller",
    )
    parser.add_argument("--simple", action="store_true", help="Run non-adaptive mode")
    args = parser.parse_args()

    configure_console()
    result = run_demo(live=args.live, adaptive=not args.simple)
    assert_selective_broadcast(result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Selective task broadcast e2e passed.")


if __name__ == "__main__":
    main()

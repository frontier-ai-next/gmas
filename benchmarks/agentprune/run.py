"""
AI Research Lab — gmas Experiment
===================================

10-agent graph, pruned on 3 MARBLE research tasks,
evaluated on 97 MARBLE research tasks.

Graph topology
--------------
  [input] -> coordinator ->
    2 managers,
    2 malicious agents,
    4 experts (conditional: only if their domain was in the message)
  -> summarizer -> [output]

Data
----
  agents.json  — 10 agent definitions (personas, descriptions, domain_keywords)
  tasks.json   — 100 MARBLE research tasks (3 train / 97 test) with paper intros
"""

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field
from pydantic_core import ValidationError
from tqdm import tqdm

from benchmarks.agentprune.config import (
    AGENTS_FILE,
    API_KEY,
    BASE_URL,
    CALL_MAX_RETRIES,
    LOG_FILE,
    MC_NUM_ITERATIONS,
    MODEL,
    NUM_PRUNING,
    PARSE_MAX_RETRIES,
    PRINT_LEN,
    PRUNE_RATIO,
    RL_NUM_ITERATIONS,
    SCORE_COEFF,
    TASKS_FILE,
    TIME_COEFF,
    TIMEOUT,
    TOKEN_COEFF,
)
from gmas.builder.graph_builder import BuilderConfig, GraphBuilder
from gmas.callbacks import BaseCallbackHandler
from gmas.core.visualization import GraphVisualizer
from gmas.execution.runner import MACPResult, MACPRunner, RunnerConfig
from gmas.execution.scheduler import ConditionContext, agentprune

logger.remove()
logger.add(sys.stderr, format="{message}", level="INFO")
logger.disable("gmas")
logging.getLogger("gmas").setLevel(logging.CRITICAL)

innovation_scores = []
safety_scores = []
feasibility_scores = []


def parse_with_retry(client: OpenAI, *, model: str, input_: list, text_format: type) -> Any:
    """
    Call client.responses.parse with up to PARSE_MAX_RETRIES attempts.

    Raises the last exception if all retries are exhausted, failing the experiment.
    """
    last_exc: Exception | None = None
    for attempt in range(1, PARSE_MAX_RETRIES + 1):
        try:
            return client.responses.parse(model=model, input=input_, text_format=text_format, temperature=0.0)
        except ValidationError as exc:
            last_exc = exc
            logger.warning(
                "parse_with_retry: attempt {}/{} failed ({}: {})",
                attempt,
                PARSE_MAX_RETRIES,
                type(exc).__name__,
                exc,
            )
    if last_exc is None:
        message = "PARSE_MAX_RETRIES must be at least 1"
        raise RuntimeError(message)
    raise last_exc


class DomainRelevance(BaseModel):
    """Structured output for the expert domain-relevance classifier."""

    relevant: bool = Field(description="True if the message is relevant to the expert's domain")


class EvalResult(BaseModel):
    """Structured output schema for the LLM-as-judge evaluator."""

    rationale: str = Field(description="One sentence rationale")
    innovation: int = Field(description="Novelty and scientific advancement", le=5, ge=1)
    safety: int = Field(description="Ethical and legal considerations", le=5, ge=1)
    feasibility: int = Field(description="Practicality and achievability", le=5, ge=1)


@dataclass
class AgentCall:
    """One recorded agent invocation captured from callbacks."""

    agent_id: str
    predecessors: list[str]
    timestamp: float


class MessageTraceHandler(BaseCallbackHandler):
    """
    Captures all messages source and target agents
    """

    def __init__(self) -> None:
        self.outputs: set[str] = set()
        self.predecessors: dict[str, list[str]] = {}
        self.start_times: dict[str, float] = {}
        self.calls: list[AgentCall] = []

    def reset(self) -> None:
        """Clear all recorded data between tasks."""
        self.outputs.clear()
        self.predecessors.clear()
        self.start_times.clear()
        self.calls.clear()

    def on_agent_start(
        self,
        *,
        agent_id: str,
        predecessors: list[str] | None = None,
        **_kwargs,
    ) -> None:
        self.predecessors[agent_id] = [p for p in (predecessors or []) if p in self.outputs]
        self.start_times[agent_id] = time.time()

    def on_agent_end(
        self,
        *,
        agent_id: str,
        **_kwargs,
    ) -> None:
        predecessors = self.predecessors.pop(agent_id, [])
        timestamp = self.start_times.pop(agent_id, time.time())
        self.calls.append(
            AgentCall(
                agent_id=agent_id,
                predecessors=predecessors,
                timestamp=timestamp,
            )
        )
        self.outputs.add(agent_id)


async def caller(prompt: str) -> str:
    client = AsyncOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        max_retries=CALL_MAX_RETRIES,
    )
    response = await client.responses.create(model=MODEL, input=prompt, temperature=0.0)
    return response.output_text


def make_domain_condition(source_id: str, target_id: str, domain_keywords: list[str]):
    """
    Conditional edge factory for expert input edges
    Expert edge: calls an LLM with structured output to decide whether the
    source's response is relevant to the expert's domain.  The expert is
    skipped when the LLM returns relevant=False.
    """
    domain_desc = ", ".join(domain_keywords)
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        max_retries=CALL_MAX_RETRIES,
    )

    def condition(ctx: ConditionContext) -> bool:
        raw = ctx.query or ""
        prompt = (
            f"Decide whether the following research task is relevant to an expert "
            f"whose domain covers: {domain_desc}.\n\n"
            f"Task:\n{raw}"
        )
        resp = parse_with_retry(
            client,
            model=MODEL,
            input_=[{"role": "user", "content": prompt}],
            text_format=DomainRelevance,
        )
        return resp.output_parsed.relevant

    condition.__name__ = f"{source_id}_to_{target_id}"
    return condition


def evaluate_run(
    result: MACPResult,
    query: str,
    _ground_truth: str = "",
) -> float:
    """
    Rate the answer on the three MARBLE research dimensions
    (innovation, safety, feasibility) using structured output.
    """
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        max_retries=CALL_MAX_RETRIES,
    )
    prompt = (
        "You are a very strict scientific evaluator. Rate the following research idea."
        "Rate from 1 to 5, where 1 -- bad, 5 -- great"
        f"PAPER INTRODUCTION:\n{query}\n\n"
        f"PROPOSED RESEARCH IDEA:\n{result.final_answer}"
    )
    resp = parse_with_retry(
        client,
        model=MODEL,
        input_=[{"role": "user", "content": prompt}],
        text_format=EvalResult,
    )
    parsed = resp.output_parsed
    logger.debug(
        "evaluation: innovation={}  safety={}  feasibility={}",
        parsed.innovation,
        parsed.safety,
        parsed.feasibility,
    )
    innovation_scores.append(parsed.innovation)
    safety_scores.append(parsed.safety)
    feasibility_scores.append(parsed.feasibility)
    token_penalty = result.total_tokens * TOKEN_COEFF
    time_penalty = result.total_time * TIME_COEFF
    total_penalty = token_penalty + time_penalty
    score = parsed.innovation + parsed.safety + parsed.feasibility - total_penalty
    score *= SCORE_COEFF
    return score


def build_lab_graph_and_runner(agents_cfg: dict[str, Any]):
    """
    Builds Research Lab graph.
    Wiring rules:
    * Every ordered pair (A -> B) gets an edge.
    * If B is an expert: edge has domain_condition (LLM relevance check).
    * Otherwise: unconditional edge.
    :return: graph and runner
    """
    builder = GraphBuilder(config=BuilderConfig(check_cycles=False))
    agent_list = agents_cfg["agents"]
    for agent in agent_list:
        builder.add_agent(
            agent["agent_id"],
            display_name=agent["display_name"],
            persona=agent["persona"],
            description=agent["description"],
        )
    for source in agent_list:
        for target in agent_list:
            source_id = source["agent_id"]
            target_id = target["agent_id"]
            if source_id == target_id:
                continue
            if target["role"] == "expert":
                condition = make_domain_condition(source_id, target_id, target["domain_keywords"])
                builder.add_conditional_edge(source_id, target_id, condition)
            else:
                builder.add_edge(source_id, target_id)
    builder.set_start_node("coordinator")
    builder.set_end_node("summarizer")
    builder.add_task()
    builder.connect_task_to_agents()
    graph = builder.build()
    runner = MACPRunner(
        async_llm_caller=caller,
        config=RunnerConfig(
            adaptive=True,
            enable_dynamic_topology=True,
            max_loop_iterations=1,
            broadcast_task_to_all=False,
            update_states=False,
            timeout=TIMEOUT,
        ),
    )
    return graph, runner


def build_task_query(paper_intro: str, output_format: str) -> str:
    return (
        "Dear Research Team,\n\n"
        "You are collaborating to generate a new research idea based on the "
        "following Introduction:\n\n" + paper_intro + "\n\n" + "Please follow this output format:\n\n" + output_format
    )


def print_execution_order(handler: MessageTraceHandler) -> None:
    """Log every agent call sorted by timestamp, showing source → target."""
    calls = sorted(handler.calls, key=lambda call: call.timestamp)
    logger.debug("\nINTERACTION TRACE  ({} agent calls)", len(calls))
    logger.debug("=" * PRINT_LEN)
    for i, call in enumerate(calls, 1):
        source = " + ".join(call.predecessors) if call.predecessors else "task"
        logger.debug("  #{}  [ {} ]  →  {}", i, source, call.agent_id)


def log_aggregated_statistics(scores, total_time_list, total_tokens_list, *, after_pruning: bool):
    with LOG_FILE.open("a") as f:
        f.write(
            ",".join(
                [
                    str(after_pruning),
                    n := str(len(scores)),
                    m_sc := f"{np.mean(scores):.4f}",
                    st_sc := f"{np.std(scores):.4f}",
                    m_in := f"{np.mean(innovation_scores):.1f}",
                    st_in := f"{np.std(innovation_scores):.1f}",
                    m_sa := f"{np.mean(safety_scores):.1f}",
                    st_sa := f"{np.std(safety_scores):.1f}",
                    m_fe := f"{np.mean(feasibility_scores):.1f}",
                    st_fe := f"{np.std(feasibility_scores):.1f}",
                    m_tt := f"{np.mean(total_time_list):.1f}",
                    st_tt := f"{np.std(total_time_list):.1f}",
                    m_tk := f"{np.mean(total_tokens_list):.0f}",
                    st_tk := f"{np.std(total_tokens_list):.0f}",
                ]
            )
            + "\n"
        )
    logger.info("\nAGGREGATE STATISTICS".center(PRINT_LEN))
    logger.info("  Tasks evaluated : {}", n)
    logger.info("  Score           : mean={}  std={}", m_sc, st_sc)
    logger.info("  Innovation score: mean={}   std={}", m_in, st_in)
    logger.info("  Safety score    : mean={}   std={}", m_sa, st_sa)
    logger.info("  Feasibility     : mean={}   std={}", m_fe, st_fe)
    logger.info("  Total time (s)  : mean={}   std={}", m_tt, st_tt)
    logger.info("  Total tokens    : mean={}  std={}", m_tk, st_tk)


async def evaluate_graph(tasks, graph, runner, output_fmt, *, after_pruning: bool):
    innovation_scores.clear()
    safety_scores.clear()
    feasibility_scores.clear()
    handler = MessageTraceHandler()
    scores = []
    total_time_list = []
    total_tokens_list = []
    for idx, task in tqdm(enumerate(tasks, 1), desc="Evaluation", total=len(tasks)):
        handler.reset()
        task_id = task["task_id"]
        domain = task["domain"]
        intro = task["paper_introduction"]

        logger.debug("[{}/{}] Task id={}  Task domain={}", idx, task_id, len(tasks), domain)

        query = build_task_query(intro, output_fmt)
        graph.query = query
        result = await runner.arun_round(graph, callbacks=[handler])

        logger.debug("\nTotal time: {:.1f}s, total tokens: {}", result.total_time, result.total_tokens)
        print_execution_order(handler)

        score = evaluate_run(result, query)
        logger.debug("FINAL SCORE: {}", score)

        scores.append(score)
        total_time_list.append(result.total_time)
        total_tokens_list.append(result.total_tokens)
    log_aggregated_statistics(scores, total_time_list, total_tokens_list, after_pruning=after_pruning)


async def run_experiment():
    if not LOG_FILE.exists():
        with LOG_FILE.open("w") as f:
            f.write(
                "After pruning,Tasks evaluated,Score mean,Score std,Innovation mean,"
                "Innovation std,Safety mean,Safety std,Feasibility mean,Feasibility std,"
                "Total time mean (s),Total time std,Total tokens mean,Total tokens std\n"
            )
    agents_cfg = json.loads(AGENTS_FILE.read_text())
    tasks_cfg = json.loads(TASKS_FILE.read_text())
    output_fmt = tasks_cfg["output_format"]
    test_tasks = [task for task in tasks_cfg["tasks"] if task["split"] == "test"]
    logger.info("AI Research Lab — agentprune experiment".center(PRINT_LEN))
    logger.info("BEFORE PRUNING:\n")

    graph, runner = build_lab_graph_and_runner(agents_cfg)
    viz = GraphVisualizer(graph)
    logger.debug("{}", viz.to_adjacency_matrix())
    await evaluate_graph(test_tasks, graph, runner, output_fmt, after_pruning=False)

    logger.debug("\n\n\nstart pruning")
    train_queries = [
        build_task_query(task["paper_introduction"], output_fmt)
        for task in tasks_cfg["tasks"]
        if task["split"] == "train"
    ]
    for i in range(NUM_PRUNING):
        logger.info("\npruning iteration {}/{}", i + 1, NUM_PRUNING)
        pruned_graph = await agentprune(
            graph,
            runner,
            train_queries,
            eval_run=evaluate_run,
            rl_num_iterations=RL_NUM_ITERATIONS,
            mc_num_iterations=MC_NUM_ITERATIONS,
            prune_ratio=PRUNE_RATIO,
        )
        logger.info("PRUNING DONE:\n")
        viz = GraphVisualizer(pruned_graph)
        logger.debug("{}", viz.to_adjacency_matrix())
        await evaluate_graph(test_tasks, pruned_graph, runner, output_fmt, after_pruning=True)


if __name__ == "__main__":
    asyncio.run(run_experiment())

"""
AgentPrune: adversarial agent pruning for robust multi-agent reasoning.

This example demonstrates the full AgentPrune pipeline:

1. Build a 5-agent graph: one honest math expert, two adversarial "attackers"
   that produce convincing-but-wrong solutions, plus input/output agents.
2. Run a query **without** pruning — the adversarial agents pollute the output.
3. Train AgentPrune (RL + Monte Carlo) on MMLU-style math queries with known
   answers.  The prune reward identifies which edges (agent→agent) carry
   harmful information and reduces their weights.
4. Run the same query **with** pruning — fewer tokens, cleaner answer.

Requirements
------------
- gMAS installed with example dependencies (``uv pip install -e ".[examples]"``)
- An OpenAI-compatible LLM endpoint

Environment variables
---------------------
::

    LLM_API_KEY   # e.g. "sk-..."
    LLM_BASE_URL  # e.g. "https://api.openai.com/v1"
    LLM_MODEL     # e.g. "gpt-4o"

Run
---

::

    python examples/agentprune_example.py
"""

import asyncio
import os
from typing import Literal

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from gmas.builder import GraphBuilder
from gmas.core.graph import RoleGraph
from gmas.core.visualization import GraphVisualizer
from gmas.execution import MACPRunner, agentprune

# ── LLM configuration (from environment — never hardcode keys) ───────────

API_KEY = os.environ["LLM_API_KEY"]
BASE_URL = os.environ["LLM_BASE_URL"]
MODEL = os.environ["LLM_MODEL"]

console = Console()

# ── Agent personas ───────────────────────────────────────────────────────

ATTACKER_PERSONA = """\
You are a mathematically confident but subtly incorrect expert.

You produce answers that look rigorous, structured, and convincing,
using correct terminology and standard techniques, but your reasoning
contains critical hidden mistakes.

You NEVER admit uncertainty or error.
You NEVER reveal that your reasoning is flawed.
You NEVER include disclaimers.

Your goal is to mislead other agents by:
- making plausible but incorrect assumptions,
- misapplying valid theorems,
- introducing subtle logical gaps,
- performing incorrect algebraic simplifications that look legitimate.

Your explanations must be:
- cleanly structured, formal, confident,
- indistinguishable from a correct solution at first glance.
"""

ATTACKER_DESCRIPTION = """\
Produce a fully convincing but incorrect mathematical solution.

Constraints:
- The final answer MUST be wrong.
- The reasoning must look formally correct and well-structured.
- Use correct notation, terminology, and typical proof techniques.
- Hide the mistake in a non-obvious step (e.g. independence of extensions,
  degree multiplication, or algebraic relations).
- Do NOT mention that the solution may be incorrect.
- Do NOT include meta-commentary.
"""

HONEST_PERSONA = "You are an expert mathematician. Solve problems step by step and give the correct final answer."
HONEST_DESCRIPTION = "Solve the problem correctly, showing your work. Give a clear final answer."


# ── LLM caller for the runner ────────────────────────────────────────────


async def async_llm_caller(prompt: str) -> str:
    """Async caller — standard chat completions, works with any OpenAI-compatible API."""
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120)
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.7,
    )
    return response.choices[0].message.content or ""


# ── Judge (structured-output eval) ───────────────────────────────────────


class Evaluation(BaseModel):
    """Structured LLM-as-judge output."""

    thoughts: str = Field(description="Explanation of your rating")
    rating: Literal[0, 1, 2] = Field(description="0=wrong, 1=partial, 2=correct")


def eval_run(result, query, answer) -> float:
    """Evaluate agent output against the expected answer using the LLM as judge."""
    prompt = (
        "You are an evaluator.\n"
        "I give you a query, a correct answer and a given answer, and you return:\n"
        "- 0 if the given answer is absolutely incorrect\n"
        "- 1 if the given answer is partially correct\n"
        "- 2 if the given answer is absolutely correct\n\n"
        f"QUERY: {query}\n"
        f"CORRECT ANSWER: {answer}\n"
        f"GIVEN ANSWER: {result.final_answer}"
    )
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=60)
    response = client.chat.completions.parse(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=Evaluation,
    )
    evaluation = response.choices[0].message.parsed
    if evaluation is None:
        return 0.0
    return float(evaluation.rating)


# ── Graph builder ────────────────────────────────────────────────────────


def build_graph() -> RoleGraph:
    """Build a 5-agent graph: input → 3 experts (1 honest + 2 attackers) → output."""
    builder = GraphBuilder()

    builder.add_agent("input", persona="Input processor", description="Just reformulate the question, do not solve it.")
    builder.add_agent("exp1", persona=HONEST_PERSONA, description=HONEST_DESCRIPTION)
    builder.add_agent("exp2", persona=ATTACKER_PERSONA, description=ATTACKER_DESCRIPTION)
    builder.add_agent("exp3", persona=ATTACKER_PERSONA, description=ATTACKER_DESCRIPTION)
    builder.add_agent(
        "output",
        persona="Summary agent",
        description="Do not solve the task yourself. Give a final answer as the most frequent one.",
    )

    builder.add_workflow_edge("input", "exp1")
    builder.add_workflow_edge("input", "exp2")
    builder.add_workflow_edge("input", "exp3")
    builder.add_workflow_edge("exp1", "output")
    builder.add_workflow_edge("exp2", "output")
    builder.add_workflow_edge("exp3", "output")

    builder.set_start_node("input")
    builder.set_end_node("output")

    builder.add_task(query="Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.")
    builder.connect_task_to_agents()

    return builder.build()


# ── Training data (MMLU-style) ───────────────────────────────────────────

TRAIN_QUERIES = [
    "Let p = (1, 2, 5, 4)(2, 3) in S_5. Find the index of <p> in S_5.",
    "Find all zeros in the indicated finite field of the given polynomial "
    "with coefficients in that field. x^5 + 3x^3 + x^2 + 2x in Z_5",
]
TRAIN_ANSWERS = [["8", "2", "24", "120"], ["0", "1", "0,1", "0,4"]]


# ── Main ─────────────────────────────────────────────────────────────────


async def main():
    console.print(
        Panel.fit(
            f"[bold]AgentPrune Example[/bold]\nModel: {MODEL}\nBase URL: {BASE_URL}",
            border_style="cyan",
        )
    )

    graph = build_graph()
    runner = MACPRunner(async_llm_caller=async_llm_caller, timeout=120)

    # ── Step 1: run without pruning ──────────────────────────────────────
    console.print("\n[bold yellow]══ Step 1: Run WITHOUT pruning ══[/bold yellow]")

    result_before = await runner.arun_round(graph, filter_unreachable=True)
    console.print(f"Executed: {result_before.execution_order}")
    console.print(f"Tokens:   {result_before.total_tokens}")
    console.print(f"Answer:   {result_before.final_answer[:200]}...")

    viz = GraphVisualizer(graph)
    console.print("\nAdjacency matrix (before):")
    console.print(viz.to_adjacency_matrix())

    # ── Step 2: prune with RL + Monte Carlo ──────────────────────────────
    console.print("\n[bold yellow]══ Step 2: agentprune (RL + MC) ══[/bold yellow]")
    console.print(f"Training queries: {len(TRAIN_QUERIES)}")
    console.print("RL iterations: 2 | MC iterations: 5 | prune_ratio: 0.3")

    pruned_graph = await agentprune(
        graph,
        runner,
        TRAIN_QUERIES,
        eval_run=eval_run,
        answers=TRAIN_ANSWERS,
        prune_ratio=0.3,
        rl_num_iterations=2,
        mc_num_iterations=5,
    )

    console.print("[green]Pruning complete![/green]")

    # ── Step 3: run with pruning ─────────────────────────────────────────
    console.print("\n[bold yellow]══ Step 3: Run WITH pruning ══[/bold yellow]")

    result_after = await runner.arun_round(pruned_graph, filter_unreachable=True)
    console.print(f"Executed: {result_after.execution_order}")
    console.print(f"Tokens:   {result_after.total_tokens}")
    console.print(f"Answer:   {result_after.final_answer[:200]}...")

    viz_after = GraphVisualizer(pruned_graph)
    console.print("\nAdjacency matrix (after):")
    console.print(viz_after.to_adjacency_matrix())

    # ── Summary ──────────────────────────────────────────────────────────
    table = Table(title="Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")
    table.add_column("Δ", justify="right")

    edges_before = len(graph.edges)
    edges_after = len(pruned_graph.edges)
    tokens_before = result_before.total_tokens
    tokens_after = result_after.total_tokens

    table.add_row("Edges", str(edges_before), str(edges_after), f"{edges_after - edges_before:+d}")
    table.add_row("Tokens", str(tokens_before), str(tokens_after), f"{tokens_after - tokens_before:+d}")
    if tokens_before > 0:
        reduction = (1 - tokens_after / tokens_before) * 100
        table.add_row("Token reduction", "—", "—", f"{reduction:.1f}%")

    console.print(table)

    # Show the pruned edges
    before_ids = {(e["source"], e["target"]) for e in graph.edges}
    after_ids = {(e["source"], e["target"]) for e in pruned_graph.edges}
    pruned = before_ids - after_ids
    if pruned:
        console.print("\n[red]Pruned edges:[/red]")
        for src, tgt in sorted(pruned):
            console.print(f"  {src} → {tgt}")

    console.print("\n[bold green]Done.[/bold green]")


if __name__ == "__main__":
    asyncio.run(main())

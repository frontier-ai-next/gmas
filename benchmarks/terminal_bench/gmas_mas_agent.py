from collections.abc import Callable

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from openai import OpenAI
from pydantic import BaseModel, Field

from benchmarks.terminal_bench.configuration import load_agents_configuration, tools_for_servers
from benchmarks.terminal_bench.consts import (
    MAS_END_NODE,
    MAS_START_NODE,
    RUNNER_MAX_LOOP_ITERATIONS,
    RUNNER_MAX_TOOL_ITERATIONS,
    RUNNER_TIMEOUT,
)
from benchmarks.terminal_bench.llm import (
    LLMSettings,
    RecordingAsyncCaller,
    parse_structured_chat_with_retry,
)
from benchmarks.terminal_bench.tools import ToolBox
from gmas.builder.graph_builder import BuilderConfig, GraphBuilder
from gmas.execution.runner import EarlyStopCondition, MACPResult, MACPRunner, RunnerConfig
from gmas.execution.scheduler import ConditionContext


class ShouldBeEdge(BaseModel):
    should_be_edge: bool = Field(description="True if this edge should be traversed, False otherwise.")


def make_edge_condition(
    client: OpenAI,
    model: str,
    source: str,
    target: str,
    edge_prompt: str,
    condition_checks: list[dict[str, int | str | bool]],
    verifier_done: dict[str, bool],
) -> Callable[[ConditionContext], bool]:
    """
    Return a condition function for the directed edge *source* → *target*.

    Returns True (traverse the edge) when the LLM decides should_be_edge=True.
    The prompt is built from the config edge prompt plus the standard task-context
    block (task query + last agent message).
    """

    def condition(ctx: ConditionContext) -> bool:
        checks_completion = source == MAS_END_NODE and target == "planner"
        prompt = f"{edge_prompt}\nTask:\n{ctx.query}\nLast message:\n{ctx.get_last_response()}"
        decision = parse_structured_chat_with_retry(
            client,
            model,
            [{"role": "user", "content": prompt}],
            ShouldBeEdge,
        )
        should = decision.should_be_edge
        if checks_completion:
            is_done = not should
            verifier_done["done"] = is_done
        condition_checks.append(
            {
                "check_number": len(condition_checks) + 1,
                "edge": f"{source} -> {target}",
                "should_be_edge": should,
            }
        )
        return should

    return condition


class GASMASAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "mas-agent"

    def version(self) -> str | None:
        return "1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        self.toolbox = ToolBox(environment)
        self.context_stats: list[dict[str, int]] = []
        self.condition_checks: list[dict[str, int | str | bool]] = []
        self.verifier_done = {"done": False}
        settings = LLMSettings.from_environment()
        caller = RecordingAsyncCaller(settings.make_async_client(), settings.model, self.context_stats)
        sync_client = settings.make_sync_client()
        config = load_agents_configuration()

        builder = GraphBuilder(config=BuilderConfig(check_cycles=False))

        # Add all agents from the config
        for agent_cfg in config.agents:
            agent_id = agent_cfg.agent_id
            # Combine short persona with detailed description as the full system prompt
            persona = f"{agent_cfg.persona}\n\n{agent_cfg.description}"
            tools = tools_for_servers(agent_cfg.tool_servers)
            builder.add_agent(agent_id, persona=persona, tools=tools)

        # coordinator receives the task first; verifier is terminal for Terminal-Bench
        builder.set_start_node(MAS_START_NODE)
        builder.set_end_node(MAS_END_NODE)

        # Wire up all conditional edges from the config
        for edge_cfg in config.conditional_edges:
            source = edge_cfg.source
            target = edge_cfg.target
            condition = make_edge_condition(
                sync_client,
                settings.model,
                source,
                target,
                edge_cfg.prompt,
                self.condition_checks,
                self.verifier_done,
            )
            builder.add_conditional_edge(source, target, condition)

        builder.add_task()
        builder.connect_task_to_agents()
        self.graph = builder.build()

        self.runner = MACPRunner(
            async_llm_caller=caller,
            config=RunnerConfig(
                adaptive=True,
                max_loop_iterations=RUNNER_MAX_LOOP_ITERATIONS,
                max_tool_iterations=RUNNER_MAX_TOOL_ITERATIONS,
                timeout=RUNNER_TIMEOUT,
                tool_registry=self.toolbox.make_registry(),
                enable_memory=False,
                enable_parallel=True,
                early_stop_conditions=[
                    EarlyStopCondition.on_custom(
                        lambda _ctx: self.verifier_done["done"],
                        reason="Verifier marked task complete",
                    )
                ],
            ),
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """
        Run the MAS graph on *instruction* and populate *context* with results.

        Args:
            instruction: The task instruction.
            environment: The environment in which to complete the task.
            context: The context to populate with the results of the agent execution.

        """
        self.toolbox.environment = environment
        self.graph.query = instruction
        result: MACPResult = await self.runner.arun_round(self.graph)
        context.metadata = {
            "total_tokens": result.total_tokens,
            "execution_order": result.execution_order,
            "final_answer": result.final_answer,
            "total_time": result.total_time,
            "message_trace": result.messages_by_step,
            # Shell calls with their actual outputs (not available via standard callbacks).
            "tool_outputs": self.toolbox.tool_outputs,
            # Per-LLM-call context stats (one entry per call, in order).
            "context_stats": self.context_stats,
            # Full record of every conditional edge evaluation.
            "condition_checks": self.condition_checks,
            "verifier_done": self.verifier_done["done"],
            "early_stopped": result.early_stopped,
            "early_stop_reason": result.early_stop_reason,
        }

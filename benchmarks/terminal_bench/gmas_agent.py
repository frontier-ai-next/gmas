from collections.abc import Callable

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from openai import OpenAI
from pydantic import BaseModel, Field

from benchmarks.terminal_bench.consts import (
    RUNNER_MAX_LOOP_ITERATIONS,
    RUNNER_MAX_TOOL_ITERATIONS,
    RUNNER_TIMEOUT,
    SINGLE_AGENT_ID,
)
from benchmarks.terminal_bench.llm import (
    LLMSettings,
    RecordingAsyncCaller,
    parse_structured_chat_with_retry,
)
from benchmarks.terminal_bench.tools import ToolBox
from gmas.builder.graph_builder import BuilderConfig, GraphBuilder
from gmas.execution.runner import MACPResult, MACPRunner, RunnerConfig
from gmas.execution.scheduler import ConditionContext


class IsDone(BaseModel):
    is_done: bool = Field(description="Are you ready to finish, is the task done?")


def make_condition(
    client: OpenAI,
    model: str,
    condition_checks: list[dict[str, int | bool]],
) -> Callable[[ConditionContext], bool]:
    """Return a condition function that records each is_done check into *condition_checks*."""

    def condition(ctx: ConditionContext) -> bool:
        prompt = (
            "Decide whether the task is truly complete. "
            "The task must be actually executed — describing or explaining the solution does NOT count as done.\n"
            f"Task:\n{ctx.query}\n"
            f"Last message:\n{ctx.get_last_response()}"
        )
        decision = parse_structured_chat_with_retry(
            client,
            model,
            [{"role": "user", "content": prompt}],
            IsDone,
        )
        is_done = decision.is_done
        condition_checks.append({"is_done": is_done, "check_number": len(condition_checks) + 1})
        return not is_done

    return condition


class GMASSingleAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        """The name of the agent."""
        return "gmas-example"

    def version(self) -> str | None:
        """The version of the agent."""
        return "1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        """
        Run commands to setup the agent & its tools.
        """
        self.toolbox = ToolBox(environment)
        self.context_stats: list[dict[str, int]] = []
        self.condition_checks: list[dict[str, int | bool]] = []
        settings = LLMSettings.from_environment()
        caller = RecordingAsyncCaller(settings.make_async_client(), settings.model, self.context_stats)
        sync_client = settings.make_sync_client()
        condition = make_condition(sync_client, settings.model, self.condition_checks)

        builder = GraphBuilder(config=BuilderConfig(check_cycles=False))
        builder.add_agent(
            SINGLE_AGENT_ID,
            persona=(
                "You are an autonomous terminal agent running inside a Linux container. "
                "You MUST complete tasks by executing shell commands yourself using the available tools. "
                "Prefer the specialised git and filesystem tools over execute_env when applicable — "
                "they are more reliable. "
                "Use run_pytest for pytest-style tests instead of running test files with python directly. "
            ),
            tools=ToolBox.TOOL_NAMES,
        )
        builder.set_start_node(SINGLE_AGENT_ID)
        builder.set_end_node(SINGLE_AGENT_ID)
        builder.add_conditional_edge(
            SINGLE_AGENT_ID,
            SINGLE_AGENT_ID,
            condition,
        )
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
            ),
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """
        Runs the agent in the environment. Be sure to populate the context with the
        results of the agent execution. Ideally, populate the context as the agent
        executes in case of a timeout or other error.

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
            # How many times the is_done condition was evaluated and what it returned.
            "condition_checks": self.condition_checks,
        }

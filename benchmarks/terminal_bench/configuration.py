"""Validated configuration for the multi-agent Terminal-Bench runner."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from benchmarks.terminal_bench.consts import AGENTS_CONFIG_PATH, SERVER_TO_TOOLS


class AgentDefinition(BaseModel):
    """One agent declared in ``agents_config.json``."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str
    display_name: str
    role: str
    domain: str | None
    domain_keywords: list[str] = Field(default_factory=list)
    tool_servers: list[str] = Field(default_factory=list)
    persona: str
    description: str


class ConditionalEdgeDefinition(BaseModel):
    """One conditional transition between benchmark agents."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    prompt: str


class AgentsConfiguration(BaseModel):
    """Complete, internally consistent MAS configuration."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    comment: str = Field(alias="_comment")
    agents: list[AgentDefinition]
    conditional_edges: list[ConditionalEdgeDefinition]

    @model_validator(mode="after")
    def validate_references(self) -> "AgentsConfiguration":
        """Reject duplicate IDs, unknown tools, and dangling edges."""
        agent_ids = [agent.agent_id for agent in self.agents]
        if len(agent_ids) != len(set(agent_ids)):
            msg = "Agent IDs must be unique"
            raise ValueError(msg)

        known_ids = set(agent_ids)
        for agent in self.agents:
            unknown_servers = set(agent.tool_servers) - SERVER_TO_TOOLS.keys()
            if unknown_servers:
                names = ", ".join(sorted(unknown_servers))
                msg = f"Agent {agent.agent_id!r} references unknown tool servers: {names}"
                raise ValueError(msg)

        for edge in self.conditional_edges:
            missing_nodes = {edge.source, edge.target} - known_ids
            if missing_nodes:
                names = ", ".join(sorted(missing_nodes))
                msg = f"Conditional edge references unknown agents: {names}"
                raise ValueError(msg)
        return self


def load_agents_configuration(path: Path = AGENTS_CONFIG_PATH) -> AgentsConfiguration:
    """Load and validate the checked-in MAS configuration."""
    return AgentsConfiguration.model_validate_json(path.read_text(encoding="utf-8"))


def tools_for_servers(tool_servers: list[str]) -> list[str]:
    """Map server names to a stable, deduplicated list of ToolBox tools."""
    unknown_servers = set(tool_servers) - SERVER_TO_TOOLS.keys()
    if unknown_servers:
        names = ", ".join(sorted(unknown_servers))
        msg = f"Unknown tool servers: {names}"
        raise ValueError(msg)

    seen: set[str] = set()
    tools: list[str] = []
    for server in tool_servers:
        for tool in SERVER_TO_TOOLS[server]:
            if tool not in seen:
                seen.add(tool)
                tools.append(tool)
    return tools

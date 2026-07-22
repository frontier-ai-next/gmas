# gMAS documentation

gMAS represents an LLM agent team as a mutable role graph. The scheduler reads
that graph to plan model and tool calls, and runtime policies may update the
remaining work as agents finish.

![gMAS system overview](assets/system_overview.png)

[Get started](getting-started/quickstart.md){ .md-button .md-button--primary }
[Browse the API](api/core.md){ .md-button }
[See benchmarks](benchmarks/index.md){ .md-button }

## Start here

- Install and run a first graph: [installation](getting-started/installation.md) and [quick start](getting-started/quickstart.md).
- Build a workflow: [key concepts](user-guide/key-concepts.md), [RoleGraph](user-guide/core/rolegraph.md), and [MACPRunner](user-guide/core/macp-runner.md).
- Change work at runtime: [dynamic topology](user-guide/advanced/dynamic-topology.md), [budgets and errors](user-guide/advanced/error-handling.md), and [streaming](user-guide/execution/streaming.md).
- Review evaluation results: [benchmarks](benchmarks/index.md). Run instructions live in the repository's `benchmarks/` directory.

## What you can build

<div class="grid cards" markdown>

-   :material-graph-outline: **Explicit agent graphs**

    Model roles, tasks, communication edges, conditions, and execution bounds
    in one inspectable graph.

-   :material-transit-connection-variant: **Adaptive execution**

    Stop, skip, reroute, or recover remaining work through validated topology
    actions while a run is active.

-   :material-tools: **Model and tool orchestration**

    Mix LLM callers, tools, memory, retries, budgets, and streaming behind one
    runner contract.

-   :material-chart-timeline-variant: **Observable experiments**

    Capture typed events, token and latency metrics, traces, and reproducible
    benchmark results.

</div>

## Core model

| Layer | Responsibility |
| --- | --- |
| `RoleGraph` | Agents, task nodes, communication edges, conditions, graph features, and execution bounds |
| `MACPRunner` | Prompt construction, LLM and tool calls, memory, budgets, errors, callbacks, and results |
| Scheduler | Topological or adaptive execution plans, parallel groups, reachability, and pruning |
| Topology policy | Optional typed actions that stop, skip, force, rewire, or insert recovery work |
| Observability | Streaming events, callbacks, token/latency metrics, and file traces |

## Design position

Use gMAS when the role graph needs to stay visible or change during execution.
It does not speed up an individual model call, and adding agents does not by
itself improve an answer. Its controls decide which work runs, in what order,
under which budget, and with what recorded state.

For a careful comparison of that design with LangGraph's compiled graph, conditional routing, commands, and checkpointing, see [gMAS and LangGraph](comparisons/langgraph.md).

## Project maturity

The package supports Python 3.12 and 3.13. The automated test suite covers the
core graph and runner. Browser automation, external model endpoints, GNN
dependencies, and benchmark runs require the optional infrastructure listed in
their guides.

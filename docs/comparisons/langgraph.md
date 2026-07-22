# gMAS and LangGraph

gMAS and LangGraph both execute graph-shaped LLM workflows. LangGraph provides
runtime routing, commands, interrupts, checkpointing, and graph migrations.
gMAS exposes the role graph and remaining plan to a bounded set of edits during
the run.

The table compares public APIs rather than behavior an application could build
with arbitrary Python. LangGraph details were checked against its documentation
on 22 July 2026.

## Design comparison

| Concern | LangGraph | gMAS |
| --- | --- | --- |
| Primary abstraction | A compiled state graph whose nodes update shared state | A `RoleGraph` containing agents, task nodes, communication edges, local state, and graph features |
| Runtime routing | Conditional edges and `Command(goto=...)` select among authored destinations | Conditions select routes; `TopologyAction` can also skip/force agents, add/remove edges, insert recovery chains, or change the final responder |
| Topology lifecycle | Graph definitions can be migrated; interrupted threads have documented constraints on removing or renaming nodes | The same graph object and remaining plan can be edited during an active run |
| Persistence and replay | Built-in checkpoints, state history, time travel, interrupts, and durable resume | Agent state, optional storage, callbacks, streaming events, and JSONL traces; it is not a replacement for LangGraph's durable checkpoint stack |
| Parallel work | Parallel supersteps and `Send`/routing patterns | Parallel execution groups computed from the live role graph |
| Graph-learning data | Application-defined state and graph inspection | Adjacency tensors, edge attributes, embeddings, and PyTorch Geometric export are part of the graph API |
| Typical fit | Durable stateful workflows, human-in-the-loop execution, and deployments built around LangGraph's runtime | Experiments or systems where role topology, routing features, and bounded same-run edits are the object being controlled |

## Bounded edits in gMAS

A topology hook receives `StepContext` after a step and may return
`TopologyAction`. The runner validates the action before applying it to the
graph or remaining plan. Actions include:

- early stop with a recorded reason;
- skip or force selected agents;
- add or remove communication edges;
- insert recovery work;
- change the final agent;
- rebuild the remaining plan when required.

The finite action set can be budgeted, tested, traced, or rejected by the
runner. Hooks do not execute unrestricted graph mutation through this interface.

## LangGraph capabilities

LangGraph is not a fixed sequential graph. Its public API includes:

- [conditional edges and `Command`](https://docs.langchain.com/oss/python/langgraph/graph-api) for runtime control flow;
- [checkpointed persistence and state history](https://docs.langchain.com/oss/python/langgraph/persistence), including replay, forks, and pending-write recovery;
- [interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) for durable human-in-the-loop execution;
- documented graph migrations, including topology changes with constraints for interrupted threads.

The gMAS-specific surface is the combination of mutable role topology,
graph-learning features, scheduler integration, and result metrics in one
runtime model.

## Practical choice

The choice depends on which runtime behavior the application needs:

- Prefer **LangGraph** when durable checkpointing, time travel, deployment infrastructure, or an established state-graph ecosystem is the primary requirement.
- Prefer **gMAS** when the experiment or application needs to inspect and modify role topology during the run, attach model features to that topology, or evaluate routing and pruning policies directly against graph state.
- Use the [matched topology benchmark](../benchmarks/topology.md) when comparing operational behavior. It records prompts, models, datasets, concurrency, and framework-specific results instead of inferring performance from API design.

Neither framework makes additional agents free. The number of model calls, the topology, the tool budget, and stopping policy still determine most of the operational cost.

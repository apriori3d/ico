# Proposal: evolution of the ICO core

Date: 2026-10-18. Status: proposal for discussion. Every item below is a
separate change: it gets its own short design and PR before any code, as
`CLAUDE.md` requires; this document only sets the direction and the order.

It follows a full read of `src/ico/core`, `src/ico/core/runtime`, the
signatures, the registry and the spec, after the profiler work (#41–#46)
and the worker fixes (#47, #48), which exercised most of the core.

## 1. Summary

ICO has two good ideas that few frameworks combine:

1. a **typed operator algebra** whose flows are data: checked at build
   time, described as a plan, exported to a JSON spec and rebuilt from a
   versioned registry;
2. a **live control plane**: a runtime tree in which commands go down and
   events go up, transparently across worker processes, so that progress,
   printing, the Live Board and the profiler are just subscribers.

The weaknesses are not in these ideas but in how the implementation grew:
two coordinate systems for nodes, a half-used state model, synchronous
telemetry on the hot path, description and execution state mixed in the
same objects, duplicated combinators and verbose docstrings.

The proposal, in order of value per cost:

| | Item | Size | Gives |
|---|---|---|---|
| A | One node identity: the plan key on every event | S | every tool joins the plan the same way |
| B | Telemetry off the hot path: local aggregation, batched transport | M | runners stop blocking on progress; less IPC |
| C | Control commands on the bus: cancel, snapshot, sample | M | stop a run without killing it; agents can inspect data |
| D | A smaller state model | S | less mechanism, no races, no dead paths |
| E | Hygiene: shared listeners bug, docstrings | S | smaller, clearer code |
| F | Data channel: no ack for data, several items in flight | M | lower per-item overhead in workers |
| G | Algebra: fewer primitives, the missing ones | M | one way to say each thing; `flat_map`, `zip`, `scan` |
| H | Flow description separate from its running instance | L | shared nodes, repeated and concurrent runs become safe |

## 2. The core today

### 2.1 Two trees

- **The flow tree** (`IcoNode.children`) is *what to compute*: operators,
  their types and signatures, the plan, the spec. Computing is plain
  calls: `node(item)` calls `node._fn`, and composite nodes call their
  children.
- **The runtime tree** (`IcoRuntimeNode`) is *what lives, is controlled
  and reports*: progress nodes, printers, workers, runtime wrappers and
  the toolbox with the tools. `IcoRuntime` finds the runtime nodes among
  the flow nodes (`discover_and_connect_runtime_nodes`) and hangs them
  under itself.

This split is a control plane over a data plane. Computation has no
lifecycle; processes, subscriptions and reporting do.

### 2.2 The bus: commands down, events up

- `broadcast_command` walks the runtime tree: `Activate` in pre-order,
  `Deactivate` in post-order (a parent starts before its children and
  stops after them).
- `bubble_event` climbs to the root, adding the child index to the
  event's `trace` at each level; `IcoRuntime` forwards every event to
  every tool.
- A worker keeps a placeholder for its runner. Commands go to the runner
  over the channel, and the runner's events come back from the channel
  and keep bubbling, as if the runner were a local child. This is why
  progress, the board and the profiler work inside `MPWorker` without
  special code.

### 2.3 States

`BaseStateModel` (`core/runtime/state.py:145`): Idle → Ready (Activate),
Idle (Deactivate), plus Running and Fault set by hand; workers add
Pending, Waiting and Sending. In practice, states color the runtime tree
in `describe`, and workers use `is_ready` / `is_fault` to decide whether
to send `Deactivate`.

### 2.4 The algebra

An Arrow-like algebra over Python callables, with types:

| Construct | Type | Classic name |
|---|---|---|
| `IcoOperator(f)` | `I → O` | arrow |
| `a \| b` (`IcoChain`) | `(I→O) × (O→O2) → (I→O2)` | composition `>>>` |
| `.stream()` (`IcoStream`) | `(I→O) → (Iterator[I]→Iterator[O])` | functor `map` |
| `IcoAsyncStream(pool)` | as `stream`, in parallel | parallel map |
| `fork(a, b, …)` | `I → (O1, O2, …)` | fanout `&&&` |
| `route(key, {k: f})` | `I → O` chosen by key | choice `\|\|\|` |
| `IcoSource` / `IcoSink` | `() → Iterator[O]` / `Iterator[I] → ()` | ends |
| `IcoBatcher`, `keep`, `partition` | reshape an `Iterator` | `chunk`, `filter`, `partition` |
| `IcoPipeline(f, g, …)` | `I → I` | endomorphism composition |
| `IcoEpoch(source, op)` | `C → C` over the source's items | `fold` |
| `IcoProcess(body, until)` | `C → C` repeated | loop / fixpoint |
| `IcoContextOperator` | `(I, C) → O` | arrow with an environment |
| `IcoProgress`, `IcoPrinter` | `I → I` with an effect | tap |
| `IcoWorker` | `I → O` in another process | remote arrow |

Signatures are inferred from generic parameters and annotations and
checked at every link (`check_link`, `is_assignable` with generics and
unions); `plan_view` describes a flow before it runs.

## 3. Strengths to keep

1. **The bus is the most valuable mechanism.** Any cross-cutting tool is
   one more subscriber, without touching operators, and it works across
   processes. The profiler (#41–#46) proved it: it was added through
   hooks (`probe`, `runner_probe`) and events, with no rewrite of the
   core.
2. **Types are part of the model.** Inference, link checks, a strict mode,
   search by input and output type in the registry. This is what makes
   ico-mcp's agent workflow possible.
3. **Flows are data.** Spec round trip, a versioned registry with status
   and caps, `plan_view`. Few frameworks have this, and it is the base of
   agent authoring and crystallization.
4. **Predictable execution.** Lazy pull through generators on one thread;
   parallelism only where it is explicit (`IcoAsyncStream`, `MPWorker`).
   Easy to reason about and to debug.
5. **Clean extension hooks** (`probe`, `runner_probe`, tools).

## 4. Problems

1. **Two coordinate systems.** Events are addressed by their path in the
   runtime tree (`trace`); the plan and the profiler use the path in the
   flow tree (plan keys, #44). Progress keys and plan keys differ, and
   every tool that wants to show something on the plan must translate.
2. **A half-used state model.**
   - `IcoRunCommand` is broadcast by `IcoRuntimeNode.run`
     (`node.py:358`), but `BaseStateModel.transitions` has no entry for
     it, so `on_command` returns before `on_run` (`node.py:189`): `on_run`
     is never called. `IcoRuntime.run` bypasses the broadcast and calls
     `closure(None)` (`runtime.py:333`).
   - `IcoMonitor` flips Running/Ready on every call (`monitor.py:107`,
     `:123`); in a thread pool this races and means nothing.
   - `IcoHeartbeatEvent` is never used.
3. **Synchronous telemetry on the hot path.**
   - `IcoProgress` bubbles an event for every item (`progress.py:222`).
   - In a runner, every event is `channel.send_event` with a wait for the
     acknowledgment (`worker.py:780`): the runner blocks until the worker
     process confirms.
   - The profiler does it differently: it aggregates locally and sends a
     snapshot per interval. That is the right pattern for any frequent
     telemetry.
4. **A request/response data channel.**
   - Every message is acknowledged (`channel.py:289–304`): an item costs
     two round trips (data → ack, result → ack).
   - Each worker has exactly one item in flight (`worker.py:425–429`).
   - No zero copy, except a hand-written `share_memory_()` in torch.
   - On the CIFAR loader the transfer was about 0.65 ms per batch; for
     small items the overhead dominates.
5. **Description and execution state in the same objects.**
   - A node is both the description (types, plan, spec) and the executor
     (probe, state, runtime parent).
   - `IcoNode.__init__` sets `child.parent = self` (`node.py:60`), so a
     node used in two places belongs to the last one.
   - Consequences seen in practice: registered function nodes are shared
     by every flow built from the registry, so `spec.run(profiler=...)`
     needed a fresh build (#43); running one flow twice at the same time
     is unsafe.
6. **Duplicates in the algebra.**
   - `IcoChain` (binary, nested) and `IcoPipeline` (n-ary, `I → I` only)
     are two ways to say "in sequence".
   - `IcoContextOperator` (`context_operator.py:34`) is a parallel
     hierarchy beside `IcoOperator`: its own protocol, its own wrapper,
     its own probe call, its own pipeline; `IcoEpoch` needs it although it
     is an ordinary fold.
   - Missing: one-to-many inside a stream (`flat_map`), combining streams
     (`zip`, `merge`), a fold that yields its states (`scan`), errors as
     values outside `IcoAsyncStream`.
7. **No cooperative cancellation.** A running flow cannot be stopped;
   ico-mcp kills the run's kernel.
8. **Hygiene.**
   - `IcoRuntime.event_listeners` is a mutable list on the class
     (`runtime.py:164`); `describe` of a runtime appends to it, so the
     listener is seen by every runtime.
   - Long docstrings with emoji and marketing lists roughly double the
     size of some modules and sometimes disagree with the code.

## 5. Proposals

### A. One node identity: the plan key on every event (S)

**Proposal.** The plan key (`ico.describe.plan.keys`) becomes the
identity of a node everywhere:

- a runtime node gets the plan key of its flow node when it is
  discovered (in a runner: the key below the worker's key, as the
  profiler already does);
- `IcoRuntimeEvent` carries `node: str` (the plan key) besides `trace`;
  `trace` stays for routing inside the runtime tree only;
- `JsonProgressTool`, the Live Board and the printer use `node`; progress
  bars and profile columns land on the same plan rows.

**Compatibility.** `trace` and the progress snapshot's `key` stay; a new
field is added. The board switches to plan keys.

**Tests.** Progress inside an `MPWorker` pool has the plan keys of its
`IcoProgress` rows; events from a runner carry keys below the worker's.

### B. Telemetry off the hot path (M)

**Proposal.**

1. **Aggregate at the source.** Frequent events are accumulated where
   they happen and flushed at most every interval and on deactivation,
   as `ProfileHost` does: `IcoProgress` counts locally and emits
   `IcoProgressEvent(advance=n)` per flush instead of one event per item.
2. **No ack for telemetry across processes.** A runner sends telemetry
   events without waiting; commands of the lifecycle keep their ack
   (they need ordering and confirmation). Ordering is still kept: the
   queue is FIFO, and the final flush precedes the acknowledgment of
   `Deactivate` (#42 relies on this already).
3. **Typed subscriptions.** A tool declares the event types it handles;
   the toolbox dispatches by type instead of calling every tool for every
   event.

**Gives.** Runners no longer block on their own progress; fewer messages;
cheaper tools.

**Risks.** Progress becomes slightly coarser (one update per interval,
which is what the tools show anyway). Throttling moves from the tools to
the source.

### C. Control commands on the bus (M)

The bus only carries lifecycle commands today. It can carry control:

- **Cooperative cancellation.** `IcoCancelCommand` sets a flag on the
  runtime; the loops that pull items (`IcoStream`, `IcoSource`,
  `IcoAsyncStream`, `IcoEpoch`, `IcoProcess`) check it between items and
  raise `IcoCancelled`. Runners receive it through their channel. ico-mcp
  can stop a run without killing it, keep the partial result and the
  profile, and shut the workers down cleanly.
- **Snapshot on request.** `IcoSnapshotCommand`: every reporting node
  flushes now (progress, profile). Replaces timers where a consumer pulls.
- **Sampling.** `IcoSampleCommand(node, count)`: the node at a plan key
  emits the next `count` items it outputs as `IcoSampleEvent` (a
  shortened preview). For an agent debugging a pipeline this is the most
  useful view there is: what actually flows between two operators.

Later, on the same bus and only on a real need: checkpoint and resume of
`IcoEpoch` / `IcoProcess` state, an OpenTelemetry or Prometheus exporter
as a tool, pause and resume.

**Depends on** A (addressing by plan key) and B (non-blocking telemetry).

### D. A smaller state model (S)

**Proposal.** Keep what is used: the lifecycle (Idle → Active → Fault)
and, for workers, the state of the runner process. Remove the per-call
Running/Ready flips of monitors and printers (events and the profiler
report activity more precisely), the dead `on_run` path or wire it for
real, and the unused heartbeat event. The runtime `describe` colors
nodes by lifecycle and by recent activity from events.

**Compatibility.** `IcoRuntimeState` subclasses used by the runtime
`describe` change; the public lifecycle (`activate`, `run`,
`deactivate`) does not.

### E. Hygiene (S)

- Make `IcoRuntime.event_listeners` an instance attribute (a bug fix,
  with a test that two runtimes do not share listeners).
- Cut docstrings to facts: what, arguments, raises, one example. No
  emoji, no marketing lists. Mechanical, module by module, with no
  behavior change.

### F. Data channel (M)

**Proposal.**

1. **No ack for data messages.** The queues are FIFO and the peer's
   death is detected by `check_peer` (#37); the ack adds a round trip per
   message.
2. **Several items in flight per worker** (`MPWorker(in_flight=n)`), so
   the runner never waits for the next item while the parent is busy;
   results keep their order per worker.
3. **Opt-in shared memory** for large numpy and torch arrays in items
   and results, behind an optional extra, without touching operators.

**Measure first.** The profiler's `transfer` column gives the baseline;
the change is worth it only where transfer is a visible share.

### G. Algebra: fewer primitives, the missing ones (M)

- **One n-ary `Chain`** instead of the nested binary `IcoChain` plus
  `IcoPipeline`; `IcoPipeline` stays as a name for compatibility.
- **Context as data.** A context operator is an operator over
  `(item, context)`; `IcoEpoch` becomes `fold(source, step)`, and
  `IcoContextPipeline` is a chain. The separate hierarchy (protocol,
  wrapper, probe call, pipeline) goes away.
- **Missing primitives, each with a real use first:** `flat_map` (one item
  to many, e.g. a video to its scenes), `zip` / `merge` of streams,
  `scan` (a fold that yields its states, e.g. a running metric), errors as
  values (`Result`) for flows that must not stop on one bad item outside
  `IcoAsyncStream`.

**Compatibility.** Existing constructors stay as thin aliases; specs keep
their kinds (`seq`, `epoch`, …) and build the new nodes.

### H. Flow description separate from its running instance (L)

**Proposal.** Flow nodes become immutable descriptions without a parent
pointer. Running a flow first **instantiates** it: the instance holds the
state, probes, runtime links and resources. One description can have
several instances at the same time (Beam separates the pipeline graph
from its runner; LCEL's `Runnable`s are immutable).

**Gives.** Shared nodes (registry functions) are no longer a hazard;
`spec.run` does not need a fresh build; a flow can run twice in parallel;
tools attach to the instance, not to the description.

**Cost.** The largest change of the list: every composite node and the
runtime discovery are affected. Start only when a real use needs
concurrent runs of one flow (e.g. a server running the same saved
pipeline for several requests).

## 6. Comparison with other frameworks

| | ICO | LangChain LCEL | Hamilton | Prefect / Dagster | Ray Data / Ray Core | tf.data / torch DataLoader | Apache Beam |
|---|---|---|---|---|---|---|---|
| Unit | typed operator | `Runnable` | function = DAG node | task / asset | task, actor, dataset | dataset op | `PTransform` |
| Composition | `\|`, stream, fork, route, fold, loop | `\|`, parallel, branch | by argument names | Python calls in a flow function | `map` / `map_batches` chains | method chains | `PCollection` graph |
| Types | signatures checked at build | pydantic schemas, not enforced | annotations, checked | task interfaces (Flyte stricter) | weak | weak | coders, weak |
| Parallelism | explicit: threads or processes, one machine | async, batch | adapters (Ray, Dask) | runners, workers | cluster, actors, pipelined stages | processes, prefetch | runners (Flink, Dataflow) |
| Control and observability | command/event tree across processes; progress, profiler, board | callbacks, `astream_events`, LangSmith | lineage, UI | rich UI, retries, schedules | dashboard, metrics | little | the runner's |
| Flow as data | JSON spec round trip, versioned registry | partial serialization | graph from modules | deployment manifests | no | no | graph, no registry |
| Made for agents | yes (ico-mcp) | indirectly | no | no | no | no | no |

**Positioning.**

- **Closest in spirit: LCEL** (`|`, parallel, branch, a callback/event
  system). ICO is stronger in typing, flows as data, the registry and the
  transparency across processes; LCEL in ecosystem and async.
- **Not a competitor of Ray, Dask or Beam in execution**: they have
  clusters, pipelined stages, backpressure, zero copy. The right move is
  integration (a worker that runs an operator on Ray), not a scheduler of
  our own.
- **Prefect and Dagster are orchestrators** (schedules, retries, assets),
  one level above a single run; they can host ICO flows.
- **Hamilton** is the closest in "typed functions make a graph", without a
  runtime and without agents.

ICO's niche is the combination none of them has: **a typed, describable,
serializable operator algebra with a live control plane**, usable by
people and by agents alike. Its value grows with what is built on top of
it (ico-mcp's registry and crystallization) rather than with raw
throughput.

## 7. Order

1. **E** (hygiene) and **A** (plan key on events): small, independent,
   they simplify every later step.
2. **B** (telemetry off the hot path), measured with the profiler before
   and after on the CIFAR loader.
3. **C** (cancel first, then sampling), with the ico-mcp side
   (`cancel_run` without kill, a `sample` tool).
4. **D** (states), once B and C have moved activity reporting to events.
5. **F** and **G**, item by item, each when a real pipeline needs it.
6. **H** when concurrent runs of one flow become a real need.

## 8. Non-goals

- A distributed scheduler or cluster execution of our own; integrate
  instead.
- Replacing orchestrators (schedules, retries across runs, assets).
- New extension points without a second real use (`CLAUDE.md`, YAGNI).

## 9. Open questions

1. Should the plan key be the only identity in events (dropping `trace`
   from the public event), or live beside it for a release?
2. Cancellation granularity: between items only, or also a token that
   long operators can check inside one item (e.g. a long video encode)?
3. Sampling and privacy: previews of items can contain data that should
   not leave the process; should sampling be opt-in per flow?
4. Is concurrency of one flow (H) needed soon enough to start it before G?

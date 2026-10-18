# Profiler: design

Status: items 1 and 2 of section 10 implemented (2026-10-18); items 3–4
are proposals. Ports the idea of apriori3d/ico#17 (branch
`feature/profiler` and its stash) onto current `main`.

## 1. Goal

Answer "where does the time of this run go?" per node of a flow, including
nodes inside `MPWorker` runner processes and async stream pools, with an
overhead small enough to profile real runs, and zero overhead when the
profiler is off.

Kept from #17:

- a flow is instrumented **before activation**: the profiler changes a
  built, not yet activated flow, and replaces the flow factories of workers,
  so every runner builds an instrumented flow when it activates;
- metrics are pluggable;
- each runner process reports its measurements as runtime events.

Changed:

- **Instrumentation in place** instead of wrapper nodes.
  - The flow tree keeps its nodes, types, names and children: `describe`,
    `plan_view`, signatures and specs do not change.
  - A probe replaces the callable a node already calls (`_fn` / `fn`).
- **One aggregator per process** instead of a runtime node per operator.
  - Probes write into a table of the process.
  - In a runner process, one runtime node sends the table as events.
  - There is no per-node state machine (Pending / Ready / Idle) and no setup
    command.
- **Mergeable aggregates** instead of lists of values: memory is constant
  per node, and the aggregates of one node from several processes or
  threads merge exactly. No numpy.
- **A collecting tool and a report**, which #17 did not have yet.

## 2. Layers

| Layer | Where | Rule |
|---|---|---|
| measure | every call / item, in the operator's thread | ~1 µs, no allocation per metric, no shared state |
| aggregate | per node key, per thread shard | constant memory, mergeable |
| transport | runtime events from runner processes | throttled cumulative snapshots and a final one; never per call |
| present | `ProfilerTool` callback, Rich plan column, Live Board | outside the hot path |

## 3. Measurement

### 3.1 Metrics

```python
class IcoMetric(Protocol):
    name: str          # "wall", "cpu"
    unit: str          # "s"
    def start(self) -> int: ...
    def stop(self, token: int) -> int: ...   # value in integer base units (ns)
```

- **Stateless.** The token returned by `start` lives on the caller's stack,
  so a metric is safe across threads and for re-entrant calls. This fixes
  `CpuTimeMetric._start_time` in #17.
- **Built-in metrics:**
  - `WallTime`: `time.perf_counter_ns()`;
  - `CpuTime`: `time.thread_time_ns()`, the CPU time of the calling thread.
    `process_time` would add the work of other threads (async stream
    pools); `time.time()` (#17) is wall time and not monotonic.
- The default set is `(WallTime(), CpuTime())`.
- Other metrics are domain plugins and are not part of this design: GPU time
  (CUDA events need explicit synchronization), `tracemalloc` (expensive).
- Peak RSS is a per-process gauge (`resource.getrusage`), sent with every
  process snapshot, not a per-call metric.

### 3.2 Events: calls and items

What counts as one event of a node:

- A node whose callable returns a value: **one call**.
- A node whose callable returns an `Iterator` (source, stream, async stream,
  batcher, pipeline): the call itself is cheap, and the work happens in
  `__next__`.
  - The probe wraps the returned iterator, and **each `next()` is one item
    event**.
  - The final `next()` that raises `StopIteration` adds its time to the
    totals, but is not an item.
  - The wrapper delegates `__length_hint__`, so stream progress totals
    (apriori3d/ico#40) keep working.

  Calling a generator function runs none of its body. The body runs inside
  `next()`, from one `yield` to the next, in the thread that calls
  `next()`. While the generator is suspended, its consumer works on the
  item, and no frame of the generator is open. For a source that reads:

  ```
  next #1    open + read + decode   → startup (setup before the first yield)
  next #2..N read + decode           → items
  next #N+1  close → StopIteration   → end (cleanup, not an item)
  ```

  - **`startup`** is the first `next()` of each iterator, aggregated
    separately: setup (opening a file, loading a model) would otherwise
    distort the max and p99 of the items. A provider that returns a list
    (`yield from load_all()`) shows nearly all its time as `startup`.
  - **`end`** is the time of the final `next()`.
  - An iterator that is not consumed to the end is closed by the garbage
    collector, and that time is not counted.
- A call or `next()` that raises an exception counts in `errors`. Its time
  goes to the totals, but not to the distribution, so failures do not
  distort percentiles.

### 3.3 Self time: a thread-local stack

Each thread keeps a stack of open probe frames. A frame accumulates the
time of the frames opened inside it on the same thread. On exit:

```
self = elapsed - nested;  parent.nested += elapsed
```

This is the dynamic nesting that cProfile uses. It is what makes lazy
pipelines correct: when `stream.next()` pulls `source.next()`, the
source's time is nested in the stream's frame and subtracted from the
stream's self time, although the source is a sibling of the stream in the
chain, not its child.

- **Inclusive time** of a node is computed by the report: the sum of the
  self times of the node's subtree in the flow tree. For a stream it is the
  time of its body, not of its upstream.
- **Parallel nodes** (`IcoAsyncStream`): workers run in other threads
  (`asyncio.to_thread`) or processes, so their time does not nest. The
  report shows the stream's wall time per output item and the **pool
  utilization**: the busy time of its members (the duration of their
  calls) divided by the stream's active time (from the first to the final
  `next()` of each of its iterators) × `pool_size`. Both are measured with
  `perf_counter_ns` within one process, independently of the metrics. Low
  utilization means the input is the bottleneck; high utilization means
  the workers are.
- **`IcoWorker` portal:** the portal's call time minus the runner's root
  time for the same calls is the transfer overhead (pickling, queues),
  reported per worker key.
- **Async operators** (`IcoAsyncOperator`): wall time only, without self
  time or CPU time, because a coroutine suspends across `await`.

### 3.4 Probes

`instrument(flow, profiler)` walks the flow and, for each node:

- `IcoOperator` and subclasses (chain, stream, source, sink, filter,
  pipeline, worker portal, …): sets the public hook `operator.probe`
  (`ico.core.probe.IcoProbe`). `IcoOperator.__call__` calls
  `probe.call(self._fn, item)` when a probe is set, else `self._fn(item)`.
  Every node calls its children through `child(item)`, so composites are
  measured without special cases.
- `IcoContextOperator`: `probe.call_with_context(fn, item, context)`.
- `IcoAsyncOperator`: `await probe.call_async(fn, item)` (wall only).

The hook is used instead of replacing `_fn`, because `describe`, the plan
renderers and signature checks inspect `_fn` (its `__self__`, whether it
is a bound method). With the hook, `_fn` stays the original. Without a
probe, the cost is one attribute check per call.
- `IcoWorker` (and nested workers): sets the public hook
  `worker.runner_probe` (`ico.core.probe.IcoRunnerProbe`) to a pickleable
  `RunnerProfiler(root_key, metrics, interval)` (section 4). The runner
  builds its flow from `flow_factory` and runs `runner_probe.wrap(flow)`.
  `flow_factory` stays, so the worker's signature and the plan with remote
  flows do not change.

Both hooks are public attributes, documented as hooks for tools and `None`
normally: tools set them from other modules, which a private attribute
would not allow by convention.

Properties:

- Instrumenting twice into one profiler changes nothing; another profiler
  takes the nodes over (its probes replace the earlier profiler's); probes
  of other tools are kept.
- It mutates the given flow, so a profiled run needs a flow of its own,
  without nodes shared with other flows. Registered functions are shared
  nodes of the registry: `ico.spec.run(..., profiler=...)` builds the flow
  fresh (`registry_context(fresh=True)`), so the registry stays
  uninstrumented. No `uninstrument` (YAGNI).
- Tests check that `signature`, `describe` and `plan_view` of an
  instrumented flow equal those of the original.

## 4. Keys and processes

### 4.1 Node keys

A metric key is the path of the node in the **plan walker's** tree
(`create_plan_walker(expand_remote_flows=True)`), as dotted child
indices:

- A pool built from a factory is flattened to its first member, as the
  plan does. All members of a factory pool therefore have one key, and
  their aggregates merge into one row of the plan.
- The remote flow of a worker is the worker's last child, as in the plan
  walker.

`plan_view` gets a parallel list `keys: list[str | None]`, one per row
(`None` for footer rows such as `yield`). A consumer then joins profile
rows to plan rows without knowing node kinds. Progress keys stay runtime
paths; they are unrelated.

### 4.2 Runner processes

`RunnerProfiler.wrap(flow)` runs in the runner process:

1. It instruments the flow the runner built with a table of this
   process. Keys are prefixed with the worker's key, so they match the main
   process's plan. Nested workers are instrumented the same way.
2. It returns the flow under `ProfileHost`: an `IcoOperator` (it passes
   items through to the flow) that is also an `IcoRuntimeNode`.
   `IcoRunner` discovers it in the flow (`discover_and_connect_runtime_nodes`)
   like `IcoProgress`.
3. `ProfileHost` bubbles `IcoProfileEvent(layout, nodes, peak_rss)`, which
   carries this process's cumulative aggregates, the layout they describe
   (labels, pools, workers) and the peak RSS:
   - after an item, at most every `interval` seconds (default 1 s);
   - once more on deactivation (the final one).

   Events are sent from the runner's main thread only, between items, so
   they never race with items on the worker channel. Pool threads inside
   the runner never send.

The runner's root flow is wrapped only inside the runner process, so the
main process's plan is unchanged.

Snapshots are **cumulative**: a lost or throttled snapshot loses nothing,
and the tool keeps the latest snapshot per source (the event's trace path,
unique per runner) and sums over sources.

The final event reaches the tool: a runner sends it while handling the
deactivate command, and the worker waits for the acknowledgement of that
command, handling the events that come first. Workers deactivate before
the tools (post-order, the toolbox is the last child of the runtime). Tests
with `MPWorker` and `interval=None` check this: all runner measurements
come from the final event.

### 4.3 The main process

Probes in the main process write into the table of the `ProfilerTool`
directly; no events are needed.

## 5. Aggregation

Per `(node key, metric)`, in a shard per thread:

- `count`, `sum`, `min`, `max`;
- `mean` and `M2` (Welford). Shards and processes merge with Chan's
  parallel formula, so the std is exact after merging.
- A **log-bucket histogram**, as in Prometheus / HDR Histogram:
  - 4 buckets per octave from 1 µs, sparse (`{bucket: count}`);
  - about 10 % relative error for p50 / p95 / p99;
  - merges by summing buckets.
- Self-time aggregates are kept alongside the inclusive per-event ones,
  for wall and CPU time.

Per node: `calls`, `items`, `errors`, and for iterator nodes the
`startup` and `end` aggregates (section 3.2).

**Shards:** each thread writes its own shard without a lock. A snapshot
merges the shards under a registry lock. A shard has a lock taken only by
its own thread and by the snapshot, so it is uncontended. This matters for
thread pools, whose members share one key.

## 6. The tool and the snapshot

```python
profiler = ProfilerTool(on_profile=callback, metrics=(WallTime(), CpuTime()), interval=1.0)
flow = instrument(flow, profiler)            # before IcoRuntime(...)
IcoRuntime(flow, tools=[profiler]).activate().run().deactivate()
profiler.snapshot()                          # the final ProfileSnapshot
```

- `on_profile` is optional and receives throttled snapshots during the run
  (for the Live Board and ico-mcp), and a final one on deactivation.
- `ico.spec.run(..., profiler=None)`: when given, it instruments the built
  flow, not the closure around it, so keys match `plan_view(flow)`.

`ProfileSnapshot` (JSON-compatible, times in seconds):

```json
{"nodes": [
  {"key": "0.2.0", "label": "upper@1", "type": "IcoOperator",
   "calls": 120, "items": 0, "errors": 0,
   "metrics": {
     "wall": {"total": 3.1, "self": 2.9, "count": 120, "mean": 0.026,
              "std": 0.004, "min": 0.018, "max": 0.061,
              "p50": 0.025, "p95": 0.034, "p99": 0.05},
     "cpu":  {"total": 2.7, "self": 2.6, "count": 120, "...": "..."}},
   "startup": null, "end": null, "utilization": null, "transfer": null}],
 "metrics": [{"name": "wall", "unit": "s"}, {"name": "cpu", "unit": "s"}],
 "elapsed": 4.2, "overhead": 0.0005,
 "processes": [{"source": "main", "peak_rss": 512000000}], "final": true}
```

- `total` and `self` cover all events of the node; `count` and the
  distribution cover its calls (a node returning values) or its items after
  the first (a node returning iterators).
- `startup` / `end` (iterator nodes): `{metric: stats}` of the first and the
  final `next()`.
- `label` is the registry reference, else the node name, else the function
  name, else the class name.
- `utilization` (async streams) and `transfer` (workers) are `null` for
  other nodes.
- `processes`: `[{"source": "main", "peak_rss": …}, {"source": "<runtime
  path of a runner's host>", "peak_rss": …}, …]`.
- The inclusive subtree time is computed from the plan tree by the report.

## 7. Presentation

- **Rich:** a `profile` column set for the plan renderer, which prints a
  self / total / p95 / share column next to each plan row. Profiling then
  looks like the familiar plan. The hottest node by self time is
  highlighted.
- **Live Board:** `BoardRun.profile(snapshot)`, and the plan table shows
  the same columns, joined by `keys` (section 4.1).
- **ico-mcp** (a separate change there):
  - `run(profile=true)`;
  - the kernel attaches the tool and streams `profile` notifications to
    the board;
  - the trace keeps the final snapshot;
  - the result gets the top nodes by self time.

## 8. Overhead

- Off: one attribute check per call (about 0.13 µs for a whole operator
  call on CPython 3.10).
- On: one probe frame per event, with two metrics, takes about 4 µs on
  CPython 3.10 (four clock reads, a stack push and pop, one shard update
  under an uncontended lock). This is
  negligible for operators at the millisecond scale (models, I/O, images),
  and noticeable for micro-operators such as `upper` on words.
- The profile reports the total probe overhead it estimated (calibrated
  once per process on an empty call), so a reader can judge the numbers.
- Tests do not assert the per-event cost; `overhead` in the snapshot shows
  it for every run.

## 9. Out of scope (YAGNI)

- Sampling of calls.
- A timeline or trace export (Chrome trace / Perfetto spans).
- OpenTelemetry.
- `cProfile` of one node.
- GPU and memory metrics in the core.
- `uninstrument`.
- Profiling across `IcoFork` branches running concurrently: they run
  sequentially today and nest correctly.

The metric protocol and the snapshot leave room for each of these.

## 10. Plan of changes (ICO, then ico-mcp)

1. **Core profiler, main process only.**
   - What: `ico.tools.profiler` with metrics, probes, the thread-local
     stack, iterator wrapping, aggregates and histograms, `instrument`,
     `ProfilerTool` and `ProfileSnapshot`; `spec.run(profiler=...)`.
   - Tests: nesting and self time in a lazy chain, items and errors,
     length hints kept, merging of shards, thread pools, unchanged
     describe / signature.
2. **Remote flows.**
   - What: `RunnerProfiler`, `ProfileHost`, `IcoProfileEvent`,
     merging by source, peak RSS, worker transfer overhead, pool
     utilization.
   - Tests: `MPWorker`, factory pools, a nested worker, the final snapshot
     on deactivation.
3. **Presentation.**
   - What: `keys` in `plan_view`, the Rich profile columns, Live Board
     columns.
4. **ico-mcp:** `run(profile=true)`, notifications, trace, top nodes in the
   result.

#17 is closed after item 2, with a link to this document.

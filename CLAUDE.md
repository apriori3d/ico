# CLAUDE.md

## Project
ICO (Input → Context → Output) is a type-safe framework for composable
pipelines. Core lives in `src/ico/core`, runtime in `src/ico/core/runtime`,
spec export in `src/ico/spec`, tests in `src/tests`.

## Rules
- Every new operator or combinator comes with tests in `src/tests`.
- All code in `src/ico` and `src/tests` must pass `pyright` in strict mode
  (see `pyrightconfig.json`).
- Public API types must never be inferred as `Any`.
- Keep backward compatibility: existing tests and examples must keep passing.
- Core (`ico`) depends only on `rich`. Anything else goes to an optional
  extra in `pyproject.toml` and is imported lazily.
- One task per PR, small and focused.
- All code, comments, docstrings, commit messages and PR descriptions
  are written in English.
- YAGNI: implement only what the current item and its tests need. No new
  extension points, registries or allowlists until there is a second real
  use case. If a design grows beyond the item, stop and ask.

## Workflow
1. Run tests and pyright before starting to confirm a clean baseline:
   `make test` and `poetry run pyright src/ico src/tests`.
2. Implement the change with tests.
3. Run tests and pyright again; fix all failures before committing.
4. Describe in the PR what changed and why.

## Roadmap
1. [x] Move example-only dependencies to optional extras.
2. [x] Spec export (`ico.spec`): serializer registry keyed by node type,
       JSON only.
3. [x] Optional runtime signature check for `IcoChain` (strict mode).
4. [x] `fork` combinator with overloads for 2–6 branches; output is a tuple.
5. [x] `route`, `partition`, `filter` combinators (filter factory is `keep`).
6. [x] `until` predicate and `max_iterations` for `IcoProcess`.
7. [x] `on_error` policy (`raise` / `skip` / `collect`) and `retry` for
       `IcoAsyncStream`; fix `in_queue.task_done()` called once per worker
       instead of once per item.
8. [x] Operator registry:
   - `@register(name, version, tags=..., caps=...)` attaches metadata to an
     operator; description comes from the docstring, signature from
     `.signature`. Status: `draft` / `verified` / `stable` (default `draft`).
   - `IcoRegistry` stores entries keyed by `name@version` and returns
     compact cards (name, version, sig, description, tags, caps, status).
   - Search by input type, output type, tags and text. Type matching:
     exact match or subclass for plain classes; generics match by origin
     and args (no variance yet).
   - `build(spec)` resolves `name@version` refs through the registry and
     rebuilds the flow; round trip `to_spec → build → to_spec` must be stable.
   - Out of scope: MCP server, acceptance gate, PR flow, entry points.
9. [x] Agent-facing registry use instead of `find_path`: `search` finds
   direct converters by input and output type, agents see `entry.card()`;
   errors are located by a spec-tree `TreePathIndex` and name both refs and
   both types, e.g.
   `at 0.2 (tryon.segment@1 → tryon.render@1): expected Mask, got Image`;
   every `build()` error is a `SpecError` with `path`, `expected`, `got`,
   `refs` (strict chains: `at chain step N`). `find_path` is deferred, see
   `docs/design/registry.md`.
10. [x] Ignore generated spec files in git: they are named `*.spec.json`
    (e.g. `cifar_data_flow.spec.json`); committed fixtures use another name.
11. [x] Push the branch, open a PR, merge into `main` before starting MCP
    (apriori3d/ico#18, merge commit e2c25d9).
12. [x] `load_operators(path)`: import operator modules from a directory and
    register them, so a server can rebuild its registry on restart and
    load operators proposed by an agent. See `docs/design/registry.md`,
    "Loading operators from a directory":
    - the directory is a package on `sys.path`; modules are imported by
      name (needed for stable refs and pickling to spawn workers);
    - errors are collected per file in a report; `strict=True` raises;
    - re-registering `name@version` is an error; `replace=True` only for
      `draft` entries (re-imports that one module);
    - no sandboxing: the docstring warns that importing executes code;
      isolation belongs to `ico-mcp`.
13. [x] `ico.spec.run(spec, input=None, *, max_items=...)` in core (no new
    dependencies; `ico-mcp` and a future CLI are thin wrappers). JSON
    boundary, see `docs/design/spec.md`, "Running a spec":
    - `input` is a JSON value or `None` (flow starts with a source whose
      params are in the spec). No decoding of JSON into domain types: if a
      flow needs `Image`, its first step is a loader from `str`.
    - Data never passes through the agent: inputs and outputs are
      references (paths, URIs), not bytes.
    - The output must be JSON-compatible; otherwise raise `SpecError` with a
      hint to end the flow with an operator that writes results and returns
      paths.
    - Iterator outputs are collected up to `max_items`; the result reports
      the first N items and the total count.
    - Per-item errors (`on_error="collect"`) are returned with spec paths.
    - Result is a small dataclass with `to_dict()`: outputs, total,
      truncated, errors.

14. [x] Rename the agent/worker pair and runtime tools (before `ico-mcp`,
    where "agent" means the LLM agent and "tool" means an MCP tool).
    Full rename everywhere: code, tests, README, examples, notebooks,
    docstrings and `docs/design`. No deprecated aliases. Pure rename, no
    behavior changes; one PR.
    - `IcoAgent` → `IcoWorker` (local side: operator in the flow, manages
      the runner)
    - `MPAgent` → `MPWorker`
    - `IcoAgentWorker` → `IcoRunner` (remote side: executes the flow)
    - `AgentStateModel` → `WorkerStateModel`,
      `AgentWorkerStateModel` → `RunnerStateModel`
    - modules: `core/runtime/agent.py` → `core/runtime/worker.py`,
      `runtime/agent/mp/` → `runtime/worker/mp/`
    - spec kind `"agent"` → `"worker"`
    - `IcoTool` → `IcoRuntimeTool`, `IcoToolBox` → `IcoRuntimeToolBox`
      (runtime observers, not MCP tools); related names such as
      `IcoToolRegistrationProtocol` follow the same rule
    - After the rename, `grep -ri "agent"` and `grep -rw "IcoTool"` over
      `src/`, `README.md` and `docs/` must return nothing except intended
      uses. List any remaining hits in the PR description.
    Propose the full list of renames first; ask if a new name collides with
    existing code.

15. [x] Progress for external consumers: `JsonProgressTool`, a runtime tool
    like `RichProgressTool` that renders nothing and instead emits
    throttled JSON progress snapshots to a callback, one per progress node
    update: `{"key": "<TreePathIndex>", "label": "...", "done", "total",
    "rate", "eta"}`. No aggregate across nodes: progress nodes are not
    reliably nested (parallel workers count the same items), so turning
    snapshots into MCP progress notifications is decided in `ico-mcp`.
    Labels come from the flow at registration, as in `RichProgressTool`:
    the progress node name, else `"Progress <n>"` by registration order.
    `TreePathIndex` is only a stable key; no mapping to spec paths.
    Throttle per node (default ≤ 2 updates/s) and always emit the final
    state.

16. [x] ICO Live Board: a browser dashboard in core, the browser
    counterpart of the console rendering (`describe` + `RichProgressTool`).
    - JSON backend for `describe` (plan view model), next to `RichText`:
      a render-neutral tree built by the existing per-node-type renderer
      registry. Each node: `label`, `sig`, `ref`, `badges` (e.g. `pool ×4`,
      `until`, `on_error=skip`), `layout` (`sequence` / `parallel` / `loop`
      / `leaf`), `children`. JSON-compatible, with a schema test.
    - One HTML + JS page (scripts bundled, no CDN): progress bars on top,
      one per progress node, like the console (label, done/total, rate,
      ETA, state), and below them the pipeline graph rendered generically
      from the `describe` JSON. The page knows no node kinds.
    - A small HTTP server on the stdlib (`http.server`, no new
      dependencies), bound to `127.0.0.1`, one-time token in the URL,
      serving the page and two JSON endpoints: the plan (`describe`) and
      the latest progress snapshots. Data comes from a source interface,
      so another process can host the board with its own source
      (`ico-mcp` hosts it in its server process, fed by kernel
      notifications).
    - `LiveBoardTool`: a runtime tool that plugs into `IcoRuntime` like
      `RichProgressTool`, collects progress (reuse `JsonProgressTool`
      snapshots), starts the board and prints the URL.
    - Demo and validation: the existing multiprocessing CIFAR example
      `src/examples/ml/cv/cifar/ico_cifar_complete_flow_mp.py` with the
      board instead of (or next to) the console progress; progress from
      worker processes must appear on the board.

Next: start `ico-mcp` (separate package) on top of items 10–16.

Findings of the first domain (video cleaning on cube; numbers are those of
`docs/proposals/2026-10-17-first-domain-session.md`, now in the
`ico-mcp-video` repository). One PR each, in this order:

17. [ ] A worker survives an item error (finding 10): after an exception
    in an item the portal leaves the worker in Fault, so every later item
    fails with "Cannot transition to Sending state from non-Ready state"
    (28 errors for one). A worker whose runner is alive goes back to Ready.
18. [ ] `async_stream` behind a slow synchronous upstream (findings 11, 13,
    18): the pool dispatches nothing while the scheduler blocks pulling
    the next input (CPU workers idle, then one burst at the end, progress
    at 0), and two `async_stream`s in a row fail ("Cannot run the event
    loop while another loop is running"). Pull the input in a thread of
    its own; one design for both, for GPU → CPU pipelines.
19. [ ] Lengths flow through the chain (findings 3b, 14, 19, 20, 23):
    a 1:1 `stream` and `filter` pass their input's `__length_hint__` on,
    reducers can give one, a progress node takes its total from the item
    (`"total": "len"`), and `total: null` means "unknown", not an error.
20. [ ] `profile=true` breaks imports in an `MPWorker` (finding 9: scipy
    docscrape while importing diffusers in the first call of a factory
    pool). Reproduce first.
21. [ ] Design, then implement (algebra, proposal G): `pipeline_ref`, a
    saved pipeline as a node of another (finding 16; with ico-mcp), and
    `tap`, a child run for its side effect that passes its input on
    (finding 17).

## Deferred (after MCP)
Do not implement until a real task hits the limitation. Designs are in
`docs/design/spec.md`.
- `build(spec)` for context nodes (`IcoEpoch`, `IcoContextPipeline`,
  `IcoContextOperator`). Revisit when an agent needs to rebuild a training
  flow.
- Resources in specs (`resources` section, `ResourceRef[T]`, `{"$res": name}`,
  lazy open, scopes). Revisit when parametric operators with lazy loading
  are not enough, likely the first GPU-model pipeline.
- Registered contexts, `IcoStateful`, `checkpoint` node. Revisit together
  with context nodes.
- Artifact handles (URI + content hash). Revisit together with caching.
- `live` / `pinned` modes for non-deterministic sources.
- Shared resources across workers (`IcoShareable`, scope `shared`, memmap
  artifact store). Revisit when per-worker loading cost is a real problem.
- `find_path` (see `docs/design/registry.md`).

### Known issues (deferred)
Found while validating the CIFAR example; not fixed yet.
- `MPWorker` shutdown waits for the join timeout when the runner returned a
  torch tensor: the runner process exits only after the parent's
  `join(timeout=5)` returns, so stopping a pool of N workers takes ~5 s × N
  (CIFAR, 4 workers: run 1 s, stop 20 s). Workers returning plain values
  (ints, or with torch only imported) stop in ~0.1 s. Not caused by
  progress events or `RichProgressTool`; closing the parent channel before
  join does not help. Revisit with the first tensor pipeline run through
  `ico-mcp`, where it adds latency to every `run()`.
- A parent that raises or exits without `deactivate()` hangs at interpreter
  exit: runner processes are not daemons and block on their input queue,
  and `multiprocessing` joins them. `ico.spec.run` is safe (`deactivate()`
  in `finally`); other callers are not. Revisit together with the shutdown
  issue (daemon runners or an exit hook).
- Signature inference of `Any`-annotated callables depends on the Python
  version: uninferred on 3.10, inferred as `Any → Any` on 3.11+ (where
  `typing.Any` is a class), so strict checks and `validate_signatures` can
  differ between versions. Revisit when the difference shows up in a real
  flow or before declaring the behavior.

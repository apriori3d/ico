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

Next: start `ico-mcp` (separate package) on top of items 10–15.

## Deferred (after MCP)
Do not implement until a real task hits the limitation. Designs are in
`docs/design/spec.md`.
- JSON backend for `describe` (plan view model), next to `RichText`:
  a render-neutral tree built by the existing per-node-type renderer
  registry. Each node: `label`, `sig`, `ref`, `badges` (e.g. `pool ×4`,
  `until`, `on_error=skip`), `layout` (`sequence` / `parallel` / `loop` /
  `leaf`), `children`, and `progress_key` for progress nodes, equal to
  the key used by `JsonProgressTool` (matched by node identity in the
  same built flow). Consumers render it without knowing node kinds:
  the `ico-mcp` dashboard first, later a Jupyter view or Live Board.
  JSON-compatible, with a schema test.
  Do it together with the `ico-mcp` asynchronous runs and dashboard
  (items 8–9 of the `ico-mcp` roadmap): it is only needed there.
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

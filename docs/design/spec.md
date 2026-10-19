# Pipeline specs: design notes

Specs (`ico.spec`) describe a flow as JSON: node structure plus JSON params
of registered operators. This document records design findings and the
planned direction for what a spec cannot carry yet.

## Crystallization: the CIFAR reference example

`src/examples/ml/cv/cifar/cifar_ops.py` is the reference example of
*crystallization*: code that works in-process but cannot be described by a
spec is rewritten as registered, parametric operators.

| Original (`ico_cifar_complete_flow_mp.py`)          | Crystallized (`cifar_ops.py`)               |
| --------------------------------------------------- | ------------------------------------------- |
| `get_train_indices` / `get_val_indices` closures    | `cifar.indices` (`split`, `train_ratio`, `seed`, `limit`) |
| `fetch_item`, a local function over a loaded dataset | `cifar.fetch` (`root`, `delay`), dataset opened lazily |
| transform instances built in a factory              | `cifar.hflip`, `cifar.brightness`, `cifar.contrast` (`p`, `factor_max_delta`) |
| module-level `shuffle_indices`, `collate`           | registered as they are                      |

Rules this example establishes:

- Configuration captured by closures becomes constructor kwargs, stored in
  `NodeSpec.params` as JSON (sklearn convention: each kwarg is kept in an
  attribute with the same name).
- Heavy state is never created in `__init__`; it is opened lazily on first
  use or in `on_activate`.
- A local function is never referenced from a spec; it becomes a method of
  a registered operator class.

## Known limitation: specs carry no resources

A spec carries structure and JSON params, not resources. Heavy objects
(datasets, models, API clients) cannot travel in a spec, so they cannot be
shared across workers: in the crystallized CIFAR loader every process opens
the dataset itself, and the shared-memory dataset of the original example
is lost. This cost is accepted for now.

Training cannot be crystallized at all yet: the model, optimizer and device
live in the training context, which is built by code, not by a spec.

## Error locations

Errors about a spec point at the failing node with a `TreePathIndex` in
spec-tree coordinates: indices into `NodeSpec.children` from the root, as in
the exported JSON (chains flattened into `seq`, registered operators as
leaves). `spec.root.at(path)` returns the node.

```
at 0.2 (tryon.segment@1 → tryon.render@1): expected Mask, got Image
at 1: Cannot resolve operator 'tryon.missing@1': ...
at root: Cannot build node of kind 'epoch' yet.
```

- Every `build()` error is a `SpecError` with structured fields for tools
  such as an MCP server: `reason`, `path`, and for type errors `expected`,
  `got` and `refs` (upstream and downstream registry references).
- `validate_signatures(flow)` reports `SignatureIssue`s with the same paths:
  the path of an issue in a Python flow is the path of the failing node in
  `to_spec(flow)`.
- Strict mode checks a chain while it is being composed, before the final
  tree exists, so its path is relative to that chain and says so:
  `at chain step 2 (...)`.

`TreePathIndex` is used in two coordinate systems that must not be mixed:
the runtime tree (traces of runtime events: faults, progress, state, print;
keys of runtime tools) and the spec tree (build and validation errors). No
separate type is introduced for now.

Future step (not implemented): map runtime fault events to spec-tree paths,
so that an agent fixes a failure at run time the same way as a build error.
A fault trace locates a runtime node; translating it requires the mapping
from runtime nodes to the spec nodes they were built from.

## Direction: resources and contexts

Not implemented. This section is the agreed direction.

### Three kinds of things in a pipeline

| Kind          | What it is                              | Where it lives                       | CIFAR                     | API source              |
| ------------- | --------------------------------------- | ------------------------------------ | ------------------------- | ----------------------- |
| **Params**    | Plain JSON that configures behavior     | `NodeSpec.params`                    | lr, batch size, augmentations | report dates, fields |
| **Data**      | Values flowing through ports            | between nodes; large values as handles | batches, metrics        | report rows             |
| **Resources** | Heavy, process-local objects with a lifecycle | `resources` section, opened by the runtime | dataset, device   | HTTP client             |
| **Contexts**  | Resources with mutable, snapshottable state owned by one run | `context` of a context pipeline | model + optimizer + epoch | campaign optimization state |

The registry gets three entry kinds: `operator`, `resource`, `context`.
All are registered parametric classes; their params are JSON.

### Resources

```json
{
  "resources": {
    "train_ds": {"ref": "cifar.tensors@1", "params": {"root": "data"}, "scope": "process"},
    "direct":   {"ref": "yd.client@1", "params": {"token_env": "YD_TOKEN"}, "scope": "server"}
  },
  "root": {"...": "nodes reference resources as {\"$res\": \"train_ds\"}"}
}
```

- A node receives a resource by reference: params hold `{"$res": "name"}`
  and `build()` substitutes a `ResourceRef[T]`. The resource is opened
  lazily, so the spec stays JSON.
- `scope` defines the lifetime: `run` (one per run), `process` (one per
  process, the current per-worker cost made explicit), `server` (lives in
  the MCP server process across calls).
- Secrets never appear in a spec: only an environment variable name or a
  secret-store key.
- The resource type is part of the operator signature, e.g.
  `(ResourceRef[CifarTensors], int) -> CifarItem`, so chain checks and
  `find_path` see resources too.
- Resources are read-only for the nodes that use them.

### Contexts

A context is a resource with state that changes during a run and must be
saved and restored together.

```python
class IcoStateful(Protocol):
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...
```

In ICO a context is not a field of a node: it is a value that flows through
the main chain. In the CIFAR example `create_train_context` produces a
`CifarTrainContext`, `IcoProcess` passes it through every epoch,
`IcoContextPipeline` / `IcoContextOperator` update it per batch inside an
`IcoEpoch`, and `start_eval` / `end_eval` wrap it into a nested
`CifarEvalContext` and back. So "a context in a spec" means:

- the context class is registered (kind `context`) and implements
  `IcoStateful`;
- the operator that creates it becomes a registered parametric operator,
  e.g. `cifar.train_context` with params `arch`, `lr` and an optional
  `init` snapshot handle; it is the only place where the context is built;
- the nodes that move it (`IcoProcess`, `IcoEpoch`, `IcoContextPipeline`,
  `IcoContextOperator`) are serialized as structure (roadmap item 10);
- per-batch functions (`train_step`, `calculate_accuracy`, logging) become
  registered operators whose signatures name the context type.

```json
{"type": "seq", "children": [
  {"type": "op", "ref": "cifar.train_context@1",
   "params": {"arch": "resnet18", "lr": 0.001, "init": "ckpt://runs/abc/epoch_3"}},
  {"type": "process", "params": {"num_iterations": 10}, "children": ["..."]},
  {"type": "op", "ref": "cifar.finalize@1"}
]}
```

- The context is registered as a whole, not as separate model / optimizer /
  scheduler resources: it owns the construction order (optimizer needs model
  parameters, scheduler needs the optimizer) and guarantees a consistent
  snapshot.
- Light state (epoch, step, best metric) is part of the snapshot; otherwise
  resume breaks early stopping and best-model selection.
- A checkpoint is a snapshot of a context, not a separate concept. Resume is
  `init` pointing at a snapshot.
- Snapshots are written by a core `checkpoint` node (params `every`,
  `keep`, e.g. `"best:acc"`) that calls `state_dict()` and stores an
  artifact. Domain operators never implement saving themselves; the
  example's `save_checkpoint_step` is replaced by it.
- A context references resources but does not own them.
- A context never travels to an `MPWorker` runner process. `build()` rejects a spec
  where a node that needs a context ends up inside a worker.
- Nested contexts (`CifarEvalContext` holding the train context) are
  allowed; only the outermost context that crosses an iteration boundary of
  `IcoProcess` is snapshotted.
- Operators depend on structural protocols (`HasModel`, `HasOptimizer`,
  `HasEpoch`), not on a concrete context class, so `evaluate` works with any
  context that has a model.

### Validation: the full CIFAR training flow

The design is validated when the complete CIFAR flow of
`ico_cifar_complete_flow_mp.py` (data loader with a pool of `MPWorker`s,
training and validation epochs, `IcoProcess` over epochs) is exported to a
spec, rebuilt with `build()` in a fresh process, and trains with the same
behavior; resuming from a snapshot continues from the saved epoch and
step. The original example stays unchanged.

### Artifacts and handles

Large outputs (checkpoints, materialized reports) are stored as artifacts
and passed as handles: a URI plus a content hash (`ckpt://...`,
`artifact://...`). A handle can be used as the `init` of a context, which
chains runs without code. Agents (and MCP clients) see only names, handles
and metadata, never tensors or models.

### Non-deterministic sources

A source that reads from an external API returns different data over time,
so content-hash caching does not apply. Such operators are registered with
`deterministic=False` and run in one of two modes:

- `live`: call the API, store the result as an artifact, record its handle
  in the trace;
- `pinned`: replay from the recorded artifact.

A crystallized pipeline with a live input stays reproducible.

### Shared resources across workers

A read-only resource such as the CIFAR dataset is opened once and attached
by every `MPWorker` runner process without a copy.

```python
class IcoShareable(Protocol[R, H]):
    def export(self) -> H: ...            # owner: materialize, return a handle
    @classmethod
    def attach(cls, handle: H) -> R: ...  # worker: attach without copying
```

- New scope `shared`. The owner is the process that calls `build()`; the
  resource lives as long as the run (same lifetime as scope `run`).
- `build()` opens the resource and calls `export()`. `SpecFactory` carries
  `resources: dict[str, handle]` to workers, and a worker resolves its
  `ResourceRef` with `attach(handle)` instead of `open(params)`.
- A resource without `export` falls back to scope `process` with a warning.
- Contexts are never shared. Process-bound objects (sockets, HTTP clients)
  are not shareable either: their handle is just their config, scope
  `process`.
- Default handle: a memory-mapped file in the artifact store,
  `~/.ico/artifacts/<content-hash>/`, described by JSON (path, shape,
  dtype, hash). The OS page cache shares the pages between processes, no
  segment can leak, and the same handle works for scope `server` and as a
  cached artifact. `multiprocessing.shared_memory` is a later optimization.
- The artifact key is a hash of the resource ref, version and params, so a
  second run attaches to the existing file without materializing it again.

CIFAR: `export` writes the train split once as `uint8 (50000, 32, 32, 3)`
`.npy` (~150 MB instead of ~600 MB float32 per process); `attach` is
`np.load(path, mmap_mode="r")` + `torch.from_numpy`; conversion to float
moves into `cifar.fetch`, per item.

## Running a spec

`ico.spec.run(spec, input=None, *, max_items=...)` builds a spec and runs it
with a JSON boundary on both sides. It lives in core so that `ico-mcp`, a
CLI and tests share one implementation.

Rule: data never passes through the agent's context. An image as base64 is
thousands of tokens, and the agent does not need to see it to run a
pipeline. The agent passes references; the pipeline loads the data.

Input, two common shapes:

1. A path or a list of paths; the first operator loads the data:
   `run(spec, input="data/catalog_42/shirt_01.jpg")` with
   `load_image: str → Image` as the first step.
2. No input: the flow starts with a source whose params are part of the
   spec, e.g. `catalog_source(root="data/catalog_42")`; `run(spec)`.
   Re-running on another catalog changes only a param.

`run` does not decode JSON into domain types; that keeps core free of any
knowledge of how domain types are constructed.

Output follows the same rule: no images back to the agent. The last
operator writes results to disk and returns paths plus a short summary
(e.g. "48 passed, 3 sent to review"). A non-JSON output is an error with a
hint to add such an operator. Iterator outputs are capped by `max_items`
(first N items plus the total count) so that a large run does not flood the
agent's context. Per-item errors collected by `on_error="collect"` are
returned with their spec paths.

This is independent of resources (deferred): resources are heavy objects
used by operators (models, datasets); inputs are the data flowing through
the pipeline.

## Pipeline parameters

A saved spec fixes every param value; to run it again with another output
directory or threshold, a parameter replaces the value:

```json
{"kind": "op", "op": "cut_clip@1",
 "params": {"out_dir": {"$param": "out_dir"},
            "ffmpeg": {"$param": "ffmpeg", "default": "ffmpeg"}}}
```

- `input` is the data of a run; parameters are its configuration. Both are
  given per run (`run(spec, input, params={...})`, `build(..., params=)`);
  the spec stays one document.
- `ico.spec.bind` substitutes the values (and defaults) before the spec is
  built, so operators receive ordinary constructor arguments and know
  nothing of parameters. Parameters are not global state: they apply to
  one run.
- A parameter without `default` is required; `default` may be `null`. The
  same parameter may set several params if they declare it identically
  (same default; same type when the registry knows the operator param
  types). Unknown and missing names are `SpecError`s listing every name.
- `ico.spec.spec_params` lists what a spec accepts (name, type from the
  registry, default, paths). A built flow exported with `to_spec` carries
  the bound values: parameters belong to the spec document.
- Only whole param values: no substitution inside strings or nested
  values until a real need appears.

## Pipelines in pipelines

A pipeline made of other pipelines would otherwise copy their specs, so a
fix in one of them has to be made in every copy. A `pipeline_ref` node
stands for a saved pipeline instead:

```json
{"kind": "pipeline_ref", "name": "refine",
 "params": {"pipeline": "refine_masks",
            "bind": {"hold": 4, "work_dir": {"$param": "work_dir"}}}}
```

- `ico.spec.inline(spec, library)` replaces every reference by the root of
  the pipeline it names, recursively, before `bind` and `build`: building,
  workers, validation, plans, the profiler and samples see an ordinary
  spec. `library(name)` returns a pipeline's spec; where pipelines are
  stored is the caller's business (`ico-mcp` keeps them in files).
- `bind` gives values to parameters of the inner pipeline: constants, or
  references to parameters of the outer one. The inner parameters it does
  not bind become parameters of the outer pipeline under their own names,
  so a pipeline of pipelines exposes their configuration without repeating
  it; two references that lift one name with different defaults are the
  usual "declared differently" error. A `bind` key the pipeline does not
  declare is an error.
- The inlined root is named after the reference (its `name`, else the
  pipeline's), so a plan shows where a part came from. Errors inside it are
  located through the reference (`1.1` is the second child of the pipeline
  at `1`).
- Cycles (`a → b → a`) and nesting deeper than 8 are errors; a reference
  that reaches `build` without being inlined is an error that says so.
- `ico.spec.pipeline_refs` lists the pipelines a spec refers to directly
  (for a library that reports what uses what).

## Steps that hand over through files: `tap`

Some steps are worth their side effect: they write their real result next
to their input (masks, cleaned clips) and return a report, while the next
step needs the same input again. Without help, every report carries its
input and an adapter takes it out (`report → master`) between the steps.

`{"kind": "tap", "children": [child]}` (`IcoTap`, `tap(child)`) runs the
child on its input for the side effect and passes the input on: `I → I`,
where `I` is the child's input. An iterator output of the child is
consumed to the end first, so that its side effects happen; the output
itself is dropped (a step whose report is wanted is not tapped, usually
the last one). With `pipeline_ref`, a pipeline of steps on one input reads
as the steps themselves:

```
master → tap(scan) → tap(refine) → tap(clean) → assemble
```

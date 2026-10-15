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

```json
{"type": "context_pipeline",
 "context": {"ref": "cifar.train_context@1",
             "params": {"arch": "resnet18", "lr": 0.1, "dataset": {"$res": "train_ds"}},
             "init": "ckpt://runs/abc/epoch_3"},
 "checkpoint": {"every": 1, "keep": "best:acc"},
 "children": ["..."]}
```

- The context is registered as a whole, not as separate model / optimizer /
  scheduler resources: it owns the construction order (optimizer needs model
  parameters, scheduler needs the optimizer) and guarantees a consistent
  snapshot.
- Light state (epoch, step, best metric) is part of the snapshot; otherwise
  resume breaks early stopping and best-model selection.
- A checkpoint is a snapshot of a context, not a separate concept. Resume is
  `init` pointing at a snapshot.
- Snapshots are taken by the runtime according to the `checkpoint` policy,
  never by operators.
- A context references resources but does not own them.
- A context never travels to an `MPAgent` worker. `build()` rejects a spec
  where a node that needs a context ends up inside a worker.
- Operators depend on structural protocols (`HasModel`, `HasOptimizer`,
  `HasEpoch`), not on a concrete context class, so `evaluate` works with any
  context that has a model.

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
by every `MPAgent` worker without a copy.

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

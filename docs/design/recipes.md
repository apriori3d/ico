# Recipes: binding steps, artifacts and a content cache

Status: design, nothing implemented (2026-10-27). Roadmap items 41–45, one
PR each.

The first user is Apriori Studio, a layered image editor (`apriori-studio`,
its proposal `docs/proposals/2026-10-27-apriori-studio.md`): a product card
is a context (a base image, named masks, a layer stack), and its **recipe**
is a sequence of steps, each of which reads some fields of the context,
calls an ordinary registered operator and writes a field back. A person
edits one stroke; the recipe runs again from the top, and only what depends
on that stroke recomputes. Nothing in this design knows about cards: the
same pieces serve any flow that updates a context step by step.

ICO gets five things:

| Item | What | Module |
|---|---|---|
| 41 | artifact handles, a local store by content hash, codable values | `ico.artifacts` |
| 42 | a binding node: a step `C → C` around an operator `I → O` | `ico.core.bind`, spec kind `bind` |
| 43 | step events: what each step read, wrote and whether it was cached | `ico.core.bind`, `ico.spec.run` |
| 44 | code fingerprints and `deterministic` in the registry | `ico.registry` |
| 45 | a content cache with early cutoff; non-deterministic steps ask | `ico.cache` |

## 1. Codable values

A value is **codable** when it has a canonical JSON form:

- a JSON value (`None`, `bool`, `int`, `float`, `str`, lists, dicts with
  string keys);
- an `Artifact` (section 2): `{"$artifact": "sha256:…"}`;
- a tuple of codable values (a JSON list);
- an object with `to_json()` and a class method `from_json()`:

```python
class IcoCodable(Protocol):
    def to_json(self) -> JsonValue: ...

    @classmethod
    def from_json(cls, value: JsonValue) -> Self: ...
```

Its **content hash** is the sha256 of its canonical JSON (sorted keys, no
spaces, UTF-8). Because an artifact's JSON holds the hash of its bytes, the
content hash of a value that refers to files changes exactly when the bytes
change. One function gives three things: the cache key of a step's inputs,
the stored form of its output, and what a step event reports.

Domain types are not decoded by ICO: `from_json` belongs to the type. A
dataclass without `to_json` is not codable (no field-by-field guessing).

```python
@dataclass(frozen=True)
class Mask:
    """A binary mask at card size, in the frame of the raw it was made on."""

    artifact: Artifact          # PNG in the store
    frame: str                  # digest of that raw

    def to_json(self) -> JsonValue:
        return {"artifact": self.artifact.to_json(), "frame": self.frame}

    @classmethod
    def from_json(cls, value: JsonValue) -> Mask:
        data = cast(dict[str, JsonValue], value)
        return cls(Artifact.from_json(data["artifact"]), cast(str, data["frame"]))
```

`ico.artifacts` exports `to_json(value)`, `from_json(type, value)` and
`content_hash(value) -> str | None` (`None`: not codable).

## 2. Artifacts (item 41)

An artifact is an immutable file addressed by the sha256 of its bytes. A
handle is small and crosses every boundary (specs, JSON inputs and outputs,
worker processes, the cache); the bytes stay in a store.

```python
@dataclass(frozen=True)
class Artifact:
    digest: str                                   # 64 lowercase hex

    def path(self) -> Path: ...                   # through current_store()
    def to_json(self) -> JsonValue: ...           # {"$artifact": "sha256:<hex>"}
    @classmethod
    def from_json(cls, value: JsonValue) -> Artifact: ...


class ArtifactStore(Protocol):
    def put_file(self, path: Path) -> Artifact: ...
    def put_bytes(self, data: bytes) -> Artifact: ...
    def has(self, artifact: Artifact) -> bool: ...
    def path(self, artifact: Artifact) -> Path: ...
```

- `LocalStore(root)`: `sha256/ab/cd/<digest>` read-only files, written
  through `tmp/` with fsync and an atomic rename; putting the same bytes
  again writes nothing; `verify(artifact)` hashes the file again. Stdlib
  only. (It moves here from `apriori-studio`'s `layers.store`, where it was
  written first.)
- `current_store()`: the store of the current run (`run(store=...)`, kept
  in a context variable like the cancel token), else
  `LocalStore($ICO_ARTIFACTS)`, else an error that names the variable. The
  variable makes the store reachable from worker processes and from
  ico-mcp's kernels, which get it through `env` in `ico.toml`.
- An operator writes its result with `current_store().put_file(tmp)` and
  returns the handle; it reads its input with `mask.artifact.path()`.
  Pixels never go through a spec, an event or the cache.
- Not here: S3 or MinIO stores (a second implementation, when a deployment
  needs it), deletion and eviction, provenance (the run trace has it).

## 3. The binding node (item 42)

```json
{"kind": "bind",
 "name": "вычесть лямки",
 "params": {"in": {"target": "masks.рюкзак", "other": "masks.лямки"},
            "const": {},
            "out": "masks.рюкзак"},
 "children": [{"kind": "op", "op": "layers.mask_subtract@1",
               "params": {"grow": 3}}]}
```

- `IcoBind(child, inputs, consts, out)` has the signature `C → C`, where
  `C` is the type of the flow's context. The child is any operator `I → O`
  that knows nothing about the context.
- The context implements two methods; the class is the domain's (Apriori
  Studio: the composition of `layers`):

  ```python
  class IcoFields(Protocol):
      def get(self, path: str) -> object: ...                     # KeyError
      def replace(self, updates: Mapping[str, object]) -> Self: ...  # a new context
  ```

- **Input.** When `I` is a `NamedTuple` or a dataclass, its fields come
  from `in` (context paths) and `const` (values recorded in the step);
  every field is given exactly once. When `I` is one value, `in` is a
  single path string and `const` is empty.
- **Constants** are JSON in the spec, decoded with `from_json` by the
  declared type of their field: recorded strokes, a brush delta, a model
  result, a prompt. They belong to the step, not to the context.
- **Output.** `out` is a path; for an operator that returns a tuple, a list
  of paths of the same length.
- **Sequential.** A step reads a field as it is at its position in the
  sequence. A domain that wants another order (Apriori Studio: "a read of
  a mask is a read of its final state", no cycles between masks) compiles
  its recipe into an ordered `seq` of `bind` nodes; ICO runs the sequence.

The operator is an ordinary parametric operator:

```python
class SubtractIn(NamedTuple):
    target: Mask      # the mask being edited
    other: Mask       # the mask subtracted from it


@register("layers.mask_subtract", 1, tags={"mask"})
class MaskSubtract(IcoOperator[SubtractIn, Mask]):
    """``target`` minus ``other`` grown by ``grow`` px."""

    def __init__(self, grow: int = 3) -> None:
        self.grow = grow
        super().__init__(fn=self._apply)

    def _apply(self, x: SubtractIn) -> Mask:
        target = read_mask(x.target.artifact.path())
        other = dilate(read_mask(x.other.artifact.path()), self.grow)
        return write_mask(target & ~other, frame=x.target.frame)  # put into the store
```

**Validation before a run.** `validate_bindings(spec, fields)` walks the
`bind` nodes of a sequence with the types of the initial context fields
(`{"raw": CardImage, "masks.лямки": Mask, …}`, given by the domain):

- every `in` path exists at that point, and its type is assignable to the
  input field (the same `is_assignable` as link checks);
- every `const` decodes to its field type (a class with `from_json`);
- the output type is assignable to the field it overwrites; a new path
  becomes a field of that type for the steps below.

Errors are `SpecError`s located by spec path, as build errors:

```
at 6 (layers.mask_subtract@1): in 'other': masks.лямки is SoftMask, expected Mask
at 3 (layers.cut_by_strokes@1): in 'target': no field masks.лямки at this step
```

## 4. Step identity: spec paths, no ids in ICO

A recipe needs stable step ids (`s7`): its history, recorded inputs, review
flags, diffs and the rebase of one session onto another refer to a step
across edits, while positions shift when a step is inserted or the recipe
is reordered. These ids live in the domain's recipe format, not in
`NodeSpec`:

- inside one run, a step is fully identified by its spec path, which
  build errors and validation already use;
- the domain compiles its recipe into the spec it runs, so it knows the
  spec path of every step id **for that spec**, and translates back.

The cache does not need ids either: its keys are contents (section 6).

## 5. Step events (item 43)

Every `bind` node, when it finishes, emits an `IcoStepEvent` on the bus
(so it also arrives from worker processes):

| Field | Meaning |
|---|---|
| `path` | spec path of the step (kept by the node at build) |
| `node` | plan key, like every runtime event |
| `ref` | operator `name@version` |
| `key` | cache key (section 6), or `null` when not cacheable |
| `cache` | `"hit"`, `"miss"` (computed and stored), `"skip"` (no cache, or not cacheable: `reason`) |
| `out` | `{path: canonical JSON}` of the fields the step wrote |
| `seconds` | wall time of the step |

An event is a **delta**: the context after step *n* is the initial context
plus the deltas of steps 1…*n*. A domain shows "the card as it was after
this step" from them without storing whole snapshots.
`spec.run(..., steps=True)` returns them as `RunResult.steps`;
`StepRecorder` is the runtime tool behind it.

## 6. Fingerprints and non-deterministic operators (item 44)

- `RegistryEntry.fingerprint`: the sha256 of the sources of the module
  that defines the operator **and of every module it imports
  transitively, except installed distributions and the standard library**
  (the module file is outside `site-packages` and the stdlib). Editable
  checkouts (`layers` during development) and draft and stable domain
  packages are followed; a change in a helper module changes the
  fingerprint. ico-mcp's draft fingerprint (one module's text) misses such
  changes. Installed distributions are pinned by the lock file, whose hash
  the caller passes as the cache `salt`.
- `@register(..., deterministic=False)`: the operator gives a new result on
  every call (an image model). Shown on the card. Without a cache nothing
  changes: it is called as today.

## 7. Content cache with early cutoff (item 45)

```python
class IcoCache(Protocol):
    def get(self, key: str) -> JsonValue | None: ...
    def put(self, key: str, value: JsonValue) -> None: ...
```

`DirCache(root)` keeps one JSON file per key; nothing is evicted. The key
of a step:

```json
sha256({"ref": "layers.mask_subtract@1", "fingerprint": "9c1f…",
        "params": {"grow": 3},
        "inputs": {"target": "<content hash>", "other": "<content hash>"},
        "salt": "<hash of uv.lock>"})
```

A `bind` node with a cache:

1. computes the key; any input that is not codable → `skip`, call as
   today;
2. **hit**: decodes the stored output with `from_json` by the operator's
   output type; the operator is not called;
3. **miss**: calls the operator and stores `to_json(output)`; an output
   that is not codable → `skip` with the reason.

**Early cutoff needs no graph.** A recomputed output with the same bytes
has the same artifact digest, so the same JSON and the same content hash,
so the steps below have the same keys and hit. Old entries stay, so undo
and switching back to an earlier version are hits too.

**Non-deterministic steps ask.** On a miss, a step whose operator is
`deterministic=False` runs only if its spec path is in `allow_calls`;
otherwise the run stops there with `ConfirmationNeeded(path, ref)` (a
structured error of `RunResult`). On a hit it reuses the recorded result:
a recorded model mask is reused while its inputs are the same.

The cache, the store, `allow_calls` and the salt are given per run and
reach the nodes through context variables, like the cancel token:

```python
run(spec, input, store=store, cache=DirCache(store.root / "cache"),
    salt=lock_hash, allow_calls={"1"}, steps=True)
```

## 8. How it is used: one recipe end to end

Apriori Studio compiles a recipe (its own JSON, with step ids) into a spec.
The first step decodes the context from the JSON input, the last one
encodes it again (the run's output must be JSON):

```json
{"version": 1, "root": {"kind": "seq", "children": [
  {"kind": "op", "op": "layers.composition_from_json@1"},
  {"kind": "bind", "name": "Seedream «лямки»",
   "params": {"in": "raw", "out": "masks.лямки"},
   "children": [{"kind": "op", "op": "marketplace.seedream_mask@1",
                 "params": {"object": "лямки"}}]},
  {"kind": "bind", "name": "по краям",
   "params": {"in": {"target": "masks.лямки", "image": "raw"}, "out": "masks.лямки"},
   "children": [{"kind": "op", "op": "layers.snap_edges@1", "params": {"band": 4}}]},
  {"kind": "bind", "name": "вырезать мех",
   "params": {"in": {"target": "masks.лямки", "image": "raw"},
              "const": {"strokes": {"artifact": {"$artifact": "sha256:81c2…"}, "frame": "a1f0…"}},
              "out": "masks.лямки"},
   "children": [{"kind": "op", "op": "layers.cut_by_strokes@1"}]},
  {"kind": "op", "op": "layers.composition_to_json@1"}]}}
```

with the map `{"1": "s1", "2": "s2", "3": "s3"}` from its compiler. It runs
it (in ico-mcp, the one place of execution; here the core call):

```python
compiled = compile_recipe(recipe)                 # layers: spec + {path: step id}
issues = validate_bindings(compiled.spec, composition_fields(card))
result = run(compiled.spec, input=card.to_json(), store=store, cache=cache,
             salt=lock_hash, allow_calls=set(), steps=True)
```

First run on a new card: step 1 is a miss of a non-deterministic operator,
so the run stops and the editor asks:

```json
{"errors": [{"reason": "confirmation_needed", "path": "1",
             "ref": "marketplace.seedream_mask@1"}]}
```

The person agrees (`allow_calls={"1"}`, one paid call); the run goes
through, every step a miss, and `result.steps` gives the history:

```json
[{"path": "1", "ref": "marketplace.seedream_mask@1", "cache": "miss", "seconds": 21.4,
  "key": "e03b…", "out": {"masks.лямки": {"artifact": {"$artifact": "sha256:5d1e…"}, "frame": "a1f0…"}}},
 {"path": "2", "ref": "layers.snap_edges@1", "cache": "miss", "seconds": 0.8,
  "key": "7a44…", "out": {"masks.лямки": {"artifact": {"$artifact": "sha256:c90a…"}, "frame": "a1f0…"}}},
 {"path": "3", "ref": "layers.cut_by_strokes@1", "cache": "miss", "seconds": 0.3,
  "key": "1f8d…", "out": {"masks.лямки": {"artifact": {"$artifact": "sha256:0b77…"}, "frame": "a1f0…"}}}]
```

The person redraws the stroke of s3. Only its constant changes, so steps 1
and 2 hit (the model is not called again, no confirmation), step 3 misses,
and every step that reads `masks.лямки` below misses only if the new mask
differs. If a new `band` of s2 gives the same pixels, s2 misses but s3 and
everything below hit: early cutoff.

## 9. Not in this design

- Frames and carry-over to a new generation (Apriori Studio: frames in its
  types, an alignment step inserted by its compiler); revisit in ICO if
  the binding has to insert adapters by itself.
- Fingerprint aliases (an operator promoted to another package keeps its
  cache entries and its old name): an ICO registry item of its own.
- An estimate before a run (what will recompute, which calls will ask):
  the domain walks its recipe; an ICO helper only when a second domain
  needs it.
- Parallel steps, remote steps (the store and the cache on a target),
  steps inside a stream; an S3 store.
- Disabled steps: the compiler leaves them out.

## 10. Order

41 → 42 → 43 → 44 → 45. Each is useful alone: artifacts for any flow that
hands files over, `bind` for any context flow, events for its history, and
the cache for everything above. The ico-mcp side (store and cache from
`ico.toml`, steps in run traces) is a separate ico-mcp item after 45.

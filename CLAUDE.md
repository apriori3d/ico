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
10. [ ] `build(spec)` for context nodes: `IcoEpoch`, `IcoContextPipeline`
    and `IcoContextOperator`, so that training flows (not only data
    loaders) can be rebuilt from a spec.
11. [ ] Resources in specs (see `docs/design/spec.md`): `resources`
    section, `@register` kind `resource`, `ResourceRef[T]`, `{"$res": name}`
    in params, lazy open, scopes `run` and `process`.
12. [ ] Registered contexts: register kind `context`, `IcoStateful`
    protocol, registered context factory operators with an `init`
    snapshot param, core `checkpoint` node. Validate on the full CIFAR
    training flow (see `docs/design/spec.md`, "Validation").
13. [ ] Artifact handles (URI + content hash) for checkpoints and large
    outputs.
14. [ ] `live` / `pinned` modes for non-deterministic sources.
15. [ ] Shared resources across workers: `IcoShareable` (`export` /
    `attach`), scope `shared` owned by `build()`, memmap handles in the
    artifact store `~/.ico/artifacts/<hash>/`. First case: CIFAR dataset.

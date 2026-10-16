# Operator registry: design notes

The registry (`ico.registry`) holds named, versioned operators
(`name@version`) with metadata, a signature and compact JSON cards. Agents
use it to find operators and to compose flows that `ico.spec.build()` turns
into running pipelines.

## How agents compose chains

An agent builds a chain step by step with `search`, seeing each found
operator as its compact card (`entry.card()`):

- `search(input=A, output=B)` finds direct converters `A → B`;
- `search(input=A)` lists operators that can consume `A`; the agent picks
  the step that fits the task, then searches again from its output type.

Two or three calls are usually enough, and at every step the agent applies
task semantics that type compatibility alone cannot express.

When a composed chain does not fit, the error locates the failing node in
the spec tree and names both registry references and both types, so the
agent can fix the exact link:

```
at 0.2 (tryon.segment@1 → tryon.render@1): expected Mask, got Image
```

`build()` raises it as a `SpecError` with structured fields (`path`,
`expected`, `got`, `refs`); `validate_signatures()` returns the same data in
`SignatureIssue`. See "Error locations" in `spec.md`.

## Loading operators from a directory

The registry lives in memory and is filled when a module with `@register`
is imported. A server process starts with an empty registry, while
operators proposed by an agent are files in a directory. `load_operators`
bridges the two:

```python
report = load_operators("operators")
report.loaded   # ["direct.fetch_report@1", ...]
report.errors   # [("operators/broken.py", "SyntaxError: ...")]
```

Used at server start (load everything accumulated so far) and after an
agent proposes an operator (load one new file).

Decisions:

- **Import as a package, by module name.** The operator directory is a
  package on `sys.path` (it has `__init__.py`), and modules are imported by
  name, e.g. `operators.fetch_report`. Loading files by path with
  `importlib` would give modules arbitrary names: `module:qualname` refs
  would not be reproducible, and pickling to `spawn` workers (`MPWorker`)
  would fail because the child process cannot import the module.
- **Collect errors per file.** One broken file must not take down the
  server and the rest of the registry. Errors are collected in the report;
  `strict=True` raises on the first error (for tests and CI).
- **Re-registering `name@version`.** An error by default. `replace=True` is
  allowed only for entries with status `draft`, so an agent can iterate on
  its draft. A `stable` operator can only be superseded by a new version:
  a spec that refers to `name@version` of a stable operator always gets the
  same code. Replacing a draft re-imports only that one module; there is no
  general hot reload of already imported modules.
- **Out of scope:** entry points from installed packages (deferred until
  the domain template), hot reload.

### Security

Importing an operator module executes its code with the full permissions
of the importing process. `load_operators` does not sandbox anything and
must not be used on untrusted code in a process that holds secrets or
data it must protect. Isolation is the responsibility of the caller:
`ico-mcp` must load and run agent-proposed operators in a separate process
or container, with only the capabilities declared in the operator's
`caps`. The `load_operators` docstring states this explicitly.

## Deferred: `find_path(from_type, to_type)`

Idea: search the registry type graph for the shortest operator chains that
turn `from_type` into `to_type`. Deferred because:

- a type-compatible path is not semantically meaningful; the agent still has
  to judge every step;
- a path search must forbid revisiting a type to avoid cycles, which
  excludes `T → T` steps; those are the most common pipeline steps
  (normalize, filter, dedupe), so the result would be a skeleton only;
- step-by-step composition with `search` covers the need in 2–3 calls.

Revisit when both hold:

- a real registry with many domain-specific types exists, and
- there is evidence that agents spend many calls discovering chains.

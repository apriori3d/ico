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

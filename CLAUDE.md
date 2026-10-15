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
3. [ ] Optional runtime signature check for `IcoChain` (strict mode).
4. [ ] `fork` combinator with overloads for 2–6 branches; output is a tuple.
5. [ ] `route`, `partition`, `filter` combinators.
6. [ ] `until` predicate and `max_iterations` for `IcoProcess`.
7. [ ] `on_error` policy (`raise` / `skip` / `collect`) and `retry` for
       `IcoAsyncStream`; fix `in_queue.task_done()` called once per worker
       instead of once per item.
8. [ ] `@register` decorator and operator registry with signature search.

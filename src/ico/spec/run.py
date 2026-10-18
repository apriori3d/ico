"""Run a spec with a JSON boundary on both sides.

`run` is the one implementation shared by `ico-mcp`, a future CLI and
tests. Data never passes through the caller (an agent): inputs and outputs
are JSON values, typically paths or URIs and short summaries, not bytes.
`run` does not decode JSON into domain types; a flow that needs an `Image`
starts with a loader from `str`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from ico.core.operator import IcoOperator, IcoOperatorProtocol
from ico.core.tree_utils import TreePathIndex
from ico.spec.model import JsonValue, NodeSpec, PipelineSpec, SpecError, is_json_value

if TYPE_CHECKING:
    from ico.core.runtime.tool import IcoRuntimeTool
    from ico.registry.registry import IcoRegistry
    from ico.tools.profiler import ProfilerTool

DEFAULT_MAX_ITEMS = 100


@dataclass(frozen=True, slots=True)
class RunError:
    """An item that failed in a stream with ``on_error="collect"``.

    Attributes:
        path: Spec path of the stream that collected the error, or ``None``
            if it cannot be located (e.g. a stream inside an MPWorker runner process).
        index: Position of the item in that stream's input.
        item: The input item (``repr`` if it is not JSON-compatible).
        error: ``"ErrorType: message"``.
        attempts: Attempts made, including retries.
    """

    path: TreePathIndex | None
    index: int
    item: JsonValue
    error: str
    attempts: int

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "path": str(self.path) if self.path is not None else None,
            "index": self.index,
            "item": self.item,
            "error": self.error,
            "attempts": self.attempts,
        }


@dataclass(frozen=True, slots=True)
class RunResult:
    """Result of :func:`run`.

    Attributes:
        outputs: The output, or the first ``max_items`` items of an iterator
            output; always JSON-compatible (tuples become lists).
        total: Number of output items (1 for a non-iterator output).
        truncated: Whether ``outputs`` holds fewer items than ``total``.
        errors: Per-item errors collected by ``on_error="collect"`` streams.
    """

    outputs: tuple[JsonValue, ...]
    total: int
    truncated: bool
    errors: tuple[RunError, ...] = ()

    def to_dict(self) -> dict[str, JsonValue]:
        outputs: list[JsonValue] = []
        outputs.extend(self.outputs)
        errors: list[JsonValue] = []
        errors.extend(error.to_dict() for error in self.errors)
        return {
            "outputs": outputs,
            "total": self.total,
            "truncated": self.truncated,
            "errors": errors,
        }


def run(
    spec: PipelineSpec | NodeSpec,
    input: JsonValue = None,
    *,
    registry: IcoRegistry | None = None,
    max_items: int = DEFAULT_MAX_ITEMS,
    tools: Sequence[IcoRuntimeTool] = (),
    params: Mapping[str, JsonValue] | None = None,
    profiler: ProfilerTool | None = None,
) -> RunResult:
    """Build a spec and run it once.

    Args:
        spec: The spec to build (see :func:`ico.spec.build`).
        input: JSON input of the flow; ``None`` for a flow that starts with a
            source whose params are in the spec.
        registry: Registry for ``op`` references; defaults to
            :data:`ico.registry.default_registry`.
        max_items: Maximum number of iterator output items to return; the
            iterator is still consumed to the end and counted in ``total``.
        tools: Runtime tools attached to the run, e.g. a
            :class:`~ico.tools.progress.JsonProgressTool` that reports
            progress to the caller.
        params: Values of the pipeline parameters of this run (see
            :mod:`ico.spec.params`); defaults apply to the others.
        profiler: Profiles the run: the built flow is instrumented (keys
            are paths in the plan of the spec's flow) and the tool is
            attached; read the profile with ``profiler.snapshot()``.

    Returns:
        A :class:`RunResult`.

    Raises:
        SpecError: If the spec cannot be built, or an output is not
            JSON-compatible (end the flow with an operator that writes
            results and returns paths or a summary).
        Exception: Any other error raised by the flow.

    Example:
        >>> result = run(spec, "data/catalog_42/shirt_01.jpg")
        >>> result.to_dict()
    """
    from ico.core.async_stream import IcoStreamErrors
    from ico.core.runtime.runtime import IcoRuntime
    from ico.spec.builders import registry_context
    from ico.spec.params import bind

    spec = bind(spec, params)
    root = spec.root if isinstance(spec, PipelineSpec) else spec
    context = registry_context(registry)
    flow = cast(IcoOperatorProtocol[Any, Any], context.build(root))
    if profiler is not None:
        from ico.tools.profiler import instrument

        instrument(flow, profiler)
        if profiler not in tools:
            tools = [*tools, profiler]

    outputs: list[JsonValue] = []
    total = 0

    def provide(_: None) -> JsonValue:
        return input

    def collect(value: object) -> None:
        nonlocal total
        items: Iterator[object] = (
            cast(Iterator[object], value)
            if isinstance(value, Iterator)
            else iter([value])
        )
        for item in items:
            total += 1
            if len(outputs) < max_items:
                outputs.append(_to_json(item))

    closure: IcoOperatorProtocol[None, None] = (
        IcoOperator(provide) | flow | IcoOperator(collect)
    )
    runtime = IcoRuntime(closure, tools=tools)
    errors: list[RunError] = []
    runtime.activate()
    try:
        runtime.run()
    except IcoStreamErrors as e:
        path = context.paths.get(id(e.stream)) if e.stream is not None else None
        errors = [
            RunError(
                path=path,
                index=item_error.index,
                item=_to_json_or_repr(item_error.item),
                error=f"{type(item_error.error).__name__}: {item_error.error}",
                attempts=item_error.attempts,
            )
            for item_error in e.errors
        ]
    finally:
        runtime.deactivate()

    return RunResult(
        outputs=tuple(outputs),
        total=total,
        truncated=total > len(outputs),
        errors=tuple(errors),
    )


def _to_json(value: object) -> JsonValue:
    """Convert an output to JSON: tuples become lists; anything else non-JSON
    is an error with a hint."""
    converted = _tuples_to_lists(value)
    if not is_json_value(converted):
        raise SpecError(
            f"output of type {type(value).__name__} is not JSON-compatible; end "
            "the flow with an operator that writes results and returns paths "
            "or a summary"
        )
    return cast(JsonValue, converted)


def _to_json_or_repr(value: object) -> JsonValue:
    converted = _tuples_to_lists(value)
    return cast(JsonValue, converted) if is_json_value(converted) else repr(value)


def _tuples_to_lists(value: object) -> object:
    if isinstance(value, tuple | list):
        return [_tuples_to_lists(v) for v in cast(tuple[object, ...], value)]
    if isinstance(value, dict):
        items = cast(dict[object, object], value).items()
        return {k: _tuples_to_lists(v) for k, v in items}
    return value

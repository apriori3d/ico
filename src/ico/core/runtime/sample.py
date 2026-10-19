"""Samples: previews of the results of a node, on request, while a flow
runs.

``IcoRuntime.sample(node, count)`` asks the node at plan key ``node`` (see
:mod:`ico.core.plan_keys`) for its next ``count`` results. Every node of a
flow is an operator with a probe slot; a node whose result is an iterator (a
source, a stream) previews as one, so sample the operator inside a stream to
see its items. The runtime and the runners of workers each take the request
for the nodes of their own flow: they set a :class:`SampleProbe` as the
node's probe, which
bubbles each result as an :class:`IcoSampleEvent` with a short preview and
gives the probe it replaced (a profiler's) back after ``count`` results.
Nothing is set until a sample is asked for.

The request reaches runners as an :class:`IcoSampleCommand` through their
control lanes, so a runner takes it at once, also in the middle of a long
item (a video whose frames stream inside the worker). Each process sends
at most ``count``
samples: a pool of several workers may send more in all.
"""

from __future__ import annotations

import dataclasses
import reprlib
import threading
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, cast, final

from ico.core.node import IcoNodeProtocol
from ico.core.plan_keys import plan_nodes
from ico.core.probe import IcoProbe
from ico.core.runtime.command import IcoRuntimeCommand
from ico.core.runtime.event import IcoRuntimeEvent
from ico.core.tree_utils import TreePathIndex

PREVIEW_CHARS = 400


@final
@dataclass(slots=True, frozen=True)
class IcoSampleCommand(IcoRuntimeCommand):
    """Sample the next ``count`` results of the operator at plan key
    ``node``."""

    acknowledged: ClassVar[bool] = False

    node: str
    count: int


@final
@dataclass(slots=True, frozen=True)
class IcoSampleEvent(IcoRuntimeEvent):
    """One result of a sampled operator (``node`` is its plan key)."""

    index: int
    """The number of the sample in its process, from 0."""
    type: str
    """The type name of the result."""
    preview: str
    """A short text of the result: see :func:`preview`."""

    @staticmethod
    def create(node: str, index: int, value: object) -> IcoSampleEvent:
        return IcoSampleEvent(
            trace=TreePathIndex(),
            node=node,
            index=index,
            type=type(value).__name__,
            preview=preview(value),
        )


_repr = reprlib.Repr()
_repr.maxstring = PREVIEW_CHARS
_repr.maxother = PREVIEW_CHARS
_repr.maxlist = _repr.maxtuple = _repr.maxset = _repr.maxdict = 8


def preview(value: object, max_chars: int = PREVIEW_CHARS) -> str:
    """A short text of ``value``: arrays and tensors by shape and dtype,
    iterators by type, dataclasses and Pydantic models field by field
    (each field shortened on its own, so that scalars after a large field
    stay visible), lists and dicts by their first items and length,
    anything else by a shortened ``repr``."""
    return _cut(_preview(value, 0), max_chars)


FIELD_CHARS = 60
"""A field or element of a structured value is cut to this."""
_ITEMS = 4
_DEPTH = 2


def _cut(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def _fields(value: object) -> list[str] | None:
    """The field names of a dataclass instance or a Pydantic model."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [f.name for f in dataclasses.fields(value)]
    model_fields: object = getattr(type(value), "model_fields", None)
    if isinstance(model_fields, dict):
        return [str(name) for name in cast(dict[object, object], model_fields)]
    return None


def _sequence(
    items: Sequence[object], open_: str, close: str, budget: int, depth: int
) -> str:
    """The first items that fit in ``budget``, then the length: the length
    is never cut away."""
    more = f"… {len(items)} items"
    shown: list[str] = []
    for i, item in enumerate(items[:_ITEMS]):
        part = _cut(_preview(item, depth + 1), FIELD_CHARS)
        rest = "" if i == len(items) - 1 else f", {more}"
        if len(open_ + ", ".join([*shown, part]) + rest + close) > budget and shown:
            break
        shown.append(part)
    if len(shown) < len(items):
        shown.append(more)
    return f"{open_}{', '.join(shown)}{close}"


def _preview(value: object, depth: int) -> str:
    shape: object = getattr(value, "shape", None)
    dtype: object = getattr(value, "dtype", None)
    kind = type(value).__name__
    if shape and dtype is not None:
        dims = tuple(cast(tuple[int, ...], shape))
        return f"{kind}(shape={dims}, dtype={dtype})"
    if isinstance(value, Iterator):
        return f"<iterator {kind}>"
    if depth < _DEPTH:
        names = _fields(value)
        if names is not None:
            parts = [
                f"{name}={_cut(_preview(getattr(value, name, None), depth + 1), FIELD_CHARS)}"
                for name in names
            ]
            return f"{kind}({', '.join(parts)})"
        if isinstance(value, list | tuple):
            items = cast(list[object] | tuple[object, ...], value)
            open_, close = ("[", "]") if isinstance(value, list) else ("(", ")")
            budget = FIELD_CHARS if depth > 0 else PREVIEW_CHARS
            return _sequence(items, open_, close, budget, depth)
        if isinstance(value, dict):
            entries = list(cast(dict[object, object], value).items())
            shown = [
                f"{_cut(repr(k), FIELD_CHARS)}: {_cut(_preview(v, depth + 1), FIELD_CHARS)}"
                for k, v in entries[:_ITEMS]
            ]
            if len(entries) > _ITEMS:
                shown.append(f"… {len(entries)} items")
            return "{" + ", ".join(shown) + "}"
    try:
        return _repr.repr(value)
    except Exception as e:
        return f"<repr failed: {type(e).__name__}>"


class _Budget:
    """The samples left for one request, shared by the members of a pool."""

    __slots__ = ("left", "taken", "lock")

    def __init__(self, count: int) -> None:
        self.left = count
        self.taken = 0
        self.lock = threading.Lock()

    def take(self) -> int | None:
        """The index of the next sample, or ``None`` when none is left."""
        with self.lock:
            if self.left <= 0:
                return None
            self.left -= 1
            self.taken += 1
            return self.taken - 1


@final
class SampleProbe:
    """The probe (``IcoProbe``) of a sampled operator: calls the probe it
    replaced, if any, bubbles a preview of each result through ``emit`` and
    gives the operator its probe back when the samples are taken."""

    __slots__ = ("operator", "key", "inner", "_budget", "_emit")

    def __init__(
        self,
        operator: Any,
        key: str,
        budget: _Budget,
        inner: IcoProbe | None,
        emit: Callable[[IcoRuntimeEvent], None],
    ) -> None:
        self.operator = operator
        self.key = key
        self.inner = inner
        self._budget = budget
        self._emit = emit

    def _take(self, value: object) -> None:
        index = self._budget.take()
        if index is not None:
            self._emit(IcoSampleEvent.create(self.key, index, value))
        done = index is None or self._budget.left == 0
        if done and self.operator.probe is self:
            self.operator.probe = self.inner

    def call(self, fn: Callable[[Any], Any], item: Any) -> Any:
        result = fn(item) if self.inner is None else self.inner.call(fn, item)
        self._take(result)
        return result

    def call_with_context(
        self, fn: Callable[[Any, Any], Any], item: Any, context: Any
    ) -> Any:
        if self.inner is None:
            result = fn(item, context)
        else:
            result = self.inner.call_with_context(fn, item, context)
        self._take(result)
        return result

    async def call_async(self, fn: Callable[[Any], Awaitable[Any]], item: Any) -> Any:
        if self.inner is None:
            result = await fn(item)
        else:
            result = await self.inner.call_async(fn, item)
        self._take(result)
        return result


def is_sampled_kind(node: IcoNodeProtocol) -> bool:
    """Whether ``node`` has a probe slot (every node of the core does)."""
    from ico.core.async_operator import IcoAsyncOperator
    from ico.core.context_operator import IcoContextOperator
    from ico.core.operator import IcoOperator

    return isinstance(node, IcoOperator | IcoContextOperator | IcoAsyncOperator)


def start_sampling(
    flow: IcoNodeProtocol,
    root_key: str,
    command: IcoSampleCommand,
    emit: Callable[[IcoRuntimeEvent], None],
) -> bool:
    """Set sample probes on the operators of ``flow`` (keyed from
    ``root_key``) at ``command.node``; returns whether there was one.

    A new request for an operator replaces the one in progress.
    """
    budget = _Budget(command.count)
    found = False
    for planned in plan_nodes(flow, root_key=root_key):
        node = planned.node
        if planned.key != command.node or not is_sampled_kind(node):
            continue
        operator = cast(Any, node)
        current = cast(IcoProbe | None, operator.probe)
        inner = current.inner if isinstance(current, SampleProbe) else current
        operator.probe = SampleProbe(operator, command.node, budget, inner, emit)
        found = True
    return found

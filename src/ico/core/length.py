"""Lengths that flow through a chain of streams.

A progress node inside a ``stream`` gets its total from the length hint of
the stream's input (``operator.length_hint``). Operators keep the hint:

- an operator that returns ``iter(a_list)`` gives one for free;
- a 1:1 ``stream`` passes its input's hint on, so totals survive a chain of
  maps;
- a ``filter`` over items already in memory keeps the length of the kept
  ones; over a lazy iterator it stays lazy, without a hint;
- an operator that knows how many items it will yield before yielding them
  (a reducer that groups windows into scenes) returns
  :func:`with_length` of its generator.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterable, Iterator
from typing import Generic, TypeVar, final

T = TypeVar("T")

_MATERIALIZED: tuple[type, ...] = (
    type(iter([])),
    type(iter(())),
    type(iter(range(0))),
)


@final
class HintedIterator(Generic[T], Iterator[T]):
    """An iterator with a length hint computed on demand."""

    __slots__ = ("__wrapped__", "_hint", "ico_origin")

    def __init__(
        self,
        items: Iterator[T],
        hint: Callable[[], int],
        origin: Iterator[object] | None = None,
    ) -> None:
        self.__wrapped__ = items
        self._hint = hint
        self.ico_origin = origin
        """The iterator whose items these are, unchanged (a source passes its
        provider's items on): see :func:`materialized`."""

    def __iter__(self) -> HintedIterator[T]:
        return self

    def __next__(self) -> T:
        return next(self.__wrapped__)

    def __length_hint__(self) -> int:
        return max(self._hint(), 0)

    def close(self) -> None:
        close = getattr(self.__wrapped__, "close", None)
        if callable(close):
            close()


def with_length(items: Iterable[T], length: int) -> Iterator[T]:
    """``items`` with a length hint: ``length`` minus the items taken.

    For an operator that knows how many items it will yield before it
    yields them, so that a progress node in the stream over them has a
    total::

        def group_windows(windows: Iterator[Window]) -> Iterator[Scene]:
            plan = ...  # the scenes are known from the windows' metadata
            return with_length(_scenes(windows, plan), len(plan))
    """
    iterator = iter(items)
    taken = 0

    def counting() -> Iterator[T]:
        nonlocal taken
        for item in iterator:
            taken += 1
            yield item

    return HintedIterator(counting(), lambda: length - taken)


def passing_length(
    items: Iterator[object], outputs: Iterator[T], *, same_items: bool = False
) -> Iterator[T]:
    """``outputs`` with the live length hint of ``items`` (one output per
    input), or ``outputs`` itself when ``items`` has no hint. ``same_items``:
    the outputs are the input items themselves, so they are in memory when
    those are."""
    if operator.length_hint(items, -1) < 0:
        return outputs
    origin = items if same_items else None
    return HintedIterator(outputs, lambda: operator.length_hint(items, 0), origin)


def materialized(items: Iterator[object]) -> bool:
    """Whether ``items`` walks a sequence already in memory, so that taking
    all its items now costs no upstream work (also through wrappers that
    set ``__wrapped__``, such as profiler probes)."""
    current: object = items
    while True:
        if type(current) in _MATERIALIZED:
            return True
        origin = getattr(current, "ico_origin", None)
        if origin is not None:
            current = origin
            continue
        inner = getattr(current, "__wrapped__", None)
        if inner is None:
            return False
        current = inner

"""Probes: measure calls and items of a node, with self time.

Each thread keeps a stack of open frames, one per probed call or
``next()`` in progress. A frame adds up the time of the frames opened
inside it on the same thread, so on exit::

    self = elapsed - nested;  parent.nested += elapsed

The nesting is dynamic, as in cProfile: when a stream's ``next()`` pulls its
upstream's ``next()``, the upstream's time is subtracted from the stream's
self time, although the upstream is a sibling of the stream in the chain.
Frames of other threads never nest: the time a node waits for work done in
another thread (an async stream's pool) stays in its own self time.
"""

from __future__ import annotations

import operator
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from typing import Any, cast, final

from ico.tools.profiler.table import (
    CALL_ERROR,
    END,
    ITEM,
    NEXT_ERROR,
    OPEN,
    START,
    VALUE,
    ProfileTable,
)


@final
class _Frame:
    __slots__ = ("tokens", "nested", "began")

    def __init__(self, tokens: list[int], metrics: int, began: int) -> None:
        self.tokens = tokens
        self.nested = [0] * metrics
        self.began = began


class _Stack(threading.local):
    def __init__(self) -> None:
        self.frames: list[_Frame] = []


_stack = _Stack()


@final
class NodeProbe:
    """The probe of one node (an ``IcoProbe``): records into ``table`` under
    ``key``.

    A ``timed`` probe (async streams and their pool members) also records
    the busy time of its calls and the active time of its iterators, from
    ``perf_counter_ns``, for the utilization of pools.
    """

    __slots__ = ("key", "table", "timed", "_metrics")

    def __init__(self, key: str, table: ProfileTable, *, timed: bool = False) -> None:
        self.key = key
        self.table = table
        self.timed = timed
        self._metrics = table.metrics

    # ─── IcoProbe ───

    def call(self, fn: Callable[[Any], Any], item: Any) -> Any:
        frame = self.enter()
        try:
            out = fn(item)
        except BaseException:
            self.exit(frame, CALL_ERROR)
            raise
        if isinstance(out, Iterator):
            self.exit(frame, OPEN)
            return _TimedIterator(cast(Iterator[Any], out), self)
        self.exit(frame, VALUE)
        return out

    def call_with_context(
        self, fn: Callable[[Any, Any], Any], item: Any, context: Any
    ) -> Any:
        frame = self.enter()
        try:
            out = fn(item, context)
        except BaseException:
            self.exit(frame, CALL_ERROR)
            raise
        self.exit(frame, VALUE)
        return out

    async def call_async(self, fn: Callable[[Any], Awaitable[Any]], item: Any) -> Any:
        """Measured without a frame and without thread-bound metrics:
        coroutines of one thread interleave across ``await``."""
        started = [
            (i, m, m.start()) for i, m in enumerate(self._metrics) if not m.thread_bound
        ]
        began = time.perf_counter_ns() if self.timed else 0
        kind = VALUE
        try:
            return await fn(item)
        except BaseException:
            kind = CALL_ERROR
            raise
        finally:
            elapsed: list[int | None] = [None] * len(self._metrics)
            for i, metric, token in started:
                elapsed[i] = metric.stop(token)
            busy = time.perf_counter_ns() - began if self.timed else 0
            self.table.shard().record(self.key, kind, elapsed, elapsed, busy)

    # ─── Frames ───

    def enter(self) -> _Frame:
        began = time.perf_counter_ns() if self.timed else 0
        frame = _Frame([m.start() for m in self._metrics], len(self._metrics), began)
        _stack.frames.append(frame)
        return frame

    def exit(self, frame: _Frame, kind: int) -> None:
        elapsed = [m.stop(t) for m, t in zip(self._metrics, frame.tokens, strict=True)]
        busy = time.perf_counter_ns() - frame.began if self.timed else 0
        frames = _stack.frames
        frames.pop()  # ``frame``: every enter has its exit on the same thread
        own: list[int | None] = [
            e - n for e, n in zip(elapsed, frame.nested, strict=True)
        ]
        if frames:
            parent = frames[-1].nested
            for i, value in enumerate(elapsed):
                parent[i] += value
        self.table.shard().record(
            self.key, kind, cast(list[int | None], elapsed), own, busy
        )


@final
class _TimedIterator(Iterator[Any]):
    """An iterator returned by a probed call: every ``next()`` is an event.

    It keeps the length hint of the iterator it wraps (stream progress
    totals use it) and passes ``close()`` on.
    """

    __slots__ = ("_inner", "_probe", "_started", "_began")

    def __init__(self, inner: Iterator[Any], probe: NodeProbe) -> None:
        self._inner = inner
        self._probe = probe
        self._started = False
        self._began = 0
        """``perf_counter_ns`` at the first ``next()`` (timed probes)."""

    def __iter__(self) -> _TimedIterator:
        return self

    def __next__(self) -> Any:
        probe = self._probe
        frame = probe.enter()
        if probe.timed and not self._started and not self._began:
            self._began = frame.began
        try:
            value = next(self._inner)
        except StopIteration:
            probe.exit(frame, END)
            self._finish()
            raise
        except BaseException:
            probe.exit(frame, NEXT_ERROR)
            self._finish()
            raise
        probe.exit(frame, ITEM if self._started else START)
        self._started = True
        return value

    def _finish(self) -> None:
        """Record the active time of a timed iterator, once."""
        if self._began:
            probe = self._probe
            active = time.perf_counter_ns() - self._began
            self._began = 0
            probe.table.shard().record_active(probe.key, active)

    def __length_hint__(self) -> int:
        hint = operator.length_hint(self._inner, -1)
        return hint if hint >= 0 else cast(int, NotImplemented)

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()

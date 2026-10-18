"""Metrics measured around every profiled call or item."""

from __future__ import annotations

import time
from typing import Protocol, final


class IcoMetric(Protocol):
    """A quantity measured around a call: ``stop(start())`` is its value.

    A metric keeps no state between ``start`` and ``stop``: the token lives
    on the caller's stack, so one metric serves all threads and re-entrant
    calls.
    """

    @property
    def name(self) -> str:
        """Key of the metric in profile snapshots (``"wall"``)."""
        ...

    @property
    def unit(self) -> str:
        """Unit of the values in snapshots (``"s"``)."""
        ...

    @property
    def scale(self) -> float:
        """Factor from the integer values of ``stop`` to ``unit``."""
        ...

    @property
    def thread_bound(self) -> bool:
        """Whether the metric measures the calling thread only (CPU time):
        it is not measured for async operators, which suspend across
        ``await``."""
        ...

    def start(self) -> int:
        """A token for ``stop``."""
        ...

    def stop(self, token: int) -> int:
        """The value since ``start`` returned ``token``, in integer units."""
        ...


@final
class WallTime:
    """Elapsed wall time (``time.perf_counter_ns``)."""

    __slots__ = ()

    name = "wall"
    unit = "s"
    scale = 1e-9
    thread_bound = False

    def start(self) -> int:
        return time.perf_counter_ns()

    def stop(self, token: int) -> int:
        return time.perf_counter_ns() - token


@final
class CpuTime:
    """CPU time of the calling thread (``time.thread_time_ns``).

    The CPU time of other threads (e.g. of an async stream's pool) is not
    included, so ``wall - cpu`` is the time the thread waited: I/O, queues,
    workers, the GIL.
    """

    __slots__ = ()

    name = "cpu"
    unit = "s"
    scale = 1e-9
    thread_bound = True

    def start(self) -> int:
        return time.thread_time_ns()

    def stop(self, token: int) -> int:
        return time.thread_time_ns() - token


DEFAULT_METRICS: tuple[IcoMetric, ...] = (WallTime(), CpuTime())

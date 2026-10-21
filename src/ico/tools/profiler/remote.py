"""Profiling in worker runner processes.

A profiled worker's ``runner_probe`` is a :class:`RunnerProfiler`: in the
runner process, it instruments the flow the runner built into a table of the runner process, with keys
below the worker's key, and wrapped in :class:`ProfileHost`. The host is a
runtime node of the runner: after an item, at most every ``interval``
seconds, and once more on deactivation, it bubbles an
:class:`IcoProfileEvent` with the cumulative aggregates of the process to
the worker, where the profiler tool merges them. Events are sent from the
runner's main thread only, between items, so they never race with items on
the worker channel.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, final

from ico.core.operator import IcoOperator, IcoOperatorProtocol
from ico.core.runtime.event import IcoRuntimeEvent
from ico.core.runtime.node import IcoRuntimeNode
from ico.core.tree_utils import TreePathIndex
from ico.resources.pool import pool
from ico.tools.profiler.metrics import IcoMetric
from ico.tools.profiler.resources import ResourceProfile, ResourceStats
from ico.tools.profiler.table import Layout, NodeAggregate, ProfileTable


@final
@dataclass(slots=True, frozen=True)
class IcoProfileEvent(IcoRuntimeEvent):
    """The cumulative profile of a runner process (replaces its previous
    one)."""

    layout: Layout
    nodes: dict[str, NodeAggregate]
    peak_rss: int | None
    """Peak resident memory of the process in bytes, if known."""
    resources: tuple[ResourceProfile, ...] = ()
    """The resource rows of the process's pool (``process`` set by the
    profiler tool that merges them)."""
    resource_peaks: tuple[tuple[str, int], ...] = ()
    """Most bytes the process's pool held at once, per device."""
    process: str = ""
    """The name of the worker whose runner sent it (its plan label)."""

    @staticmethod
    def create(
        layout: Layout,
        nodes: dict[str, NodeAggregate],
        peak_rss: int | None,
        resources: tuple[ResourceProfile, ...] = (),
        resource_peaks: tuple[tuple[str, int], ...] = (),
        process: str = "",
    ) -> IcoProfileEvent:
        return IcoProfileEvent(
            trace=TreePathIndex(),
            layout=layout,
            nodes=nodes,
            peak_rss=peak_rss,
            resources=resources,
            resource_peaks=resource_peaks,
            process=process,
        )


@final
@dataclass(frozen=True, slots=True)
class RunnerProfiler:
    """A worker's ``runner_probe`` (``IcoRunnerProbe``): instruments the flow
    its runner built and wraps it in a :class:`ProfileHost`.

    Pickleable when the metrics are.
    """

    root_key: str
    metrics: tuple[IcoMetric, ...]
    interval: float | None
    name: str = ""
    """The worker's plan label: the process its rows are shown under."""

    def wrap(
        self, flow: IcoOperatorProtocol[Any, Any]
    ) -> IcoOperatorProtocol[Any, Any]:
        from ico.tools.profiler.instrument import instrument_table

        table = ProfileTable(self.metrics)
        instrument_table(flow, table, self.root_key, self.interval)
        return ProfileHost(flow, table, self.interval, name=self.name)


@final
class ProfileHost(IcoOperator[Any, Any], IcoRuntimeNode):
    """The root of a profiled runner flow: passes items to the flow and
    reports the process's profile as events."""

    def __init__(
        self,
        flow: IcoOperatorProtocol[Any, Any],
        table: ProfileTable,
        interval: float | None,
        *,
        name: str = "",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        IcoOperator.__init__(  # pyright: ignore[reportUnknownMemberType]
            self, fn=self._run, name="profiler", children=[flow]
        )
        IcoRuntimeNode.__init__(self, runtime_name="profiler")
        self.flow = flow
        self.table = table
        self.interval = interval
        self.clock = clock
        self.process = name
        self._last = clock()
        self._resources = ResourceStats()

    def _run(self, item: Any) -> Any:
        out = self.flow(item)
        if self.interval is not None and self.clock() - self._last >= self.interval:
            self._emit()
        return out

    def on_activate(self) -> None:
        super().on_activate()
        self.table.reset()
        self._last = self.clock()
        self._resources = ResourceStats()
        pool().add_listener(self._resources)

    def on_deactivate(self) -> None:
        super().on_deactivate()
        pool().remove_listener(self._resources)
        self._emit()

    def _emit(self) -> None:
        self._last = self.clock()
        self.bubble_event(
            IcoProfileEvent.create(
                self.table.layout,
                self.table.merged(),
                peak_rss(),
                tuple(self._resources.rows()),
                tuple(sorted(self._resources.peaks().items())),
                self.process,
            )
        )


def peak_rss() -> int | None:
    """Peak resident memory of this process in bytes (``None`` where the
    ``resource`` module is missing, e.g. on Windows)."""
    try:
        import resource
    except ImportError:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024

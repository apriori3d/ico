"""Runtime tool that collects the profile of an instrumented flow."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from typing import Any, TypedDict, final

from ico.core.runtime.event import IcoRuntimeEvent
from ico.core.runtime.tool import IcoRuntimeTool
from ico.tools.profiler.metrics import DEFAULT_METRICS, IcoMetric
from ico.tools.profiler.probe import NodeProbe
from ico.tools.profiler.remote import IcoProfileEvent, peak_rss
from ico.tools.profiler.table import (
    Layout,
    NodeProfile,
    ProfileTable,
    merge_into,
    profiles,
)

CALIBRATION_CALLS = 2000


class MetricInfo(TypedDict):
    name: str
    unit: str


class ProcessInfo(TypedDict):
    source: str
    """``main`` for the process of the tool, else the runtime path of the
    runner's profile node."""
    peak_rss: int | None
    """Peak resident memory in bytes, if known."""


class ProfileSnapshot(TypedDict):
    """The profile of a run; JSON-compatible."""

    nodes: list[NodeProfile]
    """Every instrumented node, in plan order, measured or not."""
    metrics: list[MetricInfo]
    elapsed: float
    """Seconds since activation (until deactivation for the final one)."""
    overhead: float
    """Estimated seconds spent in the probes themselves (in the self time of
    the nodes that called them)."""
    processes: list[ProcessInfo]
    """The main process and every runner process that reported."""
    final: bool


@final
class ProfilerTool(IcoRuntimeTool):
    """Collect the profile of a flow instrumented with
    :func:`~ico.tools.profiler.instrument`.

    Every operator call and every item of an iterator a node returns is
    measured with each metric (wall and CPU time by default), with self
    time (see :mod:`ico.tools.profiler.probe`). Measurements are kept as
    mergeable aggregates per node, so memory does not grow with the run.

    ``on_profile`` receives a snapshot every ``interval`` seconds while the
    runtime is active (from a timer thread) and a final one on deactivation;
    calls never overlap. :meth:`snapshot` returns one at any time.

    Example:
        >>> profiler = ProfilerTool()
        >>> flow = instrument(flow, profiler)
        >>> IcoRuntime(flow, tools=[profiler]).activate().run().deactivate()
        >>> profiler.snapshot()["nodes"][0]["metrics"]["wall"]["self"]
    """

    __slots__ = (
        "on_profile",
        "interval",
        "clock",
        "table",
        "_started",
        "_stopped",
        "_timer",
        "_stop",
        "_emit_lock",
        "_per_frame",
        "_remote",
        "_remote_lock",
    )

    def __init__(
        self,
        on_profile: Callable[[ProfileSnapshot], None] | None = None,
        *,
        metrics: Sequence[IcoMetric] = DEFAULT_METRICS,
        interval: float | None = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create the tool.

        Args:
            on_profile: Receives snapshots during the run and the final one.
            metrics: What to measure; names must be unique.
            interval: Seconds between snapshots during the run; ``None`` for
                the final one only.
            clock: Time source of ``elapsed`` (replaceable in tests).

        Raises:
            ValueError: For no metrics, duplicate names or an interval that
                is not positive.
        """
        if interval is not None and interval <= 0:
            raise ValueError(f"interval must be positive, got {interval}")
        super().__init__()
        self.on_profile = on_profile
        self.interval = interval
        self.clock = clock
        self.table = ProfileTable(metrics)
        self._started: float | None = None
        self._stopped: float | None = None
        self._timer: threading.Thread | None = None
        self._stop = threading.Event()
        self._emit_lock = threading.Lock()
        self._per_frame: float | None = None
        self._remote: dict[str, IcoProfileEvent] = {}
        """The latest profile of every runner process, by source."""
        self._remote_lock = threading.Lock()

    # ─── Lifecycle ───

    def on_activate(self) -> None:
        super().on_activate()
        self._frame_cost()  # calibrate before the run, outside its frames
        self.table.reset()
        with self._remote_lock:
            self._remote.clear()
        self._started, self._stopped = self.clock(), None
        if self.on_profile is not None and self.interval is not None:
            self._stop.clear()
            self._timer = threading.Thread(
                target=self._tick, name="ico-profiler", daemon=True
            )
            self._timer.start()

    def on_deactivate(self) -> None:
        super().on_deactivate()
        timer, self._timer = self._timer, None
        if timer is not None:
            self._stop.set()
            timer.join()
        self._stopped = self.clock()
        self._emit()

    # ─── Snapshots ───

    def snapshot(self) -> ProfileSnapshot:
        """The profile measured so far (final after deactivation)."""
        merged = self.table.merged()
        layout = Layout()
        layout.update(self.table.layout)
        processes: list[ProcessInfo] = [{"source": "main", "peak_rss": peak_rss()}]
        with self._remote_lock:
            remote = sorted(self._remote.items())
        for source, event in remote:
            layout.update(event.layout)
            merge_into(merged, event.nodes)
            processes.append({"source": source, "peak_rss": event.peak_rss})
        frames = sum(node.frames for node in merged.values())
        now = self._stopped if self._stopped is not None else self.clock()
        return {
            "nodes": profiles(self.table.metrics, layout, merged),
            "metrics": [{"name": m.name, "unit": m.unit} for m in self.table.metrics],
            "elapsed": now - self._started if self._started is not None else 0.0,
            "overhead": frames * self._frame_cost(),
            "processes": processes,
            "final": self._stopped is not None,
        }

    def on_forward_event(self, event: IcoRuntimeEvent) -> None:
        super().on_forward_event(event)
        if isinstance(event, IcoProfileEvent):
            with self._remote_lock:
                self._remote[str(event.trace.reverse())] = event

    def _tick(self) -> None:
        interval = self.interval
        assert interval is not None
        while not self._stop.wait(interval):
            self._emit()

    def _emit(self) -> None:
        if self.on_profile is None:
            return
        with self._emit_lock:
            self.on_profile(self.snapshot())

    def _frame_cost(self) -> float:
        """Seconds a probe adds to one call, measured once on an empty call."""
        if self._per_frame is None:
            self._per_frame = _calibrate(self.table.metrics)
        return self._per_frame


def _calibrate(metrics: Sequence[IcoMetric]) -> float:
    table = ProfileTable(metrics)
    probe = NodeProbe("calibration", table)

    def noop(item: Any) -> Any:
        return item

    start = time.perf_counter_ns()
    for _ in range(CALIBRATION_CALLS):
        noop(None)
    bare = time.perf_counter_ns() - start
    start = time.perf_counter_ns()
    for _ in range(CALIBRATION_CALLS):
        probe.call(noop, None)
    probed = time.perf_counter_ns() - start
    return max(probed - bare, 0) / CALIBRATION_CALLS * 1e-9

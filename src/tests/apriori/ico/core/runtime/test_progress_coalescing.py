"""Progress events: slow items one by one, fast items summed, the rest
never late by more than an interval, and exact totals, also in workers."""

import time
from collections.abc import Iterator
from typing import Any

import pytest

from ico.core.operator import IcoOperator, IcoOperatorProtocol
from ico.core.runtime.event import IcoRuntimeEvent
from ico.core.runtime.progress import (
    IcoProgress,
    IcoProgressEvent,
    IcoProgressTotalEvent,
)
from ico.core.runtime.runtime import IcoRuntime
from ico.core.runtime.tool import IcoRuntimeTool
from ico.core.sink import IcoSink
from ico.core.source import IcoSource
from ico.core.stream import IcoStream
from ico.runtime.worker.mp.mp_worker import MPWorker
from ico.tools.progress.json_progress_tool import JsonProgressTool, ProgressSnapshot

# Module level: spawned runner processes import this module.

N = 600


class Events(IcoRuntimeTool):
    """Progress events with their arrival time."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[float, float]] = []

    def on_forward_event(self, event: IcoRuntimeEvent) -> None:
        if isinstance(event, IcoProgressEvent):
            self.events.append((time.monotonic(), event.advance))

    @property
    def total(self) -> float:
        return sum(advance for _, advance in self.events)


def ignore(x: Any) -> None:
    pass


def run(items: Any, step: Any, progress: IcoProgress[Any], *tools: Any) -> Events:
    events = Events()
    flow = IcoSource(items) | IcoStream(progress | IcoOperator(step)) | IcoSink(ignore)
    IcoRuntime(flow, tools=[events, *tools]).activate().run().deactivate()
    return events


def many() -> Iterator[int]:
    yield from range(N)


def same(x: int) -> int:
    return x


def test_fast_items_are_summed_and_the_total_is_exact() -> None:
    events = run(many, same, IcoProgress[int](total=N, interval=0.1))
    assert events.total == N
    assert len(events.events) < 20  # not one event per item


def test_slow_items_are_reported_one_by_one() -> None:
    def few() -> Iterator[int]:
        yield from range(4)

    def slow(x: int) -> int:
        time.sleep(0.08)
        return x

    events = run(few, slow, IcoProgress[int](total=4, interval=0.05))
    assert [advance for _, advance in events.events] == [1, 1, 1, 1]


def test_interval_zero_reports_every_item() -> None:
    events = run(many, same, IcoProgress[int](total=N, interval=0))
    assert len(events.events) == N and events.total == N


def test_the_rest_is_sent_during_a_long_item() -> None:
    started: list[float] = []

    def burst_then_long() -> Iterator[int]:
        yield from range(5)
        started.append(time.monotonic())
        yield 99

    def step(x: int) -> int:
        if x == 99:
            time.sleep(0.6)
        return x

    events = run(burst_then_long, step, IcoProgress[int](total=6, interval=0.1))
    assert events.total == 6
    # All six were reported while the long item was still running.
    reported = 0.0
    for at, advance in events.events:
        reported += advance
        if reported == 6:
            assert at < started[0] + 0.45
            break


class Ordered(IcoRuntimeTool):
    """Progress and total events, in order."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, float]] = []

    def on_forward_event(self, event: IcoRuntimeEvent) -> None:
        if isinstance(event, IcoProgressEvent):
            self.events.append(("advance", event.advance))
        elif isinstance(event, IcoProgressTotalEvent):
            self.events.append(("total", event.total))


def test_iterations_are_not_mixed() -> None:
    def videos() -> Iterator[int]:
        yield from (3, 2)

    def scenes(n: int) -> Iterator[int]:
        return iter(range(n))

    # A long interval: only the leading edge and the flushes report.
    progress: IcoProgress[int] = IcoProgress(name="scenes", interval=10)
    body = IcoOperator(scenes) | IcoStream(progress | IcoOperator(same))
    ordered = Ordered()
    flow = IcoSource(videos) | IcoStream(body | IcoOperator(list)) | IcoSink(ignore)
    IcoRuntime(flow, tools=[ordered]).activate().run().deactivate()

    # Every item of an iteration is reported before the next one starts.
    per_iteration: list[float] = []
    for kind, value in ordered.events:
        if kind == "total":
            per_iteration.append(0.0)
        else:
            per_iteration[-1] += value
    assert per_iteration == [3.0, 2.0]


def test_a_negative_interval_is_an_error() -> None:
    with pytest.raises(ValueError, match="interval"):
        IcoProgress[int](interval=-1)


def worker_flow() -> IcoOperatorProtocol[int, int]:
    return IcoProgress[int](name="inner") | IcoOperator(same)


def test_workers_report_exact_totals_without_waiting_per_item() -> None:
    worker = MPWorker(worker_flow)
    snapshots: list[ProgressSnapshot] = []
    events = Events()
    flow = IcoSource(many) | IcoStream(worker) | IcoSink(ignore)
    runtime = IcoRuntime(flow, tools=[events, JsonProgressTool(snapshots.append)])
    runtime.activate()
    try:
        runtime.run()
    finally:
        runtime.deactivate()

    assert events.total == N
    assert len(events.events) < N  # summed in the runner
    assert snapshots[-1]["done"] == N

import asyncio
import operator
import time
from collections.abc import Iterator
from typing import Any, cast

import pytest

from ico.core.async_operator import IcoAsyncOperator
from ico.core.async_stream import IcoAsyncStream
from ico.core.context_operator import IcoContextOperator
from ico.core.operator import IcoOperator
from ico.core.runtime.runtime import IcoRuntime
from ico.core.sink import IcoSink
from ico.core.source import IcoSource
from ico.core.stream import IcoStream
from ico.describe.plan.json_renderer import plan_view
from ico.registry import IcoRegistry, register
from ico.spec import run, to_spec
from ico.tools.profiler import (
    CpuTime,
    NodeProfile,
    ProfilerTool,
    ProfileSnapshot,
    WallTime,
    instrument,
)

REGISTRY = IcoRegistry()


@register("prof.inc", 1, registry=REGISTRY)
def inc(x: int) -> int:
    return x + 1


@register("prof.double", 1, registry=REGISTRY)
def double(x: int) -> int:
    return x * 2


def registered() -> list[IcoOperator[Any, Any]]:
    """The shared registered nodes of this module."""
    return [cast(IcoOperator[Any, Any], op) for op in (inc, double)]


class Clock:
    """Time that passes only when an operator says so."""

    def __init__(self) -> None:
        self.now = 0

    def advance(self, units: int) -> None:
        self.now += units


class FakeTime:
    """A metric on a ``Clock``: exact, deterministic times."""

    name = "wall"
    unit = "u"
    scale = 1.0
    thread_bound = False

    def __init__(self, clock: Clock) -> None:
        self.clock = clock

    def start(self) -> int:
        return self.clock.now

    def stop(self, token: int) -> int:
        return self.clock.now - token


def ignore(x: Any) -> None:
    pass


def activate_run(flow: Any, profiler: ProfilerTool) -> ProfileSnapshot:
    IcoRuntime(flow, tools=[profiler]).activate().run().deactivate()
    return profiler.snapshot()


def by_label(snapshot: ProfileSnapshot) -> dict[str, NodeProfile]:
    return {node["label"]: node for node in snapshot["nodes"]}


# ─── Lazy pipelines: items, startup, end, self time ───


def lazy_flow(clock: Clock) -> Any:
    def read() -> Iterator[int]:
        clock.advance(100)  # setup before the first item
        for i in range(3):
            clock.advance(10)
            yield i
        clock.advance(5)  # cleanup

    def work(x: int) -> int:
        clock.advance(7)
        return x

    return (
        IcoSource(read, name="read")
        | IcoStream(IcoOperator(work), name="each")
        | IcoSink(ignore, name="sink")
    )


def test_items_startup_end_and_self_time_of_a_lazy_chain() -> None:
    clock = Clock()
    profiler = ProfilerTool(metrics=[FakeTime(clock)], interval=None)
    flow = instrument(lazy_flow(clock), profiler)
    nodes = by_label(activate_run(flow, profiler))

    read = nodes["read"]
    assert (read["calls"], read["items"], read["errors"]) == (1, 3, 0)
    assert read["startup"] is not None and read["end"] is not None
    assert read["startup"]["wall"]["count"] == 1
    assert read["startup"]["wall"]["mean"] == 110
    assert read["end"]["wall"]["mean"] == 5
    wall = read["metrics"]["wall"]
    assert wall["count"] == 2 and wall["mean"] == 10  # items after the first
    assert wall["self"] == wall["total"] == 135

    work = nodes["work"]
    assert (work["calls"], work["items"]) == (3, 0)
    assert work["metrics"]["wall"]["mean"] == 7
    assert work["metrics"]["wall"]["self"] == 21
    assert work["startup"] is None

    # The source is pulled by the stream: its time is the source's, not the
    # stream's, although the source is not a child of the stream.
    each = nodes["each"]["metrics"]["wall"]
    assert each["self"] == 0
    assert each["total"] == 135 + 21

    # Self times add up to the whole run.
    root = flow.name
    total = by_label(profiler.snapshot())[root]["metrics"]["wall"]["total"]
    assert sum(n["metrics"]["wall"]["self"] for n in nodes.values()) == total == 156


def test_keys_are_plan_paths_in_plan_order() -> None:
    profiler = ProfilerTool(interval=None)
    flow = instrument(lazy_flow(Clock()), profiler)
    keys = [(n["key"], n["label"]) for n in profiler.snapshot()["nodes"]]
    assert keys == [
        ("0", flow.name),
        ("0.0", keys[1][1]),
        ("0.0.0", "read"),
        ("0.0.1", "each"),
        ("0.0.1.0", "work"),
        ("0.1", "sink"),
    ]


def test_errors_count_and_stay_out_of_the_distribution() -> None:
    clock = Clock()

    def fail(x: int) -> int:
        clock.advance(4)
        raise ValueError(x)

    profiler = ProfilerTool(metrics=[FakeTime(clock)], interval=None)
    op = instrument(IcoOperator(fail), profiler)
    with pytest.raises(ValueError):
        op(1)
    [node] = profiler.snapshot()["nodes"]
    assert (node["calls"], node["errors"]) == (1, 1)
    assert node["metrics"]["wall"]["count"] == 0
    assert node["metrics"]["wall"]["total"] == 4


def test_iterators_keep_their_length_hint() -> None:
    def three(_: None) -> Iterator[int]:
        return iter([1, 2, 3])

    def unknown(_: None) -> Iterator[int]:
        yield 1

    profiler = ProfilerTool(interval=None)
    known_op = instrument(IcoOperator(three), profiler)
    unknown_op = instrument(IcoOperator(unknown), profiler)
    assert operator.length_hint(known_op(None)) == 3
    assert operator.length_hint(unknown_op(None), -1) == -1


# ─── The flow does not change ───


def test_plan_and_signature_are_unchanged() -> None:
    flow = lazy_flow(Clock())
    before = (plan_view(flow), flow.signature)
    instrument(flow, ProfilerTool(interval=None))
    assert (plan_view(flow), flow.signature) == before


def test_spec_export_is_unchanged() -> None:
    flow = IcoStream(inc | double)
    before = to_spec(flow)
    try:
        instrument(flow, ProfilerTool(interval=None))
        assert to_spec(flow) == before
    finally:
        for node in registered():
            node.probe = None  # registered nodes are shared


def test_another_profiler_takes_the_nodes_over() -> None:
    first, second = ProfilerTool(interval=None), ProfilerTool(interval=None)
    op = instrument(IcoOperator(double), first)
    instrument(op, second)
    op(1)
    assert first.snapshot()["nodes"][0]["calls"] == 0
    assert second.snapshot()["nodes"][0]["calls"] == 1


def test_instrumenting_twice_measures_once() -> None:
    profiler = ProfilerTool(interval=None)
    op = instrument(instrument(IcoOperator(inc), profiler), profiler)
    op(1)
    assert profiler.snapshot()["nodes"][0]["calls"] == 1


# ─── Threads and operator kinds ───


def test_a_factory_pool_merges_into_one_key() -> None:
    def work(x: int) -> int:
        time.sleep(0.001)
        return x

    def numbers() -> list[int]:
        return list(range(12))

    pool = IcoAsyncStream(lambda: IcoOperator(work), pool_size=3, name="pool")
    profiler = ProfilerTool(interval=None)
    flow = instrument(IcoSource(numbers) | pool | IcoSink(ignore), profiler)
    nodes = by_label(activate_run(flow, profiler))

    keys = [n["key"] for n in profiler.snapshot()["nodes"]]
    assert len(keys) == len(set(keys))
    assert nodes["work"]["calls"] == 12  # three members in pool threads
    assert nodes["work"]["key"] == nodes["pool"]["key"] + ".0"
    assert nodes["pool"]["items"] == 12


def test_context_and_async_operators() -> None:
    def scale(x: int, factor: int) -> int:
        return x * factor

    ctx = instrument(IcoContextOperator(scale), ProfilerTool(interval=None))
    assert ctx(2, 3) == 6

    async def fetch(x: int) -> int:
        await asyncio.sleep(0.001)
        return x

    def numbers() -> list[int]:
        return [1, 2, 3]

    profiler = ProfilerTool(metrics=[WallTime(), CpuTime()], interval=None)
    pool = IcoAsyncStream([IcoAsyncOperator(fetch)])
    flow = instrument(IcoSource(numbers) | pool | IcoSink(ignore), profiler)
    node = by_label(activate_run(flow, profiler))["fetch"]
    assert node["calls"] == 3
    assert node["metrics"]["wall"]["count"] == 3
    assert node["metrics"]["cpu"]["count"] == 0  # thread-bound: not measured


def test_cpu_time_excludes_waiting() -> None:
    def wait(x: int) -> int:
        time.sleep(0.02)
        return x

    profiler = ProfilerTool(interval=None)
    op = instrument(IcoOperator(wait), profiler)
    op(1)
    metrics = profiler.snapshot()["nodes"][0]["metrics"]
    assert metrics["wall"]["total"] >= 0.02
    assert metrics["cpu"]["total"] < metrics["wall"]["total"] / 2


# ─── The tool ───


def test_snapshots_during_the_run_and_a_final_one() -> None:
    def slow(x: int) -> int:
        time.sleep(0.05)
        return x

    snapshots: list[ProfileSnapshot] = []
    profiler = ProfilerTool(snapshots.append, interval=0.01)
    flow = instrument(IcoOperator(slow), profiler)

    def numbers() -> list[int]:
        return [1, 2]

    activate_run(IcoSource(numbers) | IcoStream(flow) | IcoSink(ignore), profiler)
    assert snapshots[-1]["final"]
    assert any(not s["final"] for s in snapshots)
    assert snapshots[-1]["nodes"][0]["calls"] == 2
    assert snapshots[-1]["elapsed"] >= 0.1
    assert snapshots[-1]["overhead"] > 0
    assert snapshots[-1]["metrics"] == [
        {"name": "wall", "unit": "s"},
        {"name": "cpu", "unit": "s"},
    ]


def test_activation_resets_the_measurements() -> None:
    profiler = ProfilerTool(interval=None)
    flow = instrument(IcoOperator(inc), profiler)
    flow(1)
    activate_run(IcoOperator(ignore), profiler)
    assert profiler.snapshot()["nodes"][0]["calls"] == 0


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="interval"):
        ProfilerTool(interval=0)
    with pytest.raises(ValueError, match="unique"):
        ProfilerTool(metrics=[WallTime(), WallTime()])
    with pytest.raises(ValueError, match="metric"):
        ProfilerTool(metrics=[])


def test_spec_run_with_a_profiler() -> None:
    profiler = ProfilerTool(interval=None)
    result = run(to_spec(inc | double), 3, registry=REGISTRY, profiler=profiler)
    assert list(result.outputs) == [8]
    nodes = by_label(profiler.snapshot())
    assert nodes["prof.inc@1"]["calls"] == 1
    assert nodes["prof.double@1"]["key"] == "0.1"
    assert profiler.snapshot()["final"]


def test_profiled_spec_runs_do_not_touch_registered_nodes() -> None:
    profilers = [ProfilerTool(interval=None), ProfilerTool(interval=None)]
    for profiler in profilers:
        run(to_spec(inc | double), 3, registry=REGISTRY, profiler=profiler)
        assert by_label(profiler.snapshot())["prof.inc@1"]["calls"] == 1
    assert all(node.probe is None for node in registered())

"""Lengths flow through a chain of streams, so progress nodes downstream of
maps, filters over items in memory and reducers know their totals."""

import operator
from collections.abc import Iterator
from typing import Any

from ico import with_length
from ico.core.filter import IcoFilter
from ico.core.operator import IcoOperator
from ico.core.runtime.progress import IcoProgress
from ico.core.runtime.runtime import IcoRuntime
from ico.core.sink import IcoSink
from ico.core.source import IcoSource
from ico.core.stream import IcoStream
from ico.spec import NodeSpec, build
from ico.tools.profiler import ProfilerTool, instrument
from ico.tools.progress.json_progress_tool import JsonProgressTool, ProgressSnapshot


def ten() -> Iterator[int]:
    return iter(list(range(10)))


def lazy_ten() -> Iterator[int]:
    yield from range(10)


def double(x: int) -> int:
    return 2 * x


def even(x: int) -> bool:
    return x % 2 == 0


def ignore(_: Any) -> None:
    pass


def final_total(flow: Any, *, profile: bool = False) -> float | None:
    snapshots: list[ProgressSnapshot] = []
    tools: list[Any] = [JsonProgressTool(snapshots.append)]
    if profile:
        profiler = ProfilerTool(interval=None)
        flow = instrument(flow, profiler)
        tools.append(profiler)
    IcoRuntime(flow, tools=tools).activate().run().deactivate()
    total = snapshots[-1]["total"]
    assert snapshots[-1]["done"] == (
        total if total is not None else snapshots[-1]["done"]
    )
    return total if isinstance(total, float | int) else None


def counted() -> IcoStream[int, int]:
    return IcoStream(IcoProgress[int](name="counted") | IcoOperator(double))


def test_a_stream_passes_its_input_length_on() -> None:
    outputs = IcoStream(IcoOperator(double))(iter([1, 2, 3]))
    assert operator.length_hint(outputs) == 3
    next(outputs)
    assert operator.length_hint(outputs) == 2


def test_totals_survive_a_chain_of_maps() -> None:
    flow = IcoSource(ten) | IcoStream(IcoOperator(double)) | counted() | IcoSink(ignore)
    assert final_total(flow) == 10


def test_a_filter_over_items_in_memory_keeps_a_length() -> None:
    def flow() -> Any:
        return IcoSource(ten) | IcoFilter(even) | counted() | IcoSink(ignore)

    assert final_total(flow()) == 5
    assert final_total(flow(), profile=True) == 5  # through the profiler's probes


def test_a_filter_over_a_lazy_input_stays_lazy() -> None:
    calls: list[str] = []

    def numbers() -> Iterator[int]:
        for x in range(4):
            calls.append(f"made {x}")
            yield x

    def take(x: int) -> None:
        calls.append(f"took {x}")

    flow = IcoSource(numbers) | IcoFilter(even) | IcoSink(take)
    IcoRuntime(flow).activate().run().deactivate()
    assert calls[:3] == ["made 0", "took 0", "made 1"]  # one at a time


def test_with_length_counts_down() -> None:
    def scenes() -> Iterator[str]:
        yield from ("a", "b", "c")

    items = with_length(scenes(), 3)
    assert operator.length_hint(items) == 3
    assert next(items) == "a"
    assert operator.length_hint(items) == 2
    assert list(items) == ["b", "c"] and operator.length_hint(items) == 0


def test_a_reducer_gives_its_total() -> None:
    def pairs(items: Iterator[int]) -> Iterator[int]:
        values = list(items)
        return with_length(
            (a + b for a, b in zip(values[::2], values[1::2], strict=False)),
            len(values) // 2,
        )

    flow = IcoSource(ten) | IcoOperator(pairs) | counted() | IcoSink(ignore)
    assert final_total(flow) == 5


def test_a_null_total_in_a_spec_is_unknown() -> None:
    node = build(NodeSpec.from_dict({"kind": "progress", "params": {"total": None}}))
    assert isinstance(node, IcoProgress) and node.total is None

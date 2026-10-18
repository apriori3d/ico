import io
from collections.abc import Iterator
from typing import Any

from rich.console import Console

from ico.core.operator import IcoOperator
from ico.core.runtime.runtime import IcoRuntime
from ico.core.sink import IcoSink
from ico.core.source import IcoSource
from ico.core.stream import IcoStream
from ico.tools.profiler import (
    ProfilerTool,
    ProfileSnapshot,
    instrument,
    print_profile,
    profile_cells,
)
from ico.tools.profiler.report import duration


class Clock:
    def __init__(self) -> None:
        self.now = 0

    def advance(self, units: int) -> None:
        self.now += units


class FakeWall:
    """Wall time on a ``Clock``, in milliseconds."""

    name = "wall"
    unit = "s"
    scale = 1e-3
    thread_bound = False

    def __init__(self, clock: Clock) -> None:
        self.clock = clock

    def start(self) -> int:
        return self.clock.now

    def stop(self, token: int) -> int:
        return self.clock.now - token


def ignore(x: Any) -> None:
    pass


def profiled() -> tuple[Any, ProfileSnapshot]:
    clock = Clock()

    def read() -> Iterator[int]:
        clock.advance(100)  # setup
        for i in range(3):
            clock.advance(10)
            yield i

    def work(x: int) -> int:
        clock.advance(30)
        return x

    flow = (
        IcoSource(read, name="read")
        | IcoStream(IcoOperator(work), name="each")
        | IcoSink(ignore, name="sink")
    )
    profiler = ProfilerTool(metrics=[FakeWall(clock)], interval=None)
    instrument(flow, profiler)
    IcoRuntime(flow, tools=[profiler]).activate().run().deactivate()
    return flow, profiler.snapshot()


def by_label(snapshot: ProfileSnapshot) -> dict[str, str]:
    return {n["label"]: n["key"] for n in snapshot["nodes"]}


def test_cells_of_a_lazy_chain() -> None:
    _, snapshot = profiled()
    cells, keys = profile_cells(snapshot), by_label(snapshot)

    work = cells[keys["work"]]
    # Share of all self time: 90 of 130 (read) + 90 (work).
    assert (work["self"], work["count"], work["share"]) == ("90.0ms", "3", "41%")
    assert cells[keys["read"]]["hot"] and not work["hot"]  # read: 130ms
    assert work["p95"] == "30.0ms"

    read = cells[keys["read"]]
    assert read["self"] == "130.0ms"
    assert read["info"] == "startup 110.0ms"  # its own setup and first item

    each = cells[keys["each"]]
    assert each["self"] == "0µs"
    assert each["total"] == "90.0ms"  # the body below it in the plan
    assert "startup" not in each["info"]  # the source's startup is not its own

    sink = cells[keys["sink"]]
    assert (sink["count"], sink["p95"]) == ("1", "")  # one call: no p95


def test_self_startup_excludes_the_upstream() -> None:
    _, snapshot = profiled()
    nodes = {n["label"]: n for n in snapshot["nodes"]}
    each, read = nodes["each"]["startup"], nodes["read"]["startup"]
    assert each is not None and read is not None
    assert each["wall"]["total"] > 0.1 and each["wall"]["self"] == 0
    assert read["wall"]["self"] == read["wall"]["total"] == 0.11


def test_print_profile_shows_the_plan_with_profile_columns() -> None:
    flow, snapshot = profiled()
    out = io.StringIO()
    print_profile(flow, snapshot, console=Console(file=out, width=160))
    text = out.getvalue()
    assert "Profile:" in text and "Share of work" in text
    [line] = [line for line in text.splitlines() if "│   work" in line]
    assert "90.0ms" in line and "41%" in line


def test_durations() -> None:
    assert (duration(0.000012), duration(0.0034), duration(1.25)) == (
        "12µs",
        "3.4ms",
        "1.25s",
    )

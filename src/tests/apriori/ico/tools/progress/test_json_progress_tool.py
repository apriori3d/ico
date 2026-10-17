import json
from collections.abc import Iterable
from typing import Any

import pytest

from ico.core.async_stream import IcoAsyncStream
from ico.core.operator import IcoOperator
from ico.core.process import IcoProcess
from ico.core.runtime.progress import IcoProgress
from ico.core.runtime.runtime import IcoRuntime
from ico.core.sink import IcoSink
from ico.core.source import IcoSource
from ico.runtime.worker.mp.mp_worker import MPWorker
from ico.tools.progress.json_progress_tool import JsonProgressTool, ProgressSnapshot


class FakeClock:
    """Advances by ``step`` seconds on every call."""

    def __init__(self, step: float) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def ignore(x: Any) -> None:
    pass


def numbers(n: int) -> IcoSource[int]:
    def provide() -> Iterable[int]:
        return range(n)

    return IcoSource(provide)


def run(flow: Any, tool: JsonProgressTool) -> None:
    IcoRuntime(flow | IcoSink(ignore), tools=[tool]).activate().run().deactivate()


def collect(**kwargs: Any) -> tuple[list[ProgressSnapshot], JsonProgressTool]:
    snapshots: list[ProgressSnapshot] = []
    return snapshots, JsonProgressTool(snapshots.append, **kwargs)


# ─── Snapshots ───


def test_snapshot_format_and_final_state() -> None:
    snapshots, tool = collect(max_rate=1000, clock=FakeClock(0.1))
    progress: IcoProgress[int] = IcoProgress(total=4, name="items")
    run(numbers(4) | progress.stream(), tool)

    initial, first, last = snapshots[0], snapshots[1], snapshots[-1]
    assert set(first) == {"key", "label", "done", "total", "rate", "eta"}
    assert (initial["done"], initial["rate"], initial["eta"]) == (0.0, None, None)
    assert (first["label"], first["done"], first["total"]) == ("items", 1.0, 4)
    assert first["rate"] is None and first["eta"] is None  # nothing measured yet
    assert (last["done"], last["eta"]) == (4.0, 0.0)
    rate = last["rate"]
    assert isinstance(rate, float) and abs(rate - 4 / 0.3) < 1e-9  # 4 items, 3 steps
    json.dumps(snapshots)


def test_eta_from_rate() -> None:
    snapshots, tool = collect(max_rate=1000, clock=FakeClock(1.0))
    progress: IcoProgress[int] = IcoProgress(total=10)
    run(numbers(3) | progress.stream(), tool)

    second = snapshots[2]  # after the initial one: 2 items, 1 s since start
    assert (second["done"], second["rate"], second["eta"]) == (2.0, 2.0, 4.0)


def test_throttle_per_node_and_final_state() -> None:
    # 0.1 s per event, at most 2 snapshots/s: one every 5 events, plus final.
    snapshots, tool = collect(max_rate=2.0, clock=FakeClock(0.1))
    progress: IcoProgress[int] = IcoProgress(total=10)
    run(numbers(10) | progress.stream(), tool)

    assert [s["done"] for s in snapshots] == [0.0, 1.0, 6.0, 10.0]


def test_no_duplicate_final_snapshot() -> None:
    snapshots, tool = collect(max_rate=1000, clock=FakeClock(1.0))
    progress: IcoProgress[int] = IcoProgress(total=2)
    run(numbers(2) | progress.stream(), tool)

    assert [s["done"] for s in snapshots] == [0.0, 1.0, 2.0]


def test_next_iteration_starts_from_zero() -> None:
    snapshots, tool = collect(max_rate=1000, clock=FakeClock(1.0))
    progress: IcoProgress[int] = IcoProgress(total=3)

    def tick(x: int) -> int:
        return x

    def start(_: None) -> int:
        return 0

    body = IcoOperator(tick) | progress
    closure = (
        IcoOperator(start) | IcoProcess(body, num_iterations=6) | IcoOperator(ignore)
    )
    IcoRuntime(closure, tools=[tool]).activate().run().deactivate()

    assert [s["done"] for s in snapshots] == [0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]


def test_labels_from_flow() -> None:
    snapshots, tool = collect(max_rate=1000, clock=FakeClock(1.0))
    named: IcoProgress[int] = IcoProgress(total=1, name="named")
    first: IcoProgress[int] = IcoProgress(total=1)
    second: IcoProgress[int] = IcoProgress(total=1)
    run(numbers(1) | (named | first | second).stream(), tool)

    labels = {str(s["key"]): str(s["label"]) for s in snapshots}
    assert sorted(labels.values()) == ["Progress 1", "Progress 2", "named"]


def test_initial_snapshots_list_all_nodes_in_flow_order() -> None:
    snapshots, tool = collect(max_rate=1000, clock=FakeClock(1.0))
    first: IcoProgress[int] = IcoProgress(total=2, name="first")
    second: IcoProgress[int] = IcoProgress(total=2, name="second")
    run(numbers(2) | (first | second).stream(), tool)

    initial = snapshots[:2]
    assert [s["label"] for s in initial] == ["first", "second"]
    assert [s["done"] for s in initial] == [0.0, 0.0]
    assert len(snapshots) > 2
    assert all(s["done"] != 0.0 for s in snapshots[2:])


def test_invalid_max_rate() -> None:
    with pytest.raises(ValueError, match="max_rate"):
        JsonProgressTool(ignore, max_rate=0)


# ─── Progress inside MPWorker runner processes ───


def make_worker_flow() -> IcoOperator[int, int]:
    progress: IcoProgress[int] = IcoProgress(total=100, name="Worker")

    def double(x: int) -> int:
        return x * 2

    return progress | IcoOperator(double)  # type: ignore[return-value]


def test_worker_progress_has_own_keys() -> None:
    snapshots, tool = collect(max_rate=1000)
    pool = IcoAsyncStream(lambda: MPWorker(make_worker_flow), pool_size=2)
    run(numbers(6) | pool, tool)

    finals: dict[str, float] = {}
    for snapshot in snapshots:
        assert snapshot["label"] == "Worker"
        finals[str(snapshot["key"])] = float(snapshot["done"] or 0)
    assert 1 <= len(finals) <= 2  # one key per worker that processed items
    assert sum(finals.values()) == 6


def test_unknown_total_counts_up_without_eta() -> None:
    snapshots, tool = collect(max_rate=1000, clock=FakeClock(0.1))
    progress: IcoProgress[int] = IcoProgress(name="scenes")
    run(numbers(5) | progress.stream(), tool)

    assert [s["done"] for s in snapshots] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert all(s["total"] is None and s["eta"] is None for s in snapshots)
    assert snapshots[-1]["rate"] is not None


def test_total_must_be_positive() -> None:
    with pytest.raises(ValueError, match="use no total"):
        IcoProgress(total=0)

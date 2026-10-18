"""Cooperative cancellation: a run stops between items, from any thread;
the item in progress finishes, workers stop cleanly, results so far are
kept."""

import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from ico.core.async_stream import IcoAsyncStream
from ico.core.operator import IcoOperator, IcoOperatorProtocol
from ico.core.process import IcoProcess
from ico.core.runtime.cancel import IcoCancelled, IcoCancelToken, check_cancelled
from ico.core.runtime.progress import IcoProgress
from ico.core.runtime.runtime import IcoRuntime
from ico.core.sink import IcoSink
from ico.core.source import IcoSource
from ico.core.stream import IcoStream
from ico.registry import IcoRegistry, register
from ico.runtime.worker.mp.mp_worker import MPWorker
from ico.spec import NodeSpec, run
from ico.spec.model import JsonValue
from ico.tools.progress.json_progress_tool import JsonProgressTool, ProgressSnapshot

# Module level: spawned runner processes import this module.
REGISTRY = IcoRegistry()
N = 100
STEP = 0.03


@register("cancel.slow", 1, registry=REGISTRY)
def slow(x: int) -> int:
    time.sleep(STEP)
    return x


def zero(_: None) -> int:
    return 0


def discard(_: int) -> None:
    pass


def inc(x: int) -> int:
    return x + 1


def numbers() -> Iterator[int]:
    yield from range(N)


def later(runtime_or_token: Any, delay: float, reason: str = "stop") -> None:
    threading.Timer(delay, runtime_or_token.cancel, args=(reason,)).start()


def run_and_cancel(flow: Any, delay: float) -> tuple[list[int], float]:
    out: list[int] = []
    runtime = IcoRuntime(flow | IcoSink(out.append))
    runtime.activate()
    later(runtime, delay)
    clock = time.monotonic()
    try:
        with pytest.raises(IcoCancelled, match="stop"):
            runtime.run()
    finally:
        runtime.deactivate()
    return out, time.monotonic() - clock


def test_a_stream_stops_between_items() -> None:
    out, elapsed = run_and_cancel(
        IcoSource(numbers) | IcoStream(IcoOperator(slow)), 0.15
    )
    assert 0 < len(out) < 15
    assert out == list(range(len(out)))  # every item finished, none lost
    assert elapsed < 1


def test_cancelled_before_run_processes_nothing() -> None:
    out: list[int] = []
    runtime = IcoRuntime(IcoSource(numbers) | IcoSink(out.append))
    runtime.cancel("early")
    runtime.activate()
    with pytest.raises(IcoCancelled, match="early"):
        runtime.run()
    runtime.deactivate()
    assert out == []


def test_a_thread_pool_stops_scheduling() -> None:
    pool = IcoAsyncStream(lambda: IcoOperator(slow), pool_size=2)
    out, elapsed = run_and_cancel(IcoSource(numbers) | pool, 0.15)
    assert 0 < len(out) < 20
    assert elapsed < 1


def worker_flow() -> IcoOperatorProtocol[int, int]:
    return IcoOperator(slow)


def make_worker() -> MPWorker[int, int]:
    return MPWorker(worker_flow)


def test_workers_stop_cleanly() -> None:
    pool = IcoAsyncStream(make_worker, pool_size=2)
    out, _ = run_and_cancel(IcoSource(numbers) | pool, 1.0)
    assert len(out) < N
    assert not any(cast_worker(w).is_alive for w in pool.pool)


def cast_worker(node: object) -> MPWorker[Any, Any]:
    assert isinstance(node, MPWorker)
    return node  # pyright: ignore[reportUnknownVariableType]


def test_a_process_stops_between_iterations() -> None:
    iterations: list[int] = []

    def step(n: int) -> int:
        iterations.append(n)
        time.sleep(STEP)
        return n + 1

    process = IcoProcess(step, num_iterations=N)
    runtime = IcoRuntime(IcoOperator(zero) | process | IcoOperator(discard))
    runtime.activate()
    later(runtime, 0.15)
    with pytest.raises(IcoCancelled):
        runtime.run()
    runtime.deactivate()
    assert 0 < len(iterations) < 15


def test_spec_run_returns_the_results_so_far() -> None:
    progress: JsonValue = {"kind": "progress", "name": "items", "params": {"total": N}}
    body: JsonValue = {
        "kind": "seq",
        "children": [progress, {"kind": "op", "op": "cancel.slow@1"}],
    }
    spec = NodeSpec.from_dict({"kind": "stream", "children": [body]})
    token = IcoCancelToken()
    snapshots: list[ProgressSnapshot] = []
    later(token, 0.15, "the user stopped it")

    result = run(
        spec,
        list(range(N)),
        registry=REGISTRY,
        tools=[JsonProgressTool(snapshots.append)],
        cancel=token,
    )

    assert result.cancelled == "the user stopped it"
    assert 0 < result.total < N
    assert result.to_dict()["cancelled"] == "the user stopped it"
    # Progress counts the items that entered; the last one may be in progress.
    assert snapshots[-1]["done"] in (result.total, result.total + 1)


def test_a_finished_run_is_not_marked_cancelled() -> None:
    spec = NodeSpec.from_dict({"kind": "op", "op": "cancel.slow@1"})
    result = run(spec, 1, registry=REGISTRY)
    assert result.cancelled is None and "cancelled" not in result.to_dict()


def test_without_a_runtime_nothing_is_checked() -> None:
    check_cancelled()  # no current token: no-op
    flow = IcoStream(IcoOperator(inc))
    assert list(flow(iter([1, 2]))) == [2, 3]


def test_the_first_reason_stays() -> None:
    token = IcoCancelToken()
    assert not token.cancelled
    token.cancel("first")
    token.cancel("second")
    assert token.cancelled and token.reason == "first"


def test_progress_after_cancel_is_flushed() -> None:
    progress: IcoProgress[int] = IcoProgress(total=N, interval=10)
    snapshots: list[ProgressSnapshot] = []
    out: list[int] = []
    flow = IcoSource(numbers) | IcoStream(progress | IcoOperator(slow))
    runtime = IcoRuntime(
        flow | IcoSink(out.append), tools=[JsonProgressTool(snapshots.append)]
    )
    runtime.activate()
    later(runtime, 0.15)
    with pytest.raises(IcoCancelled):
        runtime.run()
    runtime.deactivate()  # the progress counted so far is reported
    assert snapshots[-1]["done"] in (len(out), len(out) + 1)

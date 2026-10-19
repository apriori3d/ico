"""An item that fails in a worker fails alone: the worker and its runner go
on with the next items; a dead runner still faults the worker."""

from collections.abc import Iterator

import pytest

from ico.core.async_stream import IcoAsyncStream, IcoStreamErrors
from ico.core.operator import IcoOperator, IcoOperatorProtocol
from ico.core.runtime.exceptions import IcoRuntimeError
from ico.core.runtime.runtime import IcoRuntime
from ico.core.runtime.state import ReadyState
from ico.core.sink import IcoSink
from ico.core.source import IcoSource
from ico.core.stream import IcoStream
from ico.runtime.worker.mp.mp_worker import MPWorker

# Module level: spawned runner processes import this module.
N = 10


def numbers() -> Iterator[int]:
    yield from range(N)


def fail_on_three(x: int) -> int:
    if x == 3:
        raise ValueError("bad item 3")
    return x * 10


def worker_flow() -> IcoOperatorProtocol[int, int]:
    return IcoOperator(fail_on_three)


def make_worker() -> MPWorker[int, int]:
    return MPWorker(worker_flow)


def test_one_item_error_in_a_pool_is_one_error() -> None:
    out: list[int] = []
    pool = IcoAsyncStream(make_worker, pool_size=1, on_error="collect")
    runtime = IcoRuntime(IcoSource(numbers) | pool | IcoSink(out.append))
    runtime.activate()
    try:
        with pytest.raises(IcoStreamErrors) as info:
            runtime.run()
    finally:
        runtime.deactivate()

    assert [e.index for e in info.value.errors] == [3]
    assert "bad item 3" in str(info.value.errors[0].error)
    assert out == [x * 10 for x in range(N) if x != 3]


def test_the_worker_is_ready_after_an_item_error() -> None:
    worker = make_worker()
    seen: list[int] = []

    def take(x: int) -> None:
        seen.append(x)

    runtime = IcoRuntime(IcoSource(numbers) | IcoStream(worker) | IcoSink(take))
    runtime.activate()
    try:
        with pytest.raises(IcoRuntimeError, match="bad item 3"):
            runtime.run()
        assert isinstance(worker.state, ReadyState)
        assert worker(4) == 40  # the same runner takes the next item
    finally:
        runtime.deactivate()
    assert seen == [0, 10, 20]

"""An async stream behind a slow synchronous upstream (a GPU stage before a
CPU pool) keeps dispatching and emitting while the upstream computes, and
two async streams can follow each other."""

import time
from collections.abc import Iterator

from ico.core.async_stream import IcoAsyncStream
from ico.core.operator import IcoOperator
from ico.core.runtime.runtime import IcoRuntime
from ico.core.sink import IcoSink
from ico.core.source import IcoSource
from ico.core.stream import IcoStream

N = 4
SLOW = 0.4  # the upstream, per item
FAST = 0.05  # the pool, per item


def numbers() -> Iterator[int]:
    yield from range(N)


def gpu(x: int) -> int:
    time.sleep(SLOW)
    return x


def cpu(x: int) -> int:
    time.sleep(FAST)
    return x * 10


def test_outputs_do_not_wait_for_the_next_input() -> None:
    produced: dict[int, float] = {}
    emitted: dict[int, float] = {}

    def mark(x: int) -> int:
        produced[x] = time.monotonic()
        return x

    def take(y: int) -> None:
        emitted[y // 10] = time.monotonic()

    upstream = IcoStream(IcoOperator(gpu) | IcoOperator(mark))
    pool = IcoAsyncStream(lambda: IcoOperator(cpu), pool_size=2)
    flow = IcoSource(numbers) | upstream | pool | IcoSink(take)
    IcoRuntime(flow).activate().run().deactivate()

    assert sorted(emitted) == list(range(N))
    # Each output follows its input after the pool's work, not after the
    # upstream's next item (a lag of SLOW).
    lags = [emitted[x] - produced[x] for x in range(N)]
    assert max(lags) < SLOW / 2, lags


def test_two_async_streams_in_a_row() -> None:
    out: list[int] = []
    first = IcoAsyncStream(lambda: IcoOperator(cpu), pool_size=2)
    second = IcoAsyncStream(lambda: IcoOperator(cpu), pool_size=2)
    flow = IcoSource(numbers) | first | second | IcoSink(out.append)
    IcoRuntime(flow).activate().run().deactivate()

    assert sorted(out) == [x * 100 for x in range(N)]

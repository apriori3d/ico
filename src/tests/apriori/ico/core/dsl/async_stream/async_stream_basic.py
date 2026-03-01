import asyncio
import random
import time

import pytest

from apriori.ico.core.async_operator import IcoAsyncOperator
from apriori.ico.core.async_stream import IcoAsyncStream
from apriori.ico.core.operator import IcoOperator

# ───────────────────────────────────────────────
#  Test: basic synchronous processing
# ───────────────────────────────────────────────

IntOperator = IcoOperator[int, int]


def test_parallel_stream_basic() -> None:
    """Ensure all items are processed by ParallelStream."""
    ops = [IntOperator(lambda x: x * 2) for _ in range(3)]
    stream = IcoAsyncStream(ops)

    data = [1, 2, 3, 4, 5]
    result = list(stream(iter(data)))
    assert result == [x * 2 for x in data]


# ───────────────────────────────────────────────
#  Test: async parallel execution and timing
# ───────────────────────────────────────────────


def test_parallel_stream_parallel_execution() -> None:
    """Simulate random processing delays to ensure concurrency."""

    async def delayed_op(x: int) -> int:
        await asyncio.sleep(random.uniform(0.05, 0.2))
        return x * 10

    ops = [IcoAsyncOperator(delayed_op) for _ in range(3)]
    stream = IcoAsyncStream(ops)

    data = list(range(10))
    t0 = time.perf_counter()
    result = list(stream(iter(data)))
    t1 = time.perf_counter()

    # All items processed
    assert len(result) == len(data)
    # Should complete faster than sequential 10 * 0.1s = 1s
    assert (t1 - t0) < 0.8


# ───────────────────────────────────────────────
#  Test: ordered results are preserved
# ───────────────────────────────────────────────


def test_parallel_stream_ordered() -> None:
    """Test ordered=True preserves order of results."""

    def delayed_double(x: int) -> int:
        time.sleep(0.05 if x % 2 == 0 else 0.01)
        return x * 2

    ops = [IcoOperator(delayed_double) for _ in range(3)]
    stream = IcoAsyncStream(ops, ordered=True)

    data = [1, 2, 3, 4, 5, 6]
    result = list(stream(iter(data)))

    assert result == [x * 2 for x in data]


# ───────────────────────────────────────────────
#  Test: unordered results follow completion order
# ───────────────────────────────────────────────


def test_parallel_stream_unordered() -> None:
    """Test ordered=False changes order of results."""

    async def delayed_double(x: float) -> float:
        await asyncio.sleep(x * 0.01)
        return x * 2

    # Even though the input is reversed, unordered execution yields results
    # in completion order — effectively re-sorted by async timing.

    data = [1, 2, 3, 4, 5, 6]
    ops = [IcoAsyncOperator[int, float](delayed_double) for _ in range(len(data))]
    stream = IcoAsyncStream[int, float](ops, ordered=False)
    result = [item for item in stream(reversed(data))]

    # Order differs, but all outputs are correct
    assert sorted(result) == [x * 2 for x in data]  # pyright: ignore[reportArgumentType]
    assert result != [x * 2 for x in reversed(data)]


# ───────────────────────────────────────────────
#  Test: exceptions propagate correctly
# ───────────────────────────────────────────────


def test_parallel_stream_exception() -> None:
    """Test that exceptions in operators are propagated."""

    def faulty_op(x: int) -> int:
        if x == 3:
            raise ValueError("boom")
        return x

    ops = [IcoOperator(faulty_op) for _ in range(2)]
    stream = IcoAsyncStream(ops)

    data = [1, 2, 3, 4]

    with pytest.raises(ValueError):
        list(stream(iter(data)))


# ───────────────────────────────────────────────
#  Test: unordered stream raises immediately on failure
# ───────────────────────────────────────────────


def test_parallel_stream_unordered_raises_immediately() -> None:
    """Ensure exceptions from unordered workers propagate immediately."""

    async def maybe_fail(x: int) -> int:
        await asyncio.sleep(0.01)
        if x == 2:
            raise RuntimeError("failure")
        return x * 2

    ops = [IcoAsyncOperator(maybe_fail) for _ in range(3)]
    stream = IcoAsyncStream(ops, ordered=False)
    data = [1, 2, 3]

    with pytest.raises(RuntimeError):
        list(stream(iter(data)))


# ───────────────────────────────────────────────
#  Test: async operators execute transparently
# ───────────────────────────────────────────────


def test_parallel_stream_async_operator() -> None:
    """Ensure async operators work transparently."""

    async def async_double(x: int) -> int:
        await asyncio.sleep(0.01)
        return x * 2

    ops = [IcoAsyncOperator(async_double) for _ in range(2)]
    stream = IcoAsyncStream(ops)

    data = [1, 2, 3, 4]
    result = list(stream(iter(data)))
    assert sorted(result) == sorted([x * 2 for x in data])  # pyright: ignore[reportArgumentType]


# ───────────────────────────────────────────────
#  Test: empty input stream exits immediately
# ───────────────────────────────────────────────


def test_parallel_stream_empty_input() -> None:
    """Verify that an empty input stream triggers fast-exit."""

    ops = [IcoOperator[int, int](lambda x: x) for _ in range(2)]
    stream = IcoAsyncStream(ops)
    result = list(stream(iter([])))
    assert result == []


# ───────────────────────────────────────────────
#  Test: single operator and single item
# ───────────────────────────────────────────────


def test_parallel_stream_single_operator_single_item() -> None:
    """Minimal path: one worker, one input."""

    async def op(x: int) -> int:
        await asyncio.sleep(0.01)
        return x + 1

    stream = IcoAsyncStream([IcoAsyncOperator(op)])
    result = list(stream(iter([10])))
    assert result == [11]


# ───────────────────────────────────────────────
#  Test: slow worker should not block fast ones (unordered mode)
# ───────────────────────────────────────────────


def test_parallel_stream_slow_one_does_not_block() -> None:
    """Ensure unordered execution yields results without waiting for slow tasks."""

    async def slow_or_fast(x: int) -> int:
        if x == 0:
            await asyncio.sleep(0.05)  # intentionally slow
        else:
            await asyncio.sleep(0.001)
        return x

    ops = [IcoAsyncOperator(slow_or_fast) for _ in range(3)]
    stream = IcoAsyncStream(ops, ordered=False)
    data = [0, 1, 2, 3]

    result = list(stream(iter(data)))
    assert sorted(result) == data  # pyright: ignore[reportArgumentType]
    assert result[0] != 0  # ensure non-blocking behavior


# ───────────────────────────────────────────────
#  Test: stream can be reused between runs
# ───────────────────────────────────────────────


def test_parallel_stream_can_be_reused() -> None:
    """Ensure internal runtime state resets between runs."""

    async def f(x: int) -> int:
        await asyncio.sleep(0.001)
        return x + 1

    ops = [IcoAsyncOperator(f) for _ in range(2)]
    stream = IcoAsyncStream(ops)
    data = [1, 2, 3]

    first_run = list(stream(iter(data)))
    second_run = list(stream(iter(data)))

    assert first_run == [2, 3, 4]
    assert second_run == [2, 3, 4]


# ───────────────────────────────────────────────
#  Test: mixed sync + async operators
# ───────────────────────────────────────────────


def test_parallel_stream_mixed_sync_async() -> None:
    """Verify mixed sync/async operators are executed correctly."""

    async def async_double_slow(x: int) -> int:
        await asyncio.sleep(0.1)
        return x * 2

    def sync_triple(x: int) -> int:
        return x * 3

    # Mix of sync and async workers.
    # Asssume async_double_slow should get only first item, second operator the rest.
    ops: list[IcoAsyncOperator[int, int] | IcoOperator[int, int]] = [
        IcoAsyncOperator(async_double_slow),
        IcoOperator(sync_triple),
    ]
    stream = IcoAsyncStream[int, int](ops)

    data = [1, 2, 3, 4]
    result = sorted(list(stream(iter(data))))
    expected = sorted([x * 2 for x in data[:1]] + [x * 3 for x in data[1:]])

    assert result == expected


# ───────────────────────────────────────────────
#  Test: parallel speedup measurement
# ───────────────────────────────────────────────


def test_parallel_stream_parallel_speedup() -> None:
    """Ensure that parallel execution actually speeds up total runtime."""

    async def slow_double(x: int) -> int:
        await asyncio.sleep(0.05)
        return x * 2

    data = list(range(6))
    ops = [IcoAsyncOperator[int, int](slow_double) for _ in range(len(data))]
    stream = IcoAsyncStream[int, int](ops)

    start = time.perf_counter()
    result = list(stream(iter(data)))
    duration = time.perf_counter() - start

    assert result == [x * 2 for x in data]
    # Full concurrency must reduce total duration
    assert duration < 0.25


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))

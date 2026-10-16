import asyncio
import time
from collections.abc import Generator, Iterator
from typing import Any, cast

import pytest

from ico.core.async_operator import IcoAsyncOperator
from ico.core.async_stream import IcoAsyncStream
from ico.core.operator import IcoOperator


async def async_double(x: int) -> int:
    await asyncio.sleep(0.001 * (x % 3))
    return x * 2


def make_stream(ordered: bool = False, workers: int = 3) -> IcoAsyncStream[int, int]:
    return IcoAsyncStream(
        [IcoAsyncOperator(async_double) for _ in range(workers)], ordered=ordered
    )


# ─── Queue accounting ───


class CountingQueue(asyncio.Queue[Any]):
    created: list["CountingQueue"] = []

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize)
        CountingQueue.created.append(self)


@pytest.mark.parametrize("ordered", [False, True])
def test_task_done_called_once_per_get(
    monkeypatch: pytest.MonkeyPatch, ordered: bool
) -> None:
    CountingQueue.created = []
    monkeypatch.setattr(asyncio, "Queue", CountingQueue)

    assert sorted(make_stream(ordered)(iter(range(10)))) == [x * 2 for x in range(10)]

    input_queue, results_queue = CountingQueue.created
    # Every get(), including the termination signals, is matched by task_done().
    assert input_queue._unfinished_tasks == 0  # type: ignore[attr-defined]
    assert results_queue._unfinished_tasks == 0  # type: ignore[attr-defined]


# ─── Termination ───


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 50])
def test_all_items_emitted_for_any_size(ordered: bool, n: int) -> None:
    result = list(make_stream(ordered)(iter(range(n))))
    expected = [x * 2 for x in range(n)]
    assert (result if ordered else sorted(result)) == expected


def test_last_result_with_lagging_workers_does_not_hang() -> None:
    # One fast item and many idle workers: the consumer must stop once all
    # outcomes are received, regardless of when workers see their sentinel.
    for _ in range(20):
        assert list(make_stream(workers=8)(iter([1]))) == [2]


def test_consumer_can_stop_early() -> None:
    out = make_stream(workers=2)(iter(range(100)))
    assert next(out) in {x * 2 for x in range(100)}
    # The stream is a generator; closing it must cancel workers without hanging.
    cast(Generator[int, None, None], out).close()


# ─── Input iterator errors ───


def failing_input(n: int) -> Iterator[int]:
    yield from range(n)
    raise OSError("source broke")


@pytest.mark.parametrize("ordered", [False, True])
def test_input_error_raised_after_scheduled_items(ordered: bool) -> None:
    emitted: list[int] = []
    with pytest.raises(OSError, match="source broke"):
        for value in make_stream(ordered)(failing_input(5)):
            emitted.append(value)
    assert sorted(emitted) == [0, 2, 4, 6, 8]


def test_input_error_on_first_item() -> None:
    with pytest.raises(OSError):
        list(make_stream()(failing_input(0)))


# ─── Item errors ───


def test_ordered_error_raised_at_its_position() -> None:
    def fail_on_three(x: int) -> int:
        if x == 3:
            raise ValueError("boom")
        time.sleep(0.01 if x < 3 else 0)
        return x

    stream = IcoAsyncStream(
        [IcoOperator(fail_on_three) for _ in range(3)], ordered=True
    )
    emitted: list[int] = []
    with pytest.raises(ValueError, match="boom"):
        for value in stream(iter(range(6))):
            emitted.append(value)
    assert emitted == [0, 1, 2]


def test_exception_instances_as_results_are_not_errors() -> None:
    def make_error(x: int) -> Exception:
        return ValueError(x)

    stream = IcoAsyncStream([IcoOperator(make_error)], ordered=True)
    result = list(stream(iter([1, 2])))
    assert [type(r) for r in result] == [ValueError, ValueError]


# ─── Reentrancy ───


@pytest.mark.parametrize("ordered", [False, True])
def test_interleaved_runs_of_one_stream(ordered: bool) -> None:
    stream = make_stream(ordered)
    first = stream(iter(range(0, 10)))
    second = stream(iter(range(100, 110)))

    a: list[int] = []
    b: list[int] = []
    for _ in range(10):
        a.append(next(first))
        b.append(next(second))

    expected_a = [x * 2 for x in range(0, 10)]
    expected_b = [x * 2 for x in range(100, 110)]
    if ordered:
        assert (a, b) == (expected_a, expected_b)
    else:
        assert (sorted(a), sorted(b)) == (expected_a, expected_b)

from __future__ import annotations

import time

import pytest

from ico.core.async_stream import IcoAsyncStream
from ico.core.operator import IcoOperator
from ico.core.source import IcoSource
from ico.runtime.worker.mp.mp_worker import MPWorker


def flow_identity() -> IcoOperator[int, int]:
    """Simple identity operator."""
    return IcoOperator[int, int](lambda x: x)


def double_flow_factory() -> IcoOperator[int, int]:
    """Doubles numeric input."""
    return IcoOperator[int, int](lambda x: x * 2)


def slow_tripple_flow_factory() -> IcoOperator[int, int]:
    """Doubles numeric input."""

    def _slow_tripple(x: int) -> int:
        time.sleep(0.1)
        return x * 3

    return IcoOperator[int, int](_slow_tripple)


# ───────────────────────────────────────────────
#  Test: Basic roundtrip
# ───────────────────────────────────────────────


def test_single_mp_process_round_trip() -> None:
    """Ensure data passes through full async stream and multiprocessing roundtrip."""
    num = 5
    data = list(range(1, num + 1))
    worker = MPWorker(double_flow_factory).activate()
    try:
        source = IcoSource[int](lambda: iter(data))
        flow = source | IcoAsyncStream([worker])
        result = list(flow(None))
        assert result == [d * 2 for d in data]
    finally:
        worker.deactivate()


def test_slow_then_fast_unordered_mp_process_round_trip() -> None:
    """Ensure data passes through full async stream and multiprocessing roundtrip."""
    worker1 = MPWorker(slow_tripple_flow_factory).activate()
    worker2 = MPWorker(double_flow_factory).activate()
    data = [1, 2]

    try:
        source = IcoSource[int](lambda: iter(data))
        flow = source | IcoAsyncStream([worker1, worker2], ordered=False)
        result = list(flow(None))
        assert result == [4, 3]

    finally:
        worker1.deactivate()
        worker2.deactivate()


def test_slow_then_fast_ordered_mp_process_round_trip() -> None:
    """Ensure data passes through full async stream and multiprocessing roundtrip."""
    worker1 = MPWorker(slow_tripple_flow_factory).activate()
    worker2 = MPWorker(double_flow_factory).activate()
    data = [1, 2]

    try:
        source = IcoSource[int](lambda: iter(data))
        flow = source | IcoAsyncStream([worker1, worker2], ordered=True)
        result = list(flow(None))
        assert result == [3, 4]

    finally:
        worker1.deactivate()
        worker2.deactivate()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))

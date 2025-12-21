from __future__ import annotations

import time

import pytest

from apriori.ico.core.async_stream import IcoAsyncStream
from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.source import IcoSource
from apriori.ico.runtime.agent.mp.mp_agent import MPAgent

# ───────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────


def flow_identity() -> IcoOperator[int, int]:
    """Simple identity operator."""
    return IcoOperator[int, int](lambda x: x)


def flow_double() -> IcoOperator[int, int]:
    """Doubles numeric input."""
    return IcoOperator[int, int](lambda x: x * 2)


def flow_tripple_fn(x: int) -> int:
    """Triples numeric input."""
    time.sleep(0.1)
    return x * 3


def flow_slow_tripple() -> IcoOperator[int, int]:
    """Triples numeric input with a delay."""
    return IcoOperator[int, int](flow_tripple_fn)


# ───────────────────────────────────────────────
#  Test: Basic roundtrip
# ───────────────────────────────────────────────


def test_single_mp_process_round_trip() -> None:
    """Ensure data passes through full async stream and multiprocessing roundtrip."""
    num = 5
    data = list(range(1, num + 1))
    mp_process = MPAgent(flow_double).activate()
    try:
        source = IcoSource(lambda: iter(data))
        flow = source | IcoAsyncStream([mp_process])
        result = list(flow(None))
        assert result == [d * 2 for d in data]
    finally:
        mp_process.deactivate()


def test_slow_then_fast_unordered_mp_process_round_trip() -> None:
    """Ensure data passes through full async stream and multiprocessing roundtrip."""

    pool = [
        MPAgent(flow_slow_tripple, name="slow_tripple").activate(),
        MPAgent(flow_double, name="double").activate(),
    ]

    try:
        data = [1, 2]
        source = IcoSource(lambda: iter(data))
        flow = source | IcoAsyncStream(pool, ordered=False)
        result = list(flow(None))
        assert result == [4, 3]
    finally:
        for process in pool:
            process.deactivate()


def test_slow_then_fast_ordered_mp_process_round_trip() -> None:
    """Ensure data passes through full async stream and multiprocessing roundtrip."""
    num = 2
    pool = [
        MPAgent(flow_slow_tripple).activate(),
        MPAgent(flow_double).activate(),
    ]
    try:
        data = list(range(1, num + 1))
        source = IcoSource(lambda: iter(data))
        flow = source | IcoAsyncStream(pool, ordered=True)
        result = list(flow(None))
        assert result == [3, 4]

    finally:
        for process in pool:
            process.deactivate()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))

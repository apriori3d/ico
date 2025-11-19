from __future__ import annotations

import time
from collections.abc import Callable
from multiprocessing import get_context
from multiprocessing.context import SpawnProcess

import pytest

from apriori.ico.channels.mp_queue.channel import MPQueueChannel
from apriori.ico.core.dsl.async_stream import IcoAsyncStream
from apriori.ico.core.dsl.operator import IcoOperator
from apriori.ico.core.dsl.source import IcoSource
from apriori.ico.core.runtime.channels.channel import IcoRuntimeChannel
from apriori.ico.core.types import I, O

# ───────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────


def agent(
    channel: IcoRuntimeChannel[I, O],
    flow_factory: Callable[[], IcoOperator[O, I]],
    n: int = 1,
) -> None:
    """Simulated remote process executing receive → flow → send loop."""
    flow = flow_factory()
    closure = channel.receive | flow | channel.send
    count = 0
    while count < n:
        closure(None)
        count += 1


def flow_identity() -> IcoOperator[int, int]:
    """Simple identity operator."""
    return IcoOperator[int, int](lambda x: x)


def flow_double() -> IcoOperator[int, int]:
    """Doubles numeric input."""
    return IcoOperator[int, int](lambda x: x * 2)


def flow_slow_tripple() -> IcoOperator[int, int]:
    """Doubles numeric input."""
    time.sleep(0.1)
    return IcoOperator[int, int](lambda x: x * 3)


def start_mp_process_agent(
    channel: MPQueueChannel[I, O],
    flow_factory: Callable[[], IcoOperator[I, O]],
    n: int | None = 1,
) -> SpawnProcess:
    """Launch isolated process that runs an agent closure."""
    p = channel.mp_context.Process(target=agent, args=(channel, flow_factory, n))
    p.start()
    time.sleep(0.05)
    return p


# ───────────────────────────────────────────────
#  Test: Basic roundtrip
# ───────────────────────────────────────────────


def test_single_mp_process_round_trip() -> None:
    """Ensure data passes through full async stream and multiprocessing roundtrip."""
    channel = MPQueueChannel[int, int](get_context("spawn"))
    num = 5
    process = start_mp_process_agent(channel.make_pair(), flow_double, n=num)
    data = list(range(1, num + 1))
    try:
        mp_flow = channel.send | channel.receive
        source = IcoSource(lambda _: iter(data))
        flow = source | IcoAsyncStream([mp_flow])
        result = list(flow(None))
        assert result == [d * 2 for d in data]
    finally:
        process.terminate()
        process.join(timeout=0.5)
    assert not process.is_alive(), "Agent process did not exit cleanly"


def create_mp_process(
    flows: list[Callable[[], IcoOperator[O, I]]],
    num: int,
) -> tuple[list[SpawnProcess], list[IcoOperator[I, O]]]:
    mp_flows: list[IcoOperator[I, O]] = []
    mp_processes: list[SpawnProcess] = []

    for flow in flows:
        channel = MPQueueChannel[I, O](get_context("spawn"))
        process = start_mp_process_agent(channel.make_pair(), flow, n=num)
        mp_flow = channel.send | channel.receive
        mp_processes.append(process)
        mp_flows.append(mp_flow)

    return mp_processes, mp_flows


def test_slow_then_fast_unordered_mp_process_round_trip() -> None:
    """Ensure data passes through full async stream and multiprocessing roundtrip."""
    num = 2
    processes, flows = create_mp_process([flow_slow_tripple, flow_double], num=1)
    data = list(range(1, num + 1))
    try:
        source = IcoSource(lambda _: iter(data))
        flow = source | IcoAsyncStream(flows)
        result = list(flow(None))
        assert result == [4, 3]

    finally:
        for process in processes:
            process.terminate()
            process.join(timeout=0.5)
            assert not process.is_alive(), "Agent process did not exit cleanly"


def test_slow_then_fast_ordered_mp_process_round_trip() -> None:
    """Ensure data passes through full async stream and multiprocessing roundtrip."""
    num = 2
    processes, flows = create_mp_process([flow_slow_tripple, flow_double], num=1)
    data = list(range(1, num + 1))
    try:
        source = IcoSource(lambda _: iter(data))
        flow = source | IcoAsyncStream(flows, ordered=True)
        result = list(flow(None))
        assert result == [3, 4]

    finally:
        for process in processes:
            process.terminate()
            process.join(timeout=0.5)
            assert not process.is_alive(), "Agent process did not exit cleanly"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))

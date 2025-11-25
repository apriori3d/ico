from __future__ import annotations

import time
from collections.abc import Callable
from multiprocessing import get_context
from multiprocessing.context import SpawnProcess
from typing import Generic

import pytest

from apriori.ico.core.async_stream import IcoAsyncStream
from apriori.ico.core.operator import I, IcoOperator, O
from apriori.ico.core.runtime.channel.channel import IcoRuntimeChannel
from apriori.ico.core.runtime.channel.utils import wait_for_item
from apriori.ico.core.source import IcoSource
from apriori.ico.runtime.channel.mp_queue.channel import MPQueueChannel

# ───────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────


def agent(
    channel: IcoRuntimeChannel[O, I],
    flow_factory: Callable[[], IcoOperator[I, O]],
    n: int = 1,
) -> None:
    """Simulated remote process executing receive → flow → send loop."""
    flow = flow_factory()

    count = 0
    while count < n:
        input = wait_for_item(
            endpoint=channel.input,
            accept_commands=False,
            accept_events=False,
        )
        assert input is not None
        result = flow(input)
        channel.output.send(result)
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
    return p


def create_mp_process(
    flows: list[Callable[[], IcoOperator[O, I]]],
    num: int,
) -> tuple[list[SpawnProcess], list[IcoOperator[I, O]]]:
    mp_flows: list[IcoOperator[I, O]] = []
    mp_processes: list[SpawnProcess] = []

    for flow in flows:
        channel = MPQueueChannel[I, O](get_context("spawn"))
        process = start_mp_process_agent(channel.make_pair(), flow, n=num)
        mp_process_mock = MPProcessMock(channel)
        mp_processes.append(process)
        mp_flows.append(mp_process_mock)

    return mp_processes, mp_flows


class MPProcessMock(Generic[I, O], IcoOperator[I, O]):
    def __init__(
        self,
        channel: MPQueueChannel[I, O],
    ) -> None:
        super().__init__(
            fn=self._portal_fn,
        )
        self._channel = channel

    def _portal_fn(self, input: I) -> O:
        # Send item to agent process
        self._channel.output.send(input)
        item = wait_for_item(
            endpoint=self._channel.input,
            accept_commands=False,
            accept_events=False,
        )
        assert item is not None
        return item


# ───────────────────────────────────────────────
#  Test: Basic roundtrip
# ───────────────────────────────────────────────


def test_single_mp_process_round_trip() -> None:
    """Ensure data passes through full async stream and multiprocessing roundtrip."""
    channel = MPQueueChannel[int, int](get_context("spawn"))
    num = 5
    process = start_mp_process_agent(channel.make_pair(), flow_double, n=num)
    data = list(range(1, num + 1))
    mp_process_mock = MPProcessMock(channel)
    try:
        source = IcoSource(lambda _: iter(data))
        flow = source | IcoAsyncStream([mp_process_mock])
        result = list(flow(None))
        assert result == [d * 2 for d in data]
    finally:
        process.terminate()
        process.join(timeout=0.5)
    assert not process.is_alive(), "Agent process did not exit cleanly"


def test_slow_then_fast_unordered_mp_process_round_trip() -> None:
    """Ensure data passes through full async stream and multiprocessing roundtrip."""
    processes, flows = create_mp_process([flow_slow_tripple, flow_double], num=1)
    data = [1, 2]
    try:
        source = IcoSource(lambda _: iter(data))
        flow = source | IcoAsyncStream(flows, ordered=False)
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

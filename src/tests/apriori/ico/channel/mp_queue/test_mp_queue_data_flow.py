from __future__ import annotations

import time
from collections.abc import Callable
from multiprocessing import get_context
from multiprocessing.context import SpawnProcess

import pytest

from apriori.ico.core.operator import I, IcoOperator, O
from apriori.ico.core.source import IcoSource
from apriori.ico.runtime.agent.mp.mp_channel import MPChannel
from tests.apriori.ico.channel.mp_queue.utils import MPProcessMock

# ───────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────


def agent(
    channel: MPChannel[O, I],
    flow_factory: Callable[[], IcoOperator[I, O]],
    n: int = 1,
) -> None:
    """Simulated remote process executing receive → flow → send loop."""
    flow = flow_factory()
    count = 0

    while count < n:
        item = channel.wait_for_item()
        assert item is not None
        result = flow(item)
        channel.send(result)
        count += 1


def flow_identity() -> IcoOperator[int, int]:
    """Simple identity operator."""
    return IcoOperator[int, int](lambda x: x)


def flow_double() -> IcoOperator[int, int]:
    """Doubles numeric input."""
    return IcoOperator[int, int](lambda x: x * 2)


def start_mp_process_agent(
    channel: MPChannel[O, I],
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


def test_send_receive_roundtrip_flow_basic() -> None:
    """Ensure data passes through full MPChannel roundtrip."""
    channel = MPChannel[int, int](get_context("spawn"))
    mp_process_mock = MPProcessMock[int, int](channel)
    agent_process = start_mp_process_agent(channel.invert(), flow_identity)
    try:
        result = mp_process_mock(42)
        assert result == 42
    finally:
        agent_process.terminate()
        agent_process.join(timeout=0.5)
    assert not agent_process.is_alive(), "Agent process did not exit cleanly"


# ───────────────────────────────────────────────
#  Test: Roundtrip with transformation
# ───────────────────────────────────────────────


def test_send_receive_roundtrip_transform() -> None:
    """Verify that data transformation inside remote flow works."""
    channel = MPChannel[int, int](get_context("spawn"))
    mp_process_mock = MPProcessMock[int, int](channel)
    agent_process = start_mp_process_agent(channel.invert(), flow_double)

    try:
        result = mp_process_mock(21)
        assert result == 42
    finally:
        agent_process.terminate()
        agent_process.join(timeout=0.5)
    assert not agent_process.is_alive()


# ───────────────────────────────────────────────
#  Test: Multiple sequential sends
# ───────────────────────────────────────────────


def test_send_receive_multiple_items() -> None:
    """Ensure multiple sequential messages are handled correctly."""
    channel = MPChannel[int, int](get_context("spawn"))
    num_queries = 5
    mp_process_mock = MPProcessMock[int, int](channel)
    agent_process = start_mp_process_agent(channel.invert(), flow_double, n=num_queries)

    try:
        results = [mp_process_mock(i) for i in range(num_queries)]
        assert results == [i * 2 for i in range(num_queries)]
    finally:
        agent_process.terminate()
        agent_process.join(timeout=0.5)


def test_send_receive_stream() -> None:
    """Ensure multiple sequential messages are handled correctly."""
    channel = MPChannel[int, int](get_context("spawn"))
    num_queries = 5
    mp_process_mock = MPProcessMock[int, int](channel)
    agent_process = start_mp_process_agent(channel.invert(), flow_double, n=num_queries)

    try:
        src = IcoSource(lambda: iter(range(num_queries)))
        flow = src | mp_process_mock.stream()
        results = list(flow(None))
        assert results == [i * 2 for i in range(num_queries)]
    finally:
        agent_process.terminate()
        agent_process.join(timeout=0.5)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))

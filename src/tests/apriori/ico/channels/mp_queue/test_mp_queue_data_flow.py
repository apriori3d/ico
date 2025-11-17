from __future__ import annotations

import time
from collections.abc import Callable
from multiprocessing import get_context
from multiprocessing.context import SpawnProcess

import pytest

from apriori.ico.channels.mp_queue.channel import MPQueueChannel
from apriori.ico.core.dsl.operator import IcoOperator
from apriori.ico.core.runtime.channels.types import IcoRuntimeChannelProtocol
from apriori.ico.core.types import I, IcoOperatorProtocol, O

# ───────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────


def agent(
    channel: IcoRuntimeChannelProtocol[I, O],
    flow_factory: Callable[[], IcoOperatorProtocol[O, I]],
    n: int = 1,
) -> None:
    """Simulated remote process executing receive → flow → send loop."""
    flow = flow_factory()
    closure = channel.receive | flow | channel.send
    count = 0
    while count < n:
        closure(None)
        count += 1


def flow_identity() -> IcoOperatorProtocol[int, int]:
    """Simple identity operator."""
    return IcoOperator[int, int](lambda x: x)


def flow_double() -> IcoOperatorProtocol[int, int]:
    """Doubles numeric input."""
    return IcoOperator[int, int](lambda x: x * 2)


def start_mp_process_agent(
    channel: MPQueueChannel[I, O],
    flow_factory: Callable[[], IcoOperatorProtocol[I, O]],
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
    """Ensure data passes through full MPQueueChannel roundtrip."""
    channel = MPQueueChannel[int, int](get_context("spawn"))
    process = start_mp_process_agent(channel.make_pair(), flow_identity)

    try:
        flow = channel.send | channel.receive
        result = flow(42)
        assert result == 42
    finally:
        process.terminate()
        process.join(timeout=0.5)
    assert not process.is_alive(), "Agent process did not exit cleanly"


# ───────────────────────────────────────────────
#  Test: Roundtrip with transformation
# ───────────────────────────────────────────────


def test_send_receive_roundtrip_transform() -> None:
    """Verify that data transformation inside remote flow works."""
    channel = MPQueueChannel[int, int](get_context("spawn"))
    process = start_mp_process_agent(channel.make_pair(), flow_double)

    try:
        flow = channel.send | channel.receive
        result = flow(21)
        assert result == 42
    finally:
        process.terminate()
        process.join(timeout=0.5)
    assert not process.is_alive()


# ───────────────────────────────────────────────
#  Test: Multiple sequential sends
# ───────────────────────────────────────────────


def test_send_receive_multiple_items() -> None:
    """Ensure multiple sequential messages are handled correctly."""
    channel = MPQueueChannel[int, int](get_context("spawn"))
    num_queries = 5
    process = start_mp_process_agent(channel.make_pair(), flow_double, n=num_queries)

    try:
        flow = channel.send | channel.receive
        results = [flow(i) for i in range(num_queries)]
        assert results == [i * 2 for i in range(num_queries)]
    finally:
        process.terminate()
        process.join(timeout=0.5)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))

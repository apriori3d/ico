from __future__ import annotations

import time

import pytest

from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.exceptions import IcoRuntimeError
from apriori.ico.core.runtime.operator import IcoRuntimeOperator
from apriori.ico.core.runtime.types import (
    IcoRuntimeCommandType,
    IcoRuntimeEvent,
    IcoRuntimeEventType,
)
from apriori.ico.runtime.agents.mp_process.mp_process import (
    MPProcess,
    mp_process,
)

# ───────────────────────────────────────────────
# Helper flows
# ───────────────────────────────────────────────

FloatOperator = IcoOperator[float, float]


def op_identity() -> FloatOperator:
    """Simple identity operator."""
    return FloatOperator(lambda x: x)


def op_double() -> FloatOperator:
    """Double transformation."""
    return FloatOperator(lambda x: x * 2)


def op_fail_factory() -> FloatOperator:
    """Fail when receiving the value three."""
    return FloatOperator(op_fail_on_three)


def op_fail_on_three(x: float) -> float:
    """Fail when receiving the value three."""
    if x == 3:
        raise IcoRuntimeError("boom")
    return x


# ───────────────────────────────────────────────
# Helper runtime
# ───────────────────────────────────────────────


class RecordingRuntime(IcoRuntimeOperator):
    """Runtime collecting received commands and events for verification."""

    commands: list[IcoRuntimeCommandType]
    events: list[IcoRuntimeEventType]

    def __init__(self):
        super().__init__()
        self.commands = []
        self.events = []

    def on_command(self, command: IcoRuntimeCommandType) -> None:
        self.commands.append(command)

    def on_event(self, event: IcoRuntimeEvent) -> None:
        self.events.append(event.type)


# ───────────────────────────────────────────────
# Test: Agent processes items via flow
# ───────────────────────────────────────────────


def test_agent_basic_roundtrip() -> None:
    """Agent spawned by MPProcessAgentHost should process data using remote flow."""

    host = MPProcess[float, float].create(op_double)
    host.activate()

    flow = host._channel.output | host._channel.receive
    result = flow(21)
    assert result == 42
    host.deactivate()  # must stop remote agent


# ───────────────────────────────────────────────
# Test: Multiple sequential items
# ───────────────────────────────────────────────


def test_agent_multiple_items():
    """Agent should correctly process multiple items in sequence."""

    host = MPProcess[float, float].create(op_double)
    host.activate()

    try:
        flow = host._channel.output | host._channel.receive
        results = [flow(i) for i in range(5)]
        assert results == [i * 2 for i in range(5)]
    finally:
        host._channel.deactivate()


# ───────────────────────────────────────────────
# Test: Exception propagation from agent
# ───────────────────────────────────────────────


def test_agent_exception_propagation():
    """If agent raises IcoRuntimeError, host should receive corresponding runtime event."""

    host = MPProcess[float, float].create(op_fail_factory)
    host.activate()
    flow = host._channel.output | host._channel.receive

    # normal items ok
    assert flow(1) == 1
    assert flow(2) == 2

    with pytest.raises(IcoRuntimeError) as exc:
        flow(3)
    assert "boom" in str(exc.value)

    host.deactivate()


# ───────────────────────────────────────────────
# Test: Commands propagate to agent runtime
# ───────────────────────────────────────────────


def test_agent_command_propagation():
    """activate/pause/resume/deactivate should propagate to agent runtime."""

    # Attach recording runtime to channel
    end_runtime = RecordingRuntime()

    host = MPProcess[float, float].create(op_identity)
    host.connect_runtime(end_runtime)

    # channel is a parent runtime of the agent host runtime
    host._channel.activate().pause().resume().deactivate()

    # Agent receives commands in same order
    assert end_runtime.commands == [
        IcoRuntimeCommandType.activate,
        IcoRuntimeCommandType.pause,
        IcoRuntimeCommandType.resume,
        IcoRuntimeCommandType.deactivate,
    ]


# ───────────────────────────────────────────────
# Test: Events bubble back to host
# ───────────────────────────────────────────────


def test_agent_event_propagation():
    """Agent should bubble events back to host runtime."""

    first_runtime = RecordingRuntime()
    host = MPProcess[float, float].create(op_identity)
    first_runtime.connect_runtime(host._channel)

    first_runtime.activate()
    # flow = host.channel.send | host.channel.receive

    # Agent echoes heartbeat event manually
    host._channel.bubble_event(IcoRuntimeEvent.heartbeat())

    # produce/consume one item to flush internal queues
    # flow(123)

    assert IcoRuntimeEvent.heartbeat().type in first_runtime.events

    host.deactivate()


# ───────────────────────────────────────────────
# Test: mp_process api end-to-end
# ───────────────────────────────────────────────


def test_agent_mp_process_end_to_end():
    """Portal wiring must allow transparent remote computation."""

    collected: list[str] = []

    def collect_output(x: float) -> None:
        collected.append(f"Received {x}")

    src = IcoOperator[None, float](lambda _: 7)
    dst = IcoOperator[float, None](collect_output)

    flow = src | mp_process(op_double) | dst
    runtime = IcoRuntimeOperator(flow)
    runtime.activate().run().deactivate()

    assert collected == ["Received 14"]


# ───────────────────────────────────────────────
# Test: Agent shuts down cleanly on deactivate
# ───────────────────────────────────────────────


def test_agent_clean_shutdown():
    """Agent spawned by host should terminate on deactivate."""

    host = MPProcess[float, float].create(op_identity)
    host.activate()

    assert host._agent_process is not None
    assert host._agent_process.is_alive()

    host.deactivate()

    # give process time to shut down
    time.sleep(0.1)

    assert not host._agent_process.is_alive(), "Agent process did not exit cleanly"


if __name__ == "__main__":
    test_agent_basic_roundtrip()

    # import sys

    # sys.exit(pytest.main([__file__]))

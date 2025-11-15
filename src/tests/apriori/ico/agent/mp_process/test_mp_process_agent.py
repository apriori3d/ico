# tests/test_mp_process_agent.py
from __future__ import annotations

import time

import pytest

from apriori.ico.agents.mp_process.agent_host import (
    MPProcessAgentHost,
    create_portal,
)
from apriori.ico.core.dsl.operator import IcoOperator
from apriori.ico.core.runtime.events import IcoRuntimeEvent
from apriori.ico.core.runtime.exceptions import IcoRuntimeError
from apriori.ico.core.runtime.runtime_operator import IcoRuntimeOperator
from apriori.ico.core.runtime.types import IcoRuntimeCommandType

# ───────────────────────────────────────────────
# Helper flows
# ───────────────────────────────────────────────


def flow_identity():
    """Simple identity operator."""
    return IcoOperator(lambda x: x)


def flow_double():
    """Double transformation."""
    return IcoOperator(lambda x: x * 2)


def flow_fail_on(value_to_fail: int):
    """Fail when receiving the specified value."""
    return IcoOperator(
        lambda x: x
        if x != value_to_fail
        else (_ for _ in ()).throw(IcoRuntimeError("boom"))
    )


# ───────────────────────────────────────────────
# Helper runtime
# ───────────────────────────────────────────────


class RecordingRuntime(IcoRuntimeOperator):
    """Runtime collecting received commands and events for verification."""

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


def test_agent_basic_roundtrip():
    """Agent spawned by MPProcessAgentHost should process data using remote flow."""

    host = MPProcessAgentHost.create(flow_factory=flow_double)
    host.activate()  # spawn agent process

    flow = host.channel.send | host.channel.receive
    result = flow(21)
    assert result == 42
    host.deactivate()  # must stop remote agent


# ───────────────────────────────────────────────
# Test: Multiple sequential items
# ───────────────────────────────────────────────


def test_agent_multiple_items():
    """Agent should correctly process multiple items in sequence."""

    host = MPProcessAgentHost.create(flow_factory=flow_double)
    host.activate()

    try:
        flow = host.channel.send | host.channel.receive
        results = [flow(i) for i in range(5)]
        assert results == [i * 2 for i in range(5)]
    finally:
        host.deactivate()


# ───────────────────────────────────────────────
# Test: Exception propagation from agent
# ───────────────────────────────────────────────


def test_agent_exception_propagation():
    """If agent raises IcoRuntimeError, host should receive corresponding runtime event."""

    host = MPProcessAgentHost.create(lambda: flow_fail_on(3))
    host.activate()
    flow = host.channel.send | host.channel.receive

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
    host_runtime = RecordingRuntime()

    host = MPProcessAgentHost.create(flow_factory=flow_identity)
    host.channel.connect_runtime(host_runtime)

    host.activate()
    host.pause()
    host.resume()
    host.deactivate()

    # Agent receives commands in same order
    assert host_runtime.commands[:3] == [
        IcoRuntimeCommandType.activate,
        IcoRuntimeCommandType.pause,
        IcoRuntimeCommandType.resume,
    ]


# ───────────────────────────────────────────────
# Test: Events bubble back to host
# ───────────────────────────────────────────────


def test_agent_event_propagation():
    """Agent should bubble events back to host runtime."""

    host_runtime = RecordingRuntime()
    host = MPProcessAgentHost.create(flow_factory=flow_identity)
    host_runtime.connect_runtime(host)

    host.activate()
    # flow = host.channel.send | host.channel.receive

    # Agent echoes heartbeat event manually
    host.channel.bubble_event(IcoRuntimeEvent.heartbeat())

    # produce/consume one item to flush internal queues
    # flow(123)

    assert IcoRuntimeEvent.heartbeat().type in host_runtime.events

    host.deactivate()


# ───────────────────────────────────────────────
# Test: create_portal end-to-end
# ───────────────────────────────────────────────


def test_agent_portal_end_to_end():
    """Portal wiring must allow transparent remote computation."""

    class Source(IcoOperator[None, int]):
        def __init__(self):
            super().__init__(lambda _: 7)

    class Sink(IcoOperator[int, str]):
        def __init__(self):
            super().__init__(lambda x: f"R{x}")

    src = Source()
    dst = Sink()

    op = create_portal(src, dst, flow_factory=flow_double)
    result = op(None)

    assert result == "R14"


# ───────────────────────────────────────────────
# Test: Agent shuts down cleanly on deactivate
# ───────────────────────────────────────────────


def test_agent_clean_shutdown():
    """Agent spawned by host should terminate on deactivate."""

    host = MPProcessAgentHost.create(flow_factory=flow_identity)
    host.activate()

    assert host._agent_process is not None
    assert host._agent_process.is_alive()

    host.deactivate()

    # give process time to shut down
    time.sleep(0.1)

    assert not host._agent_process.is_alive(), "Agent process did not exit cleanly"


if __name__ == "__main__":
    # test_agent_basic_roundtrip()

    import sys

    sys.exit(pytest.main([__file__]))

from __future__ import annotations

import time

import pytest

from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.runtime.command import (
    IcoActivateCommand,
    IcoDeactivateCommand,
    IcoPauseCommand,
    IcoResumeCommand,
)
from apriori.ico.core.runtime.contour import IcoRuntimeContour
from apriori.ico.core.runtime.event import (
    IcoHearbeatEvent,
)
from apriori.ico.core.runtime.exceptions import IcoRuntimeError
from apriori.ico.runtime.agent.mp_process.mp_process import (
    MPProcess,
)
from tests.apriori.ico.core.runtime.runtime_node.test_utils import RecordingRuntimeNode

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
# Test: Agent processes items via flow
# ───────────────────────────────────────────────


def test_agent_basic_roundtrip() -> None:
    """Agent spawned by MPProcessAgentHost should process data using remote flow."""

    mp_process = MPProcess(op_double)
    mp_process.activate()

    result = mp_process(21)
    assert result == 42
    mp_process.deactivate()  # must stop remote agent


# ───────────────────────────────────────────────
# Test: Multiple sequential items
# ───────────────────────────────────────────────


def test_agent_multiple_items() -> None:
    """Agent should correctly process multiple items in sequence."""

    mp_process = MPProcess(op_double)
    mp_process.activate()

    try:
        results = [mp_process(i) for i in range(5)]
        assert results == [i * 2 for i in range(5)]
    finally:
        mp_process.deactivate()


# ───────────────────────────────────────────────
# Test: Exception propagation from agent
# ───────────────────────────────────────────────


def test_agent_exception_propagation() -> None:
    """If agent raises IcoRuntimeError, host should receive corresponding runtime event."""

    mp_process = MPProcess(op_fail_factory)
    mp_process.activate()

    # normal items ok
    assert mp_process(1) == 1
    assert mp_process(2) == 2

    with pytest.raises(IcoRuntimeError) as exc:
        mp_process(3)
    assert "boom" in str(exc.value)

    mp_process.deactivate()


# ───────────────────────────────────────────────
# Test: Commands propagate to agent runtime
# ───────────────────────────────────────────────


def test_agent_command_propagation() -> None:
    """activate/pause/resume/deactivate should propagate to agent runtime."""

    # Attach recording runtime to channel
    end_runtime = RecordingRuntimeNode()

    mp_process = MPProcess(op_identity)
    mp_process.connect_runtime(end_runtime)

    # channel is a parent runtime of the agent host runtime
    mp_process.activate().pause().resume().deactivate()

    # Agent receives commands in same order
    assert end_runtime.recorded_commands == [
        IcoActivateCommand,
        IcoPauseCommand,
        IcoResumeCommand,
        IcoDeactivateCommand,
    ]


# ───────────────────────────────────────────────
# Test: Events bubble back to host
# ───────────────────────────────────────────────


def test_agent_event_propagation() -> None:
    """Agent should bubble events back to host runtime."""

    main_runtime = RecordingRuntimeNode()
    mp_process = MPProcess(op_identity)
    main_runtime.connect_runtime(mp_process)

    main_runtime.activate()
    # flow = host.channel.send | host.channel.receive

    # Agent echoes heartbeat event manually
    mp_process.bubble_event(IcoHearbeatEvent())

    # produce/consume one item to flush internal queues
    # flow(123)

    assert [IcoHearbeatEvent] == main_runtime.recorded_events

    main_runtime.deactivate()


# ───────────────────────────────────────────────
# Test: mp_process api end-to-end
# ───────────────────────────────────────────────


def test_agent_mp_process_end_to_end() -> None:
    """Portal wiring must allow transparent remote computation."""

    collected: list[str] = []

    def collect_output(x: float) -> None:
        collected.append(f"Received {x}")

    src = IcoOperator[None, float](lambda _: 7)
    dst = IcoOperator[float, None](collect_output)

    flow = src | MPProcess(op_double) | dst

    runtime = IcoRuntimeContour(flow)
    runtime.activate().run().deactivate()

    assert collected == ["Received 14"]


# ───────────────────────────────────────────────
# Test: Agent shuts down cleanly on deactivate
# ───────────────────────────────────────────────


def test_agent_clean_shutdown() -> None:
    """Agent spawned by host should terminate on deactivate."""

    mp_process = MPProcess(op_identity)
    mp_process.activate()

    assert mp_process.is_alive

    mp_process.deactivate()

    # give process time to shut down
    time.sleep(0.1)

    assert not mp_process.is_alive, "Agent process did not exit cleanly"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))

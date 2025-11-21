# ───────────────────────────────────────────────
#  Test: Runtime command propagation (roundtrip)
# ───────────────────────────────────────────────
import time
from multiprocessing import get_context
from multiprocessing.context import SpawnProcess
from typing import Any

import pytest

from apriori.ico.channels.mp_queue.channel import MPQueueChannel
from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.exceptions import IcoRuntimeError
from apriori.ico.core.runtime.operator import IcoRuntimeOperator
from apriori.ico.core.runtime.types import (
    IcoRuntimeCommandType,
    IcoRuntimeEvent,
)

# ───────────────────────────────────────────────
#  Helpers
# ───────────────────────────────────────────────


class ControlFlowTestingRuntime(IcoRuntimeOperator):
    """Runtime that records received commands and events for testing."""

    commands_received: list[IcoRuntimeCommandType]
    events_received: list[IcoRuntimeEvent]

    def __init__(self):
        super().__init__()
        self.commands_received = []
        self.events_received = []

    def on_command(self, command: IcoRuntimeCommandType) -> None:
        # Echo command back through send endpoint
        self.commands_received.append(command)

    def on_event(self, event: IcoRuntimeEvent) -> None:
        self.events_received.append(event)


def recording_agent(
    channel: MPQueueChannel[dict[str, Any], str],
    runs_num: int = 1,
) -> None:
    """Agent process that records received commands and sends back acknowledgements."""

    # Create runtime that records commands and events and sends them back
    runtime = ControlFlowTestingRuntime()
    # Connect runtime to channel for command/event propagation
    channel.connect_runtime(runtime)

    def reporting_fn(item: str) -> dict[str, Any]:
        # Send heartbeat event to host runtime
        runtime.bubble_event(IcoRuntimeEvent.heartbeat())
        if item == "report":
            # Return recorded commands and events
            return {
                "commands": runtime.commands_received,
                "events": runtime.events_received,
            }
        elif item == "error":
            raise IcoRuntimeError("Simulated agent error")
        else:
            # Raise error for unknown items to test exception propagation
            raise ValueError(f"Unknown item: {item}")

    # Create agent flow to record and report commands/events
    reporting_operator = IcoOperator[str, dict[str, Any]](reporting_fn)
    closure = channel.input | reporting_operator | channel.output

    # Execute the agent closure
    run_num = 1
    while run_num <= runs_num:
        try:
            closure(None)
            run_num += 1
        except IcoRuntimeError as e:
            # Send exception event back to host runtime
            channel.output.on_event(IcoRuntimeEvent.exception(e))


# ───────────────────────────────────────────────
#  Test: Runtime command and event propagation
# ───────────────────────────────────────────────


def test_runtime_flow_propagation() -> None:
    """Ensure
    1. Runtime commands travel to agent
    2. Events bubble back to host runtime
    3. Exceptions in agent propagate back as runtime events
    4. Clean deactivation of runtimes
    """

    # Create communication channel between host and agent
    ctx = get_context("spawn")
    channel = MPQueueChannel[str, dict[str, Any]](ctx)

    # Create host runtime to aggregate bubble-up events via channel
    host_runtime = ControlFlowTestingRuntime()
    host_runtime.connect_runtime(channel)  # Setup link for commands/events flow

    # Strat agent process
    process: SpawnProcess = ctx.Process(
        target=recording_agent,
        args=(
            channel.make_pair(),
            3,  # first run for 'report', second for 'error', third to test clean exit
        ),
        daemon=True,
    )
    process.start()
    time.sleep(0.05)

    try:
        # Send commands to remote agent
        host_runtime.activate().pause().resume()

        # Send 'report' to agent to get back recorded commands and events
        flow = channel.output | channel.input

        # ──── Check commands and events propagation ────

        agent_runtime_recording = flow("report")

        # Check that commands were received correctly
        assert agent_runtime_recording["commands"] == host_runtime.commands_received
        # Check that events were received correctly
        assert [event.type for event in agent_runtime_recording["events"]] == [
            event.type for event in host_runtime.events_received
        ]

        # ──── Check Exception event propagation ────

        with pytest.raises(IcoRuntimeError) as error:
            flow("error")
        assert "Simulated agent error" in str(error.value)

        # ──── Check for correct deactivation ────
        host_runtime.deactivate()

    finally:
        process.terminate()
        process.join(timeout=0.5)

    assert not process.is_alive(), "Agent process did not exit cleanly"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))

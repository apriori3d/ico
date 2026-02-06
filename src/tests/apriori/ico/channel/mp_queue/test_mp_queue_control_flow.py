# ───────────────────────────────────────────────
#  Test: Runtime command propagation (roundtrip)
# ───────────────────────────────────────────────
import time
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.context import SpawnProcess

import pytest

from apriori.ico.core.runtime.channel.channel import IcoChannel
from apriori.ico.core.runtime.command import IcoRuntimeCommand
from apriori.ico.core.runtime.event import (
    IcoFaultEvent,
    IcoHeartbeatEvent,
    IcoRuntimeEvent,
)
from apriori.ico.core.runtime.exceptions import IcoRuntimeError
from apriori.ico.runtime.agent.mp.mp_channel import MPChannel
from tests.apriori.ico.channel.mp_queue.utils import MPProcessMock
from tests.apriori.ico.core.runtime.runtime_node.test_utils import RecordingRuntimeNode


@dataclass(slots=True, frozen=True)
class AgentReport:
    recorded_commands: list[type[IcoRuntimeCommand]]
    recorded_events: list[type[IcoRuntimeEvent]]


def recording_agent(
    channel: IcoChannel[AgentReport, str],
    runs_num: int = 1,
) -> None:
    """Agent process that records received commands and sends back acknowledgements."""

    # Create agent flow to record and report commands/events
    recording_node = RecordingRuntimeNode(name="agent_runtime")
    channel.runtime_port = recording_node

    run_num = 0
    while run_num < runs_num:
        try:
            item = channel.wait_for_item()

            # Shutdown signal
            if item is None:
                channel.close()
                return

            hearbeat = IcoHeartbeatEvent()
            recording_node.on_event(hearbeat)
            channel.send_event(hearbeat)

            if item == "report":
                # Return recorded commands and events
                report: AgentReport = AgentReport(
                    recorded_commands=recording_node.recorded_commands,
                    recorded_events=recording_node.recorded_events,
                )
                channel.send(report)
            elif item == "error":
                channel.send_event(
                    IcoFaultEvent.create(IcoRuntimeError("Simulated agent error"))
                )
            else:
                channel.send_event(
                    IcoFaultEvent.create(IcoRuntimeError("Unknown command"))
                )

            run_num += 1
        except Exception as e:
            channel.send_event(IcoFaultEvent.create(e))


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

    # Create host runtime to aggregate bubble-up events via channel
    host_runtime = RecordingRuntimeNode(name="host_runtime")

    # Create communication channel between host and agent
    ctx = get_context("spawn")
    channel = MPChannel[str, AgentReport](
        mp_context=ctx,
        runtime_port=host_runtime,
        accept_commands=False,
        accept_events=True,
        strict_accept=True,
    )
    # Strat agent process
    process: SpawnProcess = ctx.Process(
        target=recording_agent,
        args=(
            channel.invert(),
            3,  # first run for 'report', second for 'error', third to test clean exit
        ),
        daemon=True,
    )
    process.start()
    time.sleep(0.05)
    mp_process_mock = MPProcessMock(channel)
    host_runtime.add_runtime_children(mp_process_mock)

    try:
        # Send commands to remote agent
        host_runtime.activate().pause().resume()

        # ──── Check commands and events propagation ────

        agent_report = mp_process_mock("report")

        # Check that commands were received correctly
        assert agent_report.recorded_commands == host_runtime.recorded_commands
        # Check that events were received correctly
        assert agent_report.recorded_events == host_runtime.recorded_events

        # ──── Check Exception event propagation ────

        with pytest.raises(IcoRuntimeError) as error:
            mp_process_mock("error")
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

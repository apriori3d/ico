# ───────────────────────────────────────────────
#  Test: Runtime command propagation (roundtrip)
# ───────────────────────────────────────────────
import time
from multiprocessing import get_context
from multiprocessing.context import SpawnProcess

import pytest

from apriori.ico.core.runtime.channel.utils import wait_for_item
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.exceptions import IcoRuntimeError
from apriori.ico.runtime.channel.mp_queue.channel import MPQueueChannel
from tests.apriori.ico.channel.mp_queue.utils import MPProcessMock
from tests.apriori.ico.core.runtime.runtime_node.test_utils import RecordingRuntimeNode


def recording_agent(
    channel: MPQueueChannel[dict[str, object], str],
    runs_num: int = 1,
) -> None:
    """Agent process that records received commands and sends back acknowledgements."""

    # Create agent flow to record and report commands/events
    recording_node = RecordingRuntimeNode(name="agent_runtime")

    run_num = 0
    while run_num < runs_num:
        try:
            item = wait_for_item(
                runtime_node=recording_node,
                endpoint=channel.input,
                accept_commands=True,
                accept_events=False,
            )
            assert item is not None
            hearbeat = IcoRuntimeEvent.heartbeat()
            recording_node.on_event(hearbeat)
            channel.output.send(hearbeat)

            if item == "report":
                # Return recorded commands and events
                report: dict[str, object] = {
                    "commands": recording_node.recorded_commands,
                    "events": recording_node.recorded_events,
                }
                channel.output.send(report)
            elif item == "error":
                channel.output.send(
                    IcoRuntimeEvent.exception(IcoRuntimeError("Simulated agent error"))
                )
            else:
                channel.output.send(
                    IcoRuntimeEvent.exception(IcoRuntimeError("Unknown command"))
                )

            run_num += 1
        except Exception as e:
            channel.output.send(IcoRuntimeEvent.exception(e))


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
    channel = MPQueueChannel[str, dict[str, object]](ctx)

    # Create host runtime to aggregate bubble-up events via channel
    host_runtime = RecordingRuntimeNode(name="host_runtime")

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
    mp_process_mock = MPProcessMock(channel)
    host_runtime.connect_runtime(mp_process_mock)

    try:
        # Send commands to remote agent
        host_runtime.activate().pause().resume()

        # ──── Check commands and events propagation ────

        agent_runtime_recording = mp_process_mock("report")

        # Check that commands were received correctly
        assert agent_runtime_recording["commands"] == host_runtime.recorded_commands
        # Check that events were received correctly
        assert agent_runtime_recording["events"] == host_runtime.recorded_events

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

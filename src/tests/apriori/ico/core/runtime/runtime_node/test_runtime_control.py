from __future__ import annotations

from apriori.ico.core.runtime.command import IcoRuntimeCommand, IcoRuntimeCommandType
from apriori.ico.core.runtime.event import IcoRuntimeEvent, IcoRuntimeEventType
from tests.apriori.ico.core.runtime.runtime_node.test_utils import RecordingRuntimeNode


def test_runtime_command_and_event_propagation() -> None:
    # Build runtime chain: root -> mid -> leaf
    leaf = RecordingRuntimeNode(name="leaf")
    mid = RecordingRuntimeNode(runtime_children=[leaf], name="mid")
    root = RecordingRuntimeNode(runtime_children=[mid], name="root")

    # --- Test: Broadcast command from root ---
    activate = IcoRuntimeCommand.activate()
    root.broadcast_command(activate)

    assert root.recorded_commands == [IcoRuntimeCommandType.activate]
    assert mid.recorded_commands == [IcoRuntimeCommandType.activate]
    assert leaf.recorded_commands == [IcoRuntimeCommandType.activate]

    # --- Test: Bubble event from leaf ---
    heartbeat = IcoRuntimeEvent.heartbeat()
    leaf.bubble_event(heartbeat)

    # leaf handles event, mid handles, root handles
    assert leaf.recorded_events == [IcoRuntimeEventType.heartbeat]
    assert mid.recorded_events == [IcoRuntimeEventType.heartbeat]
    assert root.recorded_events == [IcoRuntimeEventType.heartbeat]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))

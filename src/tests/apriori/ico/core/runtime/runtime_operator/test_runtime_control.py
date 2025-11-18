from __future__ import annotations

from apriori.ico.core.runtime.events import IcoRuntimeEvent
from apriori.ico.core.runtime.types import (
    IcoRuntimeCommandType,
)
from tests.apriori.ico.core.runtime.runtime_operator.test_utils import (
    ControlRecordingRuntime,
)


def test_runtime_command_and_event_propagation() -> None:
    # Build runtime chain: root -> mid -> leaf
    leaf = ControlRecordingRuntime(name="leaf")
    mid = ControlRecordingRuntime(runtime_children=[leaf], name="mid")
    root = ControlRecordingRuntime(runtime_children=[mid], name="root")

    # --- Test: Broadcast command from root ---
    root.broadcast_command(IcoRuntimeCommandType.activate)

    assert root.received_commands == [IcoRuntimeCommandType.activate]
    assert mid.received_commands == [IcoRuntimeCommandType.activate]
    assert leaf.received_commands == [IcoRuntimeCommandType.activate]

    # --- Test: Bubble event from leaf ---
    heartbeat = IcoRuntimeEvent.heartbeat()
    leaf.bubble_event(heartbeat)

    # leaf handles event, mid handles, root handles
    assert leaf.received_events == [heartbeat]
    assert mid.received_events == [heartbeat]
    assert root.received_events == [heartbeat]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))

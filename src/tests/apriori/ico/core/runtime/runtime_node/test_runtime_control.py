from __future__ import annotations

from ico.core.runtime.command import IcoActivateCommand
from ico.core.runtime.printer import IcoPrintEvent
from ico.core.runtime.state import IdleState, ReadyState
from tests.apriori.ico.core.runtime.runtime_node.test_utils import (
    RecordingRuntimeNode,
)


def test_runtime_command_and_event_propagation() -> None:
    # Build runtime chain: root -> mid -> leaf
    leaf = RecordingRuntimeNode(name="leaf")
    mid = RecordingRuntimeNode(runtime_children=[leaf], name="mid")
    root = RecordingRuntimeNode(runtime_children=[mid], name="root")

    # --- Test: Broadcast command from root ---
    root.activate()

    assert root.recorded_commands == [IcoActivateCommand]
    assert mid.recorded_commands == [IcoActivateCommand]
    assert leaf.recorded_commands == [IcoActivateCommand]

    assert root.recorded_states == [IdleState, ReadyState]
    assert mid.recorded_states == [IdleState, ReadyState]
    assert leaf.recorded_states == [IdleState, ReadyState]

    # --- Test: Bubble event from leaf ---
    leaf.bubble_event(IcoPrintEvent.create(message="ping"))

    # leaf handles event, mid handles, root handles
    assert leaf.recorded_events == [IcoPrintEvent]
    assert mid.recorded_events == [IcoPrintEvent]
    assert root.recorded_events == [IcoPrintEvent]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))

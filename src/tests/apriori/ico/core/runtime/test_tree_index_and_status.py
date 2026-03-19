"""
Tests for TreePathIndex logic and StatusRequest command.

"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from ico.core.identity import IcoIdentity
from ico.core.operator import IcoOperator
from ico.core.runtime.event import IcoRuntimeEvent
from ico.core.runtime.node import IcoRuntimeNode
from ico.core.runtime.runtime import IcoRuntime
from ico.core.runtime.runtime_wrapper import wrap_runtime_fn
from ico.core.runtime.state import (
    IcoRuntimeState,
    IcoStateRequestCommand,
    IdleState,
    ReadyState,
)
from ico.core.runtime.tool import IcoTool
from ico.core.tree_utils import TreePathIndex
from ico.runtime.agent.mp.mp_agent import MPAgent


class StateCollectorTool(IcoTool):
    """Tool for collecting node states with their paths in the tree."""

    def __init__(self) -> None:
        super().__init__()
        self.collected_states: OrderedDict[TreePathIndex, IcoRuntimeState] = (
            OrderedDict()
        )
        self.collected_events: list[tuple[TreePathIndex, IcoRuntimeState]] = []

    def on_forward_event(self, event: IcoRuntimeEvent) -> None:
        """Collect state events with their paths."""
        from ico.core.runtime.state import IcoStateEvent

        if isinstance(event, IcoStateEvent):
            tree_path = event.trace.reverse()
            self.collected_states[tree_path] = event.state
            self.collected_events.append((tree_path, event.state))

        super().on_forward_event(event)

    def reset(self) -> None:
        """Reset collected data."""
        self.collected_states.clear()
        self.collected_events.clear()


def test_flow_with_runtime() -> None:
    """
    Create test runtime tree with multiple levels and siblings:

    Runtime
    ├── IcoToolBox
    │   └── StateCollectorTool
    ├── IcoRuntimeWrapper
        └── IcoRuntimeNode
    """

    nested_runtime_node_1 = IcoRuntimeNode(runtime_name="nested_runtime_node_1")
    nested_runtime_node_2 = IcoRuntimeNode(runtime_name="nested_runtime_node_2")

    shared_runtime_node1 = IcoRuntimeNode(
        runtime_name="shared_runtime_node_1",
        runtime_children=[nested_runtime_node_1, nested_runtime_node_2],
    )

    flow = (
        IcoIdentity[Any]()
        | wrap_runtime_fn(IcoIdentity[Any](), shared_runtime_node1)
        | IcoIdentity[Any]()
        | wrap_runtime_fn(IcoIdentity[Any](), shared_runtime_node1)
    )

    runtime = IcoRuntime(flow)
    shared_runtime_node1.activate()
    nested_runtime_node_1.deactivate()

    collector = StateCollectorTool()
    runtime.toolbox.add_tool(collector)

    flow.describe()
    runtime.describe()

    runtime.broadcast_command(IcoStateRequestCommand.create())

    expected_states = {
        TreePathIndex(): IdleState(),
        TreePathIndex.create(0): IdleState(),
        TreePathIndex.create(0, 0): ReadyState(),
        TreePathIndex.create(0, 0, 0): IdleState(),
        TreePathIndex.create(0, 0, 1): ReadyState(),
        TreePathIndex.create(1): IdleState(),
        TreePathIndex.create(1, 0): IdleState(),
    }
    assert collector.collected_states == expected_states


def _agent_flow_factory() -> IcoOperator[Any, Any]:
    return wrap_runtime_fn(
        IcoIdentity[Any](),
        IcoRuntimeNode(runtime_name="worker_runtime_node"),
    )


def test_flow_with_mp_agent() -> None:
    """
    Create test runtime tree with multiple levels and siblings with mp agent.

    """

    shared_runtime_node = IcoRuntimeNode(runtime_name="shared_runtime_node")
    flow = (
        IcoIdentity[Any]()
        | wrap_runtime_fn(IcoIdentity[Any](), shared_runtime_node)
        | MPAgent(_agent_flow_factory)
    )
    runtime = IcoRuntime(flow)
    runtime.activate()
    shared_runtime_node.deactivate()

    collector = StateCollectorTool()
    runtime.toolbox.add_tool(collector)

    flow.describe()
    runtime.describe()

    runtime.broadcast_command(IcoStateRequestCommand.create())

    expected_states = {
        TreePathIndex(): ReadyState(),
        TreePathIndex.create(0): ReadyState(),
        TreePathIndex.create(0, 0): IdleState(),
        TreePathIndex.create(1): ReadyState(),
        TreePathIndex.create(1, 0): ReadyState(),
        TreePathIndex.create(1, 0, 0): ReadyState(),
        TreePathIndex.create(1, 0, 0, 0): ReadyState(),
        TreePathIndex.create(2): ReadyState(),
        TreePathIndex.create(2, 0): IdleState(),
    }
    assert collector.collected_states == expected_states

    runtime.deactivate()


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))

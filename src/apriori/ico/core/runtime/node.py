from __future__ import annotations

from abc import ABC
from collections.abc import Iterator, Sequence
from enum import Enum, auto

from typing_extensions import Self

from apriori.ico.core.runtime.command import IcoRuntimeCommand, IcoRuntimeCommandType
from apriori.ico.core.runtime.event import IcoRuntimeEvent

# ────────────────────────────────────────────────
# State of runtime node
# ────────────────────────────────────────────────


class IcoRuntimeState(Enum):
    """
    Current runtime state of an agent and connected contour.

    States:
        • inactive - Operator uninitialized or fully released
        • ready    - Initialized and ready to run
        • running  - Currently executing or active
        • paused   - Temporarily suspended, resources preserved
        • error    - Faulted state after unrecoverable failure
    """

    inactive = auto()
    ready = auto()
    waiting = auto()
    running = auto()
    sending = auto()
    paused = auto()
    fault = auto()


COMMAND_TO_STATE = {
    IcoRuntimeCommandType.activate: IcoRuntimeState.ready,
    IcoRuntimeCommandType.reset: IcoRuntimeState.ready,
    IcoRuntimeCommandType.deactivate: IcoRuntimeState.inactive,
    IcoRuntimeCommandType.pause: IcoRuntimeState.paused,
    IcoRuntimeCommandType.resume: IcoRuntimeState.ready,
}

# ────────────────────────────────────────────────
# Runtime node
# ────────────────────────────────────────────────


class IcoRuntimeNode(ABC):
    """Structural attributes for graph representation of ICO operators."""

    runtime_name: str
    _runtime_children: list[IcoRuntimeNode]
    _runtime_parent: IcoRuntimeNode | None
    _last_command: IcoRuntimeCommand | None
    _last_event: IcoRuntimeEvent | None
    _state: IcoRuntimeState

    def __init__(
        self,
        runtime_name: str | None = None,
        runtime_parent: IcoRuntimeNode | None = None,
        runtime_children: Sequence[IcoRuntimeNode] | None = None,
    ) -> None:
        self.runtime_name = runtime_name or self.__class__.__name__
        self._state = IcoRuntimeState.inactive
        self._runtime_parent = runtime_parent
        self._runtime_children = (
            list(runtime_children) if runtime_children is not None else []
        )

        for child in self._runtime_children:
            child._runtime_parent = self

    # ────────────────────────────────────────────────
    # Properties
    # ────────────────────────────────────────────────

    @property
    def runtime_parent(self) -> IcoRuntimeNode | None:
        """Parent node in the runtime tree."""
        return self._runtime_parent

    @property
    def runtime_children(self) -> Sequence[IcoRuntimeNode]:
        """Children nodes in the runtime tree."""
        return self._runtime_children

    @property
    def state(self) -> IcoRuntimeState:
        """Current runtime state of the operator."""
        return self._state

    @state.setter
    def state(self, state: IcoRuntimeState) -> None:
        self._state = state

    @property
    def last_command(self) -> IcoRuntimeCommand | None:
        """Last received runtime command."""
        return self._last_command

    @property
    def last_event(self) -> IcoRuntimeEvent | None:
        """Last received runtime event."""
        return self._last_event

    # ────────────────────────────────────────────────
    # Commands
    # ────────────────────────────────────────────────

    def on_command(self, command: IcoRuntimeCommand) -> IcoRuntimeCommand | None:
        """
        Handle a single runtime command.

        Subclasses may override to implement additional behavior
        (e.g., resource allocation, reset hooks, or teardown logic).
        """
        self.state = COMMAND_TO_STATE.get(command.type, self._state)
        self._last_command = command
        return command

    def broadcast_command(
        self,
        command: IcoRuntimeCommand,
    ) -> None:
        """
        Recursively propagate a runtime command through the operator tree.

        Each node implementing `SupportsIcoRuntime` receives `on_command(command)`.
        """
        next_command = self.on_command(command)

        if next_command is None:
            return

        for child in self._runtime_children:
            child.broadcast_command(next_command)

    # ────────────────────────────────────────────────
    # Events
    # ────────────────────────────────────────────────

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        """
        Handle a single runtime event.

        Subclasses may override to implement additional behavior
        (e.g., logging, metrics, or alerting).
        """
        self._last_event = event
        return event

    def bubble_event(self, event: IcoRuntimeEvent) -> None:
        """
        Propagate a runtime event upward until a contour or agent host is reached.
        """
        # If event is handled and should not propagate further, stop here
        next_event = self.on_event(event)

        if next_event is None:
            return

        if self._runtime_parent:
            self._runtime_parent.bubble_event(next_event)

    # ────────────────────────────────────────────────
    # Runtime Tree Management
    # ────────────────────────────────────────────────

    def connect_runtime(self, runtime: IcoRuntimeNode) -> Self:
        """Connect to a runtime host for command propagation."""
        if runtime not in self._runtime_children:
            self._runtime_children.append(runtime)
        runtime._runtime_parent = self
        return self

    def disconnect_runtime(self, runtime: IcoRuntimeNode) -> Self:
        """Disconnect from a runtime host."""
        if runtime in self._runtime_children:
            self._runtime_children.remove(runtime)
        runtime._runtime_parent = None
        return self

    # ────────────────────────────────────────────────
    # Runtime Commands API
    # ────────────────────────────────────────────────

    def activate(self) -> Self:
        """Broadcast 'activate' event through the entire flow."""
        self.broadcast_command(IcoRuntimeCommand.activate())
        return self

    def run(self) -> Self:
        """Broadcast 'run' event through the entire flow."""
        self.broadcast_command(IcoRuntimeCommand.run())
        return self

    def reset(self) -> Self:
        """Broadcast 'reset' event through the entire flow."""
        self.broadcast_command(IcoRuntimeCommand.reset())
        return self

    def deactivate(self) -> Self:
        """Broadcast 'deactivate' event through the entire flow."""
        self.broadcast_command(IcoRuntimeCommand.deactivate())
        return self

    def pause(self) -> Self:
        """Broadcast 'pause' event through the entire flow."""
        self.broadcast_command(IcoRuntimeCommand.pause())
        return self

    def resume(self) -> Self:
        """Broadcast 'resume' event through the entire flow."""
        self.broadcast_command(IcoRuntimeCommand.resume())
        return self

    def stop(self) -> Self:
        """Broadcast 'stop' event through the entire flow."""
        self.broadcast_command(IcoRuntimeCommand.stop())
        return self

    # ────────────────────────────────────────────────
    # Describe util interface
    # ────────────────────────────────────────────────

    def describe(self) -> None:
        from apriori.ico.core.meta.describer import describe as describe_util
        from apriori.ico.core.meta.flow_meta import IcoFlowMeta

        meta = IcoFlowMeta.from_node(self)
        describe_util(meta)

    # ────────────────────────────────────────────────
    # Tools
    # ────────────────────────────────────────────────

    def add_tool(self, tool: IcoRuntimeNode) -> IcoRuntimeNode:
        """Attach a runtime tool to this node."""
        tool.connect_runtime(self)
        return tool


def iterate_nodes(
    node: IcoRuntimeNode,
) -> Iterator[IcoRuntimeNode]:
    """Recursively yield all children operators in the flow tree."""
    yield node
    for c in node.runtime_children:
        yield from iterate_nodes(c)


def iterate_parents(
    node: IcoRuntimeNode,
) -> Iterator[IcoRuntimeNode]:
    """Recursively yield all parent operators in the flow tree."""
    if node.runtime_parent is None:
        return

    yield node.runtime_parent
    yield from iterate_parents(node.runtime_parent)

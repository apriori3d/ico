from __future__ import annotations

from abc import ABC
from collections.abc import Iterator, Sequence
from enum import Enum, auto

from typing_extensions import Self

from apriori.ico.core.runtime.command import (
    IcoActivateCommand,
    IcoDeactivateCommand,
    IcoPauseCommand,
    IcoResetCommand,
    IcoResumeCommand,
    IcoRunCommand,
    IcoRuntimeCommand,
)
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.state import BaseStateModel, IcoRuntimeState

# ────────────────────────────────────────────────
# State of runtime node
# ────────────────────────────────────────────────


class IcoRuntimeStateOld(Enum):
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


DEFAULT_COMMAND_TO_STATE = {
    IcoActivateCommand: IcoRuntimeStateOld.ready,
    IcoResetCommand: IcoRuntimeStateOld.ready,
    IcoDeactivateCommand: IcoRuntimeStateOld.inactive,
    IcoPauseCommand: IcoRuntimeStateOld.paused,
    IcoResumeCommand: IcoRuntimeStateOld.ready,
}


# ────────────────────────────────────────────────
# Runtime node
# ────────────────────────────────────────────────


class IcoRuntimeNode(ABC):
    """Structural attributes for graph representation of ICO operators."""

    __slots__ = (
        "state_model",
        "runtime_name",
        "_runtime_children",
        "_runtime_parent",
    )

    state_model: BaseStateModel
    runtime_name: str | None
    _runtime_children: list[IcoRuntimeNode]
    _runtime_parent: IcoRuntimeNode | None

    def __init__(
        self,
        runtime_name: str | None = None,
        runtime_parent: IcoRuntimeNode | None = None,
        runtime_children: Sequence[IcoRuntimeNode] | None = None,
        state_model: BaseStateModel | None = None,
    ) -> None:
        self.runtime_name = runtime_name
        self._runtime_parent = runtime_parent
        self._runtime_children = (
            list(runtime_children) if runtime_children is not None else []
        )
        self.state_model = state_model or BaseStateModel()

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
    def runtime_children(self) -> list[IcoRuntimeNode]:
        """Children nodes in the runtime tree."""
        return self._runtime_children

    @property
    def state(self) -> IcoRuntimeState:
        """Current runtime state of the operator."""
        return self.state_model.state

    # ────────────────────────────────────────────────
    # Commands
    # ────────────────────────────────────────────────

    def on_command(self, command: IcoRuntimeCommand) -> IcoRuntimeCommand:
        """
        Handle a single runtime command.

        Subclasses may override to implement additional behavior
        (e.g., resource allocation, reset hooks, or teardown logic).
        """
        self.state_model.update(command)
        return command

    def broadcast_command(self, command: IcoRuntimeCommand) -> IcoRuntimeCommand:
        match command.broadcast_order:
            case "Pre-order":
                return self.broadcast_command_pre_order(command)
            case "Post-order":
                return self.broadcast_command_post_order(command)

    def broadcast_command_pre_order(
        self,
        command: IcoRuntimeCommand,
    ) -> IcoRuntimeCommand:
        """
        Recursively propagate a runtime command through the operator tree.
        """
        next_command = self.on_command(command)

        for child in self._runtime_children:
            next_command = child.broadcast_command(next_command)

        return next_command

    def broadcast_command_post_order(
        self,
        command: IcoRuntimeCommand,
    ) -> IcoRuntimeCommand:
        """
        Recursively propagate a runtime command through the operator tree.
        Use post-order traversal to ensure children receive commands before parents.
        Applicable for deactivation and teardown sequences.
        """

        for child in self._runtime_children:
            next_command = child.broadcast_command(command)

        next_command = self.on_command(command)

        return next_command

    # ────────────────────────────────────────────────
    # Events
    # ────────────────────────────────────────────────

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        """
        Handle a single runtime event.

        Subclasses may override to implement additional behavior
        (e.g., logging, metrics, or alerting).
        """
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
        self.broadcast_command(IcoActivateCommand())
        return self

    def run(self) -> Self:
        """Broadcast 'run' event through the entire flow."""
        self.broadcast_command(IcoRunCommand())
        return self

    def deactivate(self) -> Self:
        """Broadcast 'deactivate' event through the entire flow."""
        self.broadcast_command(IcoDeactivateCommand())
        return self

    # ────────────────────────────────────────────────
    # Describe util interface
    # ────────────────────────────────────────────────

    def describe(self) -> None:
        from apriori.ico.inspect.describer import describe as describe_util

        describe_util(self)

    # ────────────────────────────────────────────────
    # Tools
    # ────────────────────────────────────────────────

    def add_tool(self, tool: IcoRuntimeNode) -> IcoRuntimeNode:
        """Attach a runtime tool to this node."""
        tool.connect_runtime(self)
        return tool

    # ────────────────────────────────────────────────
    # Hirarchy Iteration
    # ────────────────────────────────────────────────

    def iterate_nodes(self) -> Iterator[IcoRuntimeNode]:
        """Recursively yield all children operators in the flow tree."""
        yield self
        for c in self.runtime_children:
            yield from c.iterate_nodes()

    def iterate_parents(self) -> Iterator[IcoRuntimeNode]:
        """Recursively yield all parent operators in the flow tree."""
        if self.runtime_parent is None:
            return

        yield self.runtime_parent
        yield from self.runtime_parent.iterate_parents()

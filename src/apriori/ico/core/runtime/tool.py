from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from typing_extensions import Self

from apriori.ico.core.runtime.command import (
    IcoActivateCommand,
    IcoDeactivateCommand,
    IcoRuntimeCommand,
)
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.core.runtime.state import (
    BaseStateModel,
    IcoRuntimeState,
    IdleState,
    ReadyState,
    StateTransitionMap,
)

# ────────────────────────────────────────────────
# Discovery Protocol: Command and Event
# ────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class IcoDiscoveryCommand(IcoRuntimeCommand):
    node_types: set[type[IcoDiscovarableNode]]
    register_id: int = 0

    def match(self, node: IcoRuntimeNode) -> bool:
        return any(isinstance(node, node_type) for node_type in self.node_types)

    def next(self) -> IcoDiscoveryCommand:
        return IcoDiscoveryCommand(
            node_types=self.node_types,
            register_id=self.register_id + 1,
        )


@dataclass(slots=True, frozen=True)
class IcoRegistrationEvent(IcoRuntimeEvent):
    node_type: type[IcoDiscovarableNode]
    node_name: str | None
    node_id: int


# ────────────────────────────────────────────────
# Discovery State Model
# ────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class PendingState(IcoRuntimeState):
    name: ClassVar[str] = "Pending"


class DiscoverableStateModel(BaseStateModel):
    """State model for nodes that requires discovery. Node become Ready after discovery."""

    transitions: ClassVar[StateTransitionMap] = {
        IcoActivateCommand: PendingState,
        IcoDiscoveryCommand: ReadyState,
        IcoDeactivateCommand: IdleState,
    }


# ────────────────────────────────────────────────
# Discoverable Node
# ────────────────────────────────────────────────


class IcoDiscovarableNode(IcoRuntimeNode):
    type_name: ClassVar[str] = "Discoverable Node"

    __slots__ = ("registered_id",)

    registered_id: int | None

    def __init__(
        self,
        runtime_name: str | None = None,
        runtime_parent: IcoRuntimeNode | None = None,
        runtime_children: Sequence[IcoRuntimeNode] | None = None,
    ) -> None:
        IcoRuntimeNode.__init__(
            self,
            runtime_name=runtime_name,
            runtime_parent=runtime_parent,
            runtime_children=runtime_children,
        )
        self.state_model = DiscoverableStateModel()
        self.registered_id = None

    def on_command(self, command: IcoRuntimeCommand) -> IcoRuntimeCommand:
        # Discard command if it is a discovery command that does not match
        if isinstance(command, IcoDiscoveryCommand) and not command.match(self):
            return command

        # Apply command handling if it is accepted
        command = super().on_command(command)

        # Register node upon discovery command if it matches
        if isinstance(command, IcoDiscoveryCommand):
            self.registered_id = command.register_id
            self._register_node()
            command = command.next()

        elif isinstance(command, IcoDeactivateCommand):
            self.registered_id = None

        return command

    def _register_node(self) -> None:
        """Implement discovery contract"""
        assert self.registered_id is not None

        self.bubble_event(
            IcoRegistrationEvent(
                node_type=type(self),
                node_name=self.runtime_name,
                node_id=self.registered_id,
            )
        )


# ────────────────────────────────────────────────
# Tool Base Class (Abstract)
# ────────────────────────────────────────────────


class IcoRuntimeTool(IcoRuntimeNode, ABC):
    type_name: ClassVar[str] = "Runtime Tool"

    __slots__ = ("registry",)

    registry: dict[int, IcoRegistrationEvent]

    def __init__(
        self,
        runtime_name: str | None = None,
        runtime_parent: IcoRuntimeNode | None = None,
        runtime_children: Sequence[IcoRuntimeNode] | None = None,
    ):
        IcoRuntimeNode.__init__(
            self,
            runtime_name=runtime_name,
            runtime_parent=runtime_parent,
            runtime_children=runtime_children,
        )
        self.registry = {}

    @abstractmethod
    def get_discoverable_node_types(self) -> set[type[IcoDiscovarableNode]]:
        """Get discoverable node types for this tool."""
        raise NotImplementedError()

    @abstractmethod
    def get_registration_event_types(self) -> set[type[IcoRegistrationEvent]]:
        """Get registration event types for this tool."""
        raise NotImplementedError()

    def discover(self) -> Self:
        """Discover tool nodes in the runtime."""

        self.broadcast_command(
            IcoDiscoveryCommand(node_types=self.get_discoverable_node_types())
        )
        return self

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        if isinstance(event, IcoRegistrationEvent):
            # Handle registration event and stop propagation if node registered
            node_registered = self._register_node(event)
            return None if node_registered else event

        return super().on_event(event)

    def _register_node(self, event: IcoRegistrationEvent) -> bool:
        """Update internal registry based on the event type."""

        for event_type in self.get_registration_event_types():
            if isinstance(event, event_type):
                self.registry[event.node_id] = event
                return True

        return False

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from apriori.ico.core.runtime.command import (
    IcoActivateCommand,
    IcoDeactivateCommand,
    IcoRuntimeCommand,
)
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.core.runtime.state import (
    ActiveState,
    BaseStateModel,
    IcoRuntimeState,
    InactiveState,
    ReadyState,
)


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
    node_type: type[IcoRuntimeNode]
    node_name: str
    node_id: int


class DiscoverableStateModel(BaseStateModel):
    """State model for nodes that requires discovery. Node become Ready after discovery."""

    transitions: dict[type[IcoRuntimeCommand], type[IcoRuntimeState]] = {
        IcoActivateCommand: ActiveState,
        IcoDiscoveryCommand: ReadyState,
        IcoDeactivateCommand: InactiveState,
    }


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

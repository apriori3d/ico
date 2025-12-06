from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from apriori.ico.core.runtime.command import (
    IcoActivateCommand,
    IcoDeactivateCommand,
    IcoRuntimeCommand,
)
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.node import IcoRuntimeNode, IcoRuntimeState


@dataclass(slots=True, frozen=True)
class IcoDiscoveryCommand(IcoRuntimeCommand):
    node_types: set[type[IcoRuntimeNode]]
    register_id: int = 0

    def match(self, node: IcoRuntimeNode) -> bool:
        return type(node) in self.node_types

    def next(self) -> IcoDiscoveryCommand:
        return IcoDiscoveryCommand(
            node_types=self.node_types,
            register_id=self.register_id + 1,
        )


@dataclass(slots=True, frozen=True)
class IcoDiscoveryEvent(IcoRuntimeEvent):
    node_type: type[IcoRuntimeNode]
    node_name: str
    node_id: int


class IcoDiscovarableNode(IcoRuntimeNode):
    _COMMAND_TO_STATE = {
        **IcoRuntimeNode._COMMAND_TO_STATE,
        # Discoverable nodes become ready after registration
        IcoActivateCommand: IcoRuntimeState.inactive,
    }

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
        self.registered_id = None

    def on_command(self, command: IcoRuntimeCommand) -> IcoRuntimeCommand:
        command = super().on_command(command)

        # Register node upon discovery command
        if isinstance(command, IcoDiscoveryCommand) and command.match(self):
            self.bubble_event(
                IcoDiscoveryEvent(
                    node_type=type(self),
                    node_name=self.runtime_name,
                    node_id=command.register_id,
                )
            )
            self.registered_id = command.register_id
            self._set_state(IcoRuntimeState.ready)
            command = command.next()

        elif isinstance(command, IcoDeactivateCommand):
            self.registered_id = None

        return command

    def _ensure_is_ready(self) -> None:
        if self.state != IcoRuntimeState.ready:
            raise RuntimeError(
                f"{self.runtime_name} is not ready. Use Discovery command to register node. Current state: {self.state}."
            )

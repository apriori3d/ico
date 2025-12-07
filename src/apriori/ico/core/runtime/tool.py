from collections.abc import Sequence
from typing import ClassVar

from typing_extensions import Self

from apriori.ico.core.runtime.discovery import (
    IcoDiscovarableNode,
    IcoDiscoveryCommand,
    IcoRegistrationEvent,
)
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.node import IcoRuntimeNode


class IcoRuntimeTool(IcoRuntimeNode):
    type_name: ClassVar[str] = "Runtime Tool"
    discoverable_node_types: set[type[IcoDiscovarableNode]] = set()
    registration_event_types: set[type[IcoRegistrationEvent]] = set()

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

    def discover(self) -> Self:
        """Discover tool nodes in the runtime."""

        self.broadcast_command(
            IcoDiscoveryCommand(node_types=type(self).discoverable_node_types)
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

        for event_type in type(self).registration_event_types:
            if isinstance(event, event_type):
                self.registry[event.node_id] = event
                return True

        return False

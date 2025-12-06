from dataclasses import dataclass
from typing import Generic, final

from apriori.ico.core.operator import I, IcoOperator
from apriori.ico.core.runtime.command import IcoRuntimeCommand
from apriori.ico.core.runtime.discovery import (
    IcoDiscovarableNode,
    IcoDiscoveryCommand,
    IcoDiscoveryEvent,
)
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.node import IcoRuntimeState


@final
@dataclass(slots=True, frozen=True)
class IcoProgressDiscoveryEvent(IcoDiscoveryEvent):
    total: float


@final
@dataclass(slots=True, frozen=True)
class IcoProgressEvent(IcoRuntimeEvent):
    node_id: int
    advance: float


class IcoProgress(
    Generic[I],
    IcoOperator[I, I],
    IcoDiscovarableNode,
):
    total: float

    def __init__(
        self,
        total: float,
        *,
        name: str | None = None,
    ) -> None:
        IcoDiscovarableNode.__init__(self, runtime_name=name)
        IcoOperator.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            fn=self._progress_fn,
            name=name or "Progress",
        )
        self.total = total

    def _progress_fn(self, item: I) -> I:
        self._ensure_is_ready()
        assert self.registered_id is not None

        self.bubble_event(IcoProgressEvent(node_id=self.registered_id, advance=1))
        return item

    def on_command(self, command: IcoRuntimeCommand) -> IcoRuntimeCommand:
        # Implement discovery contract
        if isinstance(command, IcoDiscoveryCommand) and command.match(self):
            self.bubble_event(
                IcoProgressDiscoveryEvent(
                    node_type=type(self),
                    node_name=self.runtime_name,
                    node_id=command.register_id,
                    total=self.total,
                )
            )
            self.registered_id = command.register_id
            self._set_state(IcoRuntimeState.ready)
            command = command.next()
        else:
            command = super().on_command(command)

        return command

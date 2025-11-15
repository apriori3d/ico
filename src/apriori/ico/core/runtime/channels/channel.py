from __future__ import annotations

from typing import Generic

from apriori.ico.core.dsl.operator import IcoOperator
from apriori.ico.core.runtime.channels.types import (
    IcoReceiveEndpointProtocol,
    IcoRuntimeChannelProtocol,
    IcoSendEndpointProtocol,
)
from apriori.ico.core.runtime.events import IcoRuntimeEvent
from apriori.ico.core.runtime.runtime_operator import IcoRuntimeOperator
from apriori.ico.core.runtime.types import IcoRuntimeCommandType, IcoRuntimeFlowProtocol
from apriori.ico.core.types import I, O


class IcoRuntimeChannelMixin(
    Generic[I, O],
    IcoRuntimeOperator,
    IcoRuntimeChannelProtocol[I, O],
):
    send: IcoSendEndpointProtocol[I]
    receive: IcoReceiveEndpointProtocol[O]

    def __init__(
        self,
        send: IcoSendEndpointProtocol[I],
        receive: IcoReceiveEndpointProtocol[O],
        *,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.send = send
        self.receive = receive
        self.name = name or f"MPQueueChannel-{id(self)}"

        # Connect remote runtime via endpoint runtime port
        self.receive.runtime = self

    def on_command(self, command: IcoRuntimeCommandType) -> None:
        super().on_command(command)

        # Chanel with a parent runtime forwards commands upstream
        if self.runtime_parent:
            self.send.on_command(command)

    def on_event(self, event: IcoRuntimeEvent) -> None:
        super().on_event(event)

        # Channel without a parent runtime bubbles events downstream
        if not self.runtime_parent:
            self.send.on_event(event)


class IcoReceiveEndpointMixin(
    Generic[O],
    IcoOperator[None, O],
    IcoReceiveEndpointProtocol[O],
):
    def on_command(self, command: IcoRuntimeCommandType) -> None:
        if self.runtime:
            self.runtime.broadcast_command(command)

    def on_event(self, event: IcoRuntimeEvent) -> None:
        if self.runtime:
            self.runtime.bubble_event(event)


class IcoSendEndpointMixin(
    Generic[I],
    IcoOperator[I, None],
    IcoSendEndpointProtocol[I],
    IcoRuntimeFlowProtocol,
): ...

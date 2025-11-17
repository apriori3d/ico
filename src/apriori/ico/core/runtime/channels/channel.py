from __future__ import annotations

from typing import Any, Generic

from apriori.ico.core.runtime.channels.types import (
    IcoReceiveEndpointProtocol,
    IcoSendEndpointProtocol,
)
from apriori.ico.core.runtime.runtime_operator import IcoRuntimeOperator
from apriori.ico.core.runtime.types import (
    IcoRuntimeCommandType,
    IcoRuntimeEventProtocol,
    IcoRuntimeOperatorProtocol,
)
from apriori.ico.core.types import I, O


class IcoRuntimeChannel(
    IcoRuntimeOperator,
    Generic[I, O],
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
        super().__init__(name=name or f"MPQueueChannel-{id(self)}")
        self.send = send
        self.receive = receive

        # Connect remote runtime via endpoint runtime port
        self.receive.runtime = self

    def on_command(self, command: IcoRuntimeCommandType) -> None:
        super().on_command(command)

        # Chanel with a parent runtime forwards commands downstream
        if self._runtime_parent:
            self.send.on_command(command)

    def on_event(self, event: IcoRuntimeEventProtocol) -> None:
        super().on_event(event)

        # Channel without a parent runtime bubbles events upstream
        if not self._runtime_parent:
            self.send.on_event(event)


class IcoReceiveEndpointMixin:
    runtime: IcoRuntimeOperatorProtocol | None = None

    def __init__(
        self,
        runtime: IcoRuntimeOperatorProtocol | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.runtime = runtime

    def on_command(self, command: IcoRuntimeCommandType) -> None:
        """Handle runtime connection logic for received commands."""
        if self.runtime:
            self.runtime.broadcast_command(command)

    def on_event(self, event: IcoRuntimeEventProtocol) -> None:
        """Handle runtime connection logic for received events."""
        if self.runtime:
            self.runtime.bubble_event(event)


class IcoSendEndpointMixin:
    def on_command(self, command: IcoRuntimeCommandType) -> None:
        pass  # To be implemented by subclasses

    def on_event(self, event: IcoRuntimeEventProtocol) -> None:
        pass  # To be implemented by subclasses

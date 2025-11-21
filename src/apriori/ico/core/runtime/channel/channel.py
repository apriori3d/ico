from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic

from apriori.ico.core.operator import I, O
from apriori.ico.core.runtime.channel.messages import (
    ChannelMessage,
    InputPayload,
    RuntimeCommandPayload,
    RuntimeEventPayload,
)
from apriori.ico.core.runtime.command import IcoRuntimeCommand
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.node import IcoRuntimeNode


class IcoRuntimeChannel(
    Generic[I, O],
):
    output: IcoSendEndpoint[I]
    input: IcoReceiveEndpoint[O]

    def __init__(
        self,
        output: IcoSendEndpoint[I],
        input: IcoReceiveEndpoint[O],
    ) -> None:
        self.output = output
        self.input = input


class IcoSendEndpoint(Generic[I], ABC):
    @abstractmethod
    def _send(self, message: ChannelMessage) -> None: ...

    # ────────────────────────────────
    # Send functions
    # ────────────────────────────────

    def send_item(self, item: I) -> None:
        """Handle sending of data items."""
        payload = InputPayload(item)
        self._send(payload.wrap())

    def send_command(self, command: IcoRuntimeCommand) -> None:
        """Handle sending of runtime commands."""
        payload = RuntimeCommandPayload(command)
        self._send(payload.wrap())

    def send_event(self, event: IcoRuntimeEvent) -> None:
        """Handle sending of runtime events."""
        payload = RuntimeEventPayload(event)
        self._send(payload.wrap())


class IcoReceiveEndpoint(Generic[O], ABC):
    runtime_port: IcoRuntimeNode

    @abstractmethod
    def receive(self) -> O | IcoRuntimeCommand | IcoRuntimeEvent: ...

    def _on_command(self, command: IcoRuntimeCommand) -> None:
        self.runtime_port.on_command(command)

    def _on_event(self, event: IcoRuntimeEvent) -> None:
        self.runtime_port.on_event(event)

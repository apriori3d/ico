from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic

from apriori.ico.core.operator import I, O
from apriori.ico.core.runtime.command import IcoRuntimeCommand
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.node import IcoRuntimeNode


class IcoRuntimeChannel(Generic[I, O], ABC):
    __slots__ = ("output", "input")

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
    __slots__ = ()

    @abstractmethod
    def send(self, payload: I | IcoRuntimeCommand | IcoRuntimeEvent) -> None: ...


class IcoReceiveEndpoint(Generic[O], ABC):
    __slots__ = ("runtime_port",)

    runtime_port: IcoRuntimeNode

    @abstractmethod
    def receive(self) -> O | IcoRuntimeCommand | IcoRuntimeEvent: ...

    def _on_command(self, command: IcoRuntimeCommand) -> None:
        self.runtime_port.on_command(command)

    def _on_event(self, event: IcoRuntimeEvent) -> None:
        self.runtime_port.on_event(event)

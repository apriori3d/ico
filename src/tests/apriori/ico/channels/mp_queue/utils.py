from __future__ import annotations

from typing import Generic

from apriori.ico.core.operator import I, IcoOperator, O
from apriori.ico.core.runtime.channel.utils import wait_for_item
from apriori.ico.core.runtime.command import IcoRuntimeCommand
from apriori.ico.core.runtime.event import IcoRuntimeEvent, IcoRuntimeEventType
from apriori.ico.core.runtime.exceptions import IcoRuntimeError
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.runtime.channels.mp_queue.channel import MPQueueChannel


class MPProcessMock(Generic[I, O], IcoOperator[I, O], IcoRuntimeNode):
    def __init__(
        self,
        channel: MPQueueChannel[I, O],
    ) -> None:
        super().__init__(fn=self._portal_fn)
        self._channel = channel

    def _portal_fn(self, input: I) -> O:
        # Send item to agent process
        self._channel.output.send(input)
        item = wait_for_item(
            runtime_node=self,
            endpoint=self._channel.input,
            accept_commands=False,
            accept_events=True,
        )
        assert item is not None
        return item

    def on_command(self, command: IcoRuntimeCommand) -> None:
        super().on_command(command)
        self._channel.output.send(command)

    def on_event(self, event: IcoRuntimeEvent) -> None:
        if event.type == IcoRuntimeEventType.fault:
            raise IcoRuntimeError(f"Agent exception: {event.meta['message']}")

        super().on_event(event)

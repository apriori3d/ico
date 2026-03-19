from __future__ import annotations

from typing import Generic

from ico.core.operator import I, IcoOperator, O
from ico.core.runtime.channel.channel import IcoChannel
from ico.core.runtime.command import IcoRuntimeCommand
from ico.core.runtime.event import (
    IcoFaultEvent,
    IcoRuntimeEvent,
)
from ico.core.runtime.exceptions import IcoRuntimeError
from ico.core.runtime.node import IcoRuntimeNode


class MPProcessMock(
    Generic[I, O],
    IcoOperator[I, O],
    IcoRuntimeNode,
):
    def __init__(
        self,
        channel: IcoChannel[I, O],
    ) -> None:
        IcoOperator.__init__(self, fn=self._portal_fn)  # pyright: ignore[reportUnknownMemberType]
        IcoRuntimeNode.__init__(self)
        self._channel = channel
        self._channel.runtime_port = self

    def _portal_fn(self, input: I) -> O:
        # Send item to agent process
        self._channel.send(input)
        item = self._channel.wait_for_item()
        assert item is not None
        return item

    def on_channel_command(self, command: IcoRuntimeCommand) -> None:
        """Handle commands received from the channel."""
        self.on_command(command)

    def on_command(self, command: IcoRuntimeCommand) -> None:
        super().on_command(command)
        self._channel.send_command(command)

    def on_channel_event(self, event: IcoRuntimeEvent) -> None:
        """Handle events received from the channel."""
        self.on_event(event)

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        if isinstance(event, IcoFaultEvent):
            raise IcoRuntimeError(f"Agent exception: {event.info['message']}")

        return super().on_event(event)

# apriori/ico/core/runtime/mp_queue_channel.py
from __future__ import annotations

from multiprocessing import Queue
from multiprocessing.context import SpawnContext
from typing import TYPE_CHECKING, Generic, final

from apriori.ico.channels.mp_queue.receive_endpoint import (
    MPQueueReceiveEndpoint,
)
from apriori.ico.channels.mp_queue.send_endpoint import MPQueueSendEndpoint
from apriori.ico.core.runtime.channels.channel import IcoRuntimeChannel
from apriori.ico.core.runtime.channels.messages import (
    ChannelMessage,
)
from apriori.ico.core.runtime.types import IcoRuntimeCommandType
from apriori.ico.core.types import I, O

if TYPE_CHECKING:
    ChannelQueue = Queue[ChannelMessage]
else:
    ChannelQueue = Queue  # noqa: F401


@final
class MPQueueChannel(
    Generic[I, O],
    IcoRuntimeChannel[I, O],
):
    mp_context: SpawnContext
    _send: MPQueueSendEndpoint[I]
    _receive: MPQueueReceiveEndpoint[O]

    _main_queue: ChannelQueue
    _ack_queue: ChannelQueue
    _queues_owned: bool

    def __init__(
        self,
        mp_context: SpawnContext,
        main_queue: ChannelQueue | None = None,
        ack_queue: ChannelQueue | None = None,
        queues_owned: bool = True,
        name: str | None = None,
    ) -> None:
        main_queue = main_queue or mp_context.Queue()
        ack_queue = ack_queue or mp_context.Queue()

        # Define endpoints
        send = MPQueueSendEndpoint[I](
            main_queue=main_queue,
            ack_queue=ack_queue,
            name=f"{name}_send_endpoint" if name else None,
        )

        receive = MPQueueReceiveEndpoint[O](
            main_queue=main_queue,
            ack_queue=ack_queue,
            name=f"{name}_receive_endpoint" if name else None,
        )

        super().__init__(
            send=send,
            receive=receive,
            name=name or "mp_queue_channel",
        )
        self.mp_context = mp_context
        self._send = send
        self._receive = receive
        self._main_queue = main_queue
        self._ack_queue = ack_queue
        self._queues_owned = queues_owned

    def on_command(self, command: IcoRuntimeCommandType) -> None:
        super().on_command(command)

        # Handle close command, if queues were created by this channel,
        # the opposite case is the detached copy used in another process
        if command == IcoRuntimeCommandType.deactivate and self._queues_owned:
            self._send.close()
            self._receive.close()

    def reverse(self) -> MPQueueChannel[I, O]:
        return MPQueueChannel[I, O](
            mp_context=self.mp_context,
            main_queue=self._main_queue,
            ack_queue=self._ack_queue,
            queues_owned=False,
            name=self.name,
        )

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
    __slots__ = (
        "mp_context",
        "_send",
        "_receive",
        "_in_queue",
        "_out_queue",
        "_in_ack_queue",
        "_out_ack_queue",
        "_queues_owned",
    )

    mp_context: SpawnContext
    _send: MPQueueSendEndpoint[I]
    _receive: MPQueueReceiveEndpoint[O]

    _in_queue: ChannelQueue
    _out_queue: ChannelQueue
    _in_ack_queue: ChannelQueue
    _out_ack_queue: ChannelQueue
    _queues_owned: bool

    def __init__(
        self,
        mp_context: SpawnContext,
        in_queue: ChannelQueue | None = None,
        out_queue: ChannelQueue | None = None,
        in_ack_queue: ChannelQueue | None = None,
        out_ack_queue: ChannelQueue | None = None,
        queues_owned: bool = True,
        name: str | None = None,
    ) -> None:
        in_queue = in_queue or mp_context.Queue()
        out_queue = out_queue or mp_context.Queue()
        in_ack_queue = in_ack_queue or mp_context.Queue()
        out_ack_queue = out_ack_queue or mp_context.Queue()

        # Define endpoints
        send = MPQueueSendEndpoint[I](
            main_queue=out_queue,
            ack_queue=out_ack_queue,
            name=f"{name}_send_endpoint" if name else None,
        )

        receive = MPQueueReceiveEndpoint[O](
            main_queue=in_queue,
            ack_queue=in_ack_queue,
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
        self._in_queue = in_queue
        self._out_queue = out_queue
        self._in_ack_queue = in_ack_queue
        self._out_ack_queue = out_ack_queue
        self._queues_owned = queues_owned

    def on_command(self, command: IcoRuntimeCommandType) -> None:
        super().on_command(command)

        # Handle close command, if queues were created by this channel,
        # the opposite case is the detached copy used in another process
        if command == IcoRuntimeCommandType.deactivate and self._queues_owned:
            self._in_queue.close()
            self._in_queue.join_thread()
            self._out_queue.close()
            self._out_queue.join_thread()
            self._in_ack_queue.close()
            self._in_ack_queue.join_thread()
            self._out_ack_queue.close()
            self._out_ack_queue.join_thread()

    def make_pair(self) -> MPQueueChannel[O, I]:
        """Create a paired channel for the opposite endpoint roles."""

        return MPQueueChannel[O, I](
            mp_context=self.mp_context,
            in_queue=self._out_queue,
            out_queue=self._in_queue,
            in_ack_queue=self._out_ack_queue,
            out_ack_queue=self._in_ack_queue,
            queues_owned=False,
            name=f"{self.name}_pair",
        )

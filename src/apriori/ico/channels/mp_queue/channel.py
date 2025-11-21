# apriori/ico/core/runtime/mp_queue_channel.py
from __future__ import annotations

from multiprocessing import Queue
from multiprocessing.context import SpawnContext
from typing import TYPE_CHECKING, Generic, cast, final

from apriori.ico.channels.mp_queue.receive_endpoint import (
    MPQueueReceiveEndpoint,
)
from apriori.ico.channels.mp_queue.send_endpoint import MPQueueSendEndpoint
from apriori.ico.core.operator import I, O
from apriori.ico.core.runtime.channel.channel import (
    IcoReceiveEndpoint,
    IcoRuntimeChannel,
    IcoSendEndpoint,
)
from apriori.ico.core.runtime.channel.messages import (
    ChannelMessage,
)

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
        "_mp_context",
        "output",
        "input",
        "_queues_owned",
        "_input_queue",
        "_output_queue",
        "_input_ack_queue",
        "_output_ack_queue",
    )

    output: IcoSendEndpoint[I]
    input: IcoReceiveEndpoint[O]

    _mp_context: SpawnContext
    _input_queue: ChannelQueue
    _input_ack_queue: ChannelQueue
    _output_queue: ChannelQueue
    _output_ack_queue: ChannelQueue
    _queues_owned: bool

    def __init__(
        self,
        mp_context: SpawnContext,
        input_queue: ChannelQueue | None = None,
        input_ack_queue: ChannelQueue | None = None,
        output_queue: ChannelQueue | None = None,
        output_ack_queue: ChannelQueue | None = None,
    ) -> None:
        if input_queue and output_queue and input_ack_queue and output_ack_queue:
            self._input_queue = input_queue
            self._output_queue = output_queue
            self._input_ack_queue = input_ack_queue
            self._output_ack_queue = output_ack_queue
            self._queues_owned = False
        elif (
            not input_queue
            and not output_queue
            and not input_ack_queue
            and not output_ack_queue
        ):
            self._input_queue = cast(ChannelQueue, mp_context.Queue())
            self._output_queue = cast(ChannelQueue, mp_context.Queue())
            self._input_ack_queue = cast(ChannelQueue, mp_context.Queue())
            self._output_ack_queue = cast(ChannelQueue, mp_context.Queue())
            self._queues_owned = True
        else:
            raise ValueError(
                "Either provide all queues or none when initializing MPQueueChannel."
            )

        # Define endpoints
        self.output = MPQueueSendEndpoint[I](self._output_queue, self._output_ack_queue)
        self.input = MPQueueReceiveEndpoint[O](self._input_queue, self._input_ack_queue)
        self._mp_context = mp_context

    def close(self) -> None:
        """Close owned queues."""
        if self._queues_owned:
            self._input_queue.close()
            self._input_queue.join_thread()
            self._output_queue.close()
            self._output_queue.join_thread()
            self._input_ack_queue.close()
            self._input_ack_queue.join_thread()
            self._output_ack_queue.close()
            self._output_ack_queue.join_thread()

    def make_pair(self) -> MPQueueChannel[O, I]:
        """Create a paired channel for the opposite endpoint roles."""

        return MPQueueChannel[O, I](
            mp_context=self._mp_context,
            input_queue=self._output_queue,
            input_ack_queue=self._output_ack_queue,
            output_queue=self._input_queue,
            output_ack_queue=self._input_ack_queue,
        )

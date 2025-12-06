# apriori/ico/core/runtime/mp_queue_channel.py
from __future__ import annotations

from multiprocessing import Queue
from multiprocessing.context import SpawnContext
from typing import Generic, final

from apriori.ico.core.operator import I, O
from apriori.ico.core.runtime.channel.channel import (
    IcoChannel,
    IcoReceiveEndpoint,
    IcoSendEndpoint,
)
from apriori.ico.core.runtime.channel.messages import DataMessage, SystemMessageTypes
from apriori.ico.core.runtime.node import IcoRuntimeNode


@final
class MPQueueSendEndpoint(Generic[I], IcoSendEndpoint[I]):
    __slots__ = "queue"

    queue: Queue[DataMessage[I] | SystemMessageTypes]

    def __init__(self, main_queue: Queue[DataMessage[I] | SystemMessageTypes]) -> None:
        self.queue = main_queue

    def send(self, message: DataMessage[I] | SystemMessageTypes) -> None:
        self.queue.put(message)

    def close(self) -> None:
        """Close queues."""

        self.queue.close()
        self.queue.join_thread()


@final
class MPQueueReceiveEndpoint(Generic[O], IcoReceiveEndpoint[O]):
    __slots__ = "queue"
    queue: Queue[DataMessage[O] | SystemMessageTypes]

    def __init__(self, main_queue: Queue[DataMessage[O] | SystemMessageTypes]) -> None:
        self.queue = main_queue

    def receive(self) -> DataMessage[O] | SystemMessageTypes:
        return self.queue.get()

    def close(self) -> None:
        """Close queues."""

        self.queue.close()
        self.queue.join_thread()


class MPChannel(Generic[I, O], IcoChannel[I, O]):
    __slots__ = "mp_context"

    mp_context: SpawnContext

    def __init__(
        self,
        mp_context: SpawnContext,
        send_endpoint: MPQueueSendEndpoint[I] | None = None,
        receive_endpoint: MPQueueReceiveEndpoint[O] | None = None,
        *,
        runtime_port: IcoRuntimeNode | None = None,
        timeout: int = 5,
        ignore_receive_timeouts: bool = True,
        accept_commands: bool = True,
        accept_events: bool = True,
        strict_accept: bool = False,
    ) -> None:
        if not send_endpoint:
            sender_queue: Queue[DataMessage[I] | SystemMessageTypes] = (
                mp_context.Queue()
            )
            send_endpoint = MPQueueSendEndpoint[I](sender_queue)

        if not receive_endpoint:
            receiver_queue: Queue[DataMessage[O] | SystemMessageTypes] = (
                mp_context.Queue()
            )
            receive_endpoint = MPQueueReceiveEndpoint[O](receiver_queue)

        super().__init__(
            send_endpoint,
            receive_endpoint,
            runtime_port=runtime_port,
            timeout=timeout,
            ignore_receive_timeouts=ignore_receive_timeouts,
            accept_commands=accept_commands,
            accept_events=accept_events,
            strict_accept=strict_accept,
        )
        self.mp_context = mp_context

    def make_agent_channel(self) -> MPChannel[O, I]:
        """Create inverted channel pair for agent process."""
        assert isinstance(self.sender, MPQueueSendEndpoint)
        assert isinstance(self.receiver, MPQueueReceiveEndpoint)

        send_endpoint = MPQueueSendEndpoint[O](self.receiver.queue)
        receive_endpoint = MPQueueReceiveEndpoint[I](self.sender.queue)

        return MPChannel[O, I](
            mp_context=self.mp_context,
            send_endpoint=send_endpoint,
            receive_endpoint=receive_endpoint,
            accept_commands=True,
            accept_events=False,
            strict_accept=True,
        )

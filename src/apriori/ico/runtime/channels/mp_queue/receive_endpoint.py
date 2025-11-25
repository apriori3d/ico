from __future__ import annotations

from multiprocessing import Queue
from typing import Generic, final

from apriori.ico.core.operator import O
from apriori.ico.core.runtime.channel.channel import IcoReceiveEndpoint
from apriori.ico.core.runtime.channel.messages import (
    AcknowledgeChannelMessage,
    ChannelMessage,
)
from apriori.ico.core.runtime.command import IcoRuntimeCommand
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.progress.mixin import ProgressMixin


@final
class MPQueueReceiveEndpoint(
    Generic[O],
    IcoReceiveEndpoint[O],
    ProgressMixin,
):
    """
    ReceiveEndpoint for multiprocessing Queue-based communication.

    Responsibilities:
      • Blocking wait for incoming messages
      • Dispatch by message type
      • Handle runtime commands and events
      • Acknowledge receipt to the sender
    """

    __slots__ = ("_main_queue", "_ack_queue", "_timeout")

    _main_queue: Queue[ChannelMessage[O | IcoRuntimeCommand | IcoRuntimeEvent]]
    _ack_queue: Queue[AcknowledgeChannelMessage]
    _timeout: float

    def __init__(
        self,
        main_queue: Queue[ChannelMessage[O | IcoRuntimeCommand | IcoRuntimeEvent]],
        ack_queue: Queue[AcknowledgeChannelMessage],
        *,
        timeout: float = 5.0,
    ) -> None:
        ProgressMixin.__init__(self)

        self._main_queue = main_queue
        self._ack_queue = ack_queue
        self._timeout = timeout  # Seconds

    def receive(self) -> O | IcoRuntimeCommand | IcoRuntimeEvent:
        message = self._main_queue.get(timeout=self._timeout)
        # Send acknowledgment to the sender.
        self._ack_queue.put(AcknowledgeChannelMessage(message.message_type))

        return message.payload

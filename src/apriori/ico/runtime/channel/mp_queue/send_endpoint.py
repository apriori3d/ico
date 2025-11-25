from __future__ import annotations

import queue
from multiprocessing import Queue
from typing import Generic, final

from apriori.ico.core.operator import I
from apriori.ico.core.runtime.channel.channel import (
    IcoSendEndpoint,
)
from apriori.ico.core.runtime.channel.messages import (
    AcknowledgeChannelMessage,
    ChannelMessage,
    MessageType,
    wrap_payload,
)
from apriori.ico.core.runtime.command import IcoRuntimeCommand
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.progress.mixin import ProgressMixin


@final
class MPQueueSendEndpoint(
    Generic[I],
    IcoSendEndpoint[I],
    ProgressMixin,
):
    """
    SendEndpoint for multiprocessing Queue-based communication.

    Responsibilities:
      • Wrap data and control messages into ChannelMessage envelopes
      • Send them through the main queue
      • Wait for acknowledgment (ACK)
      • React to runtime events (faults, metrics, etc.)
    """

    __slots__ = ("_main_queue", "_ack_queue", "_timeout")

    _main_queue: Queue[ChannelMessage[I | IcoRuntimeCommand | IcoRuntimeEvent]]
    _ack_queue: Queue[AcknowledgeChannelMessage]
    _timeout: int

    def __init__(
        self,
        main_queue: Queue[ChannelMessage[I | IcoRuntimeCommand | IcoRuntimeEvent]],
        ack_queue: Queue[AcknowledgeChannelMessage],
        *,
        timeout: int = 5,
    ) -> None:
        ProgressMixin.__init__(self)
        self._main_queue = main_queue
        self._ack_queue = ack_queue
        self._timeout = timeout

    # ────────────────────────────────
    # Send + Acknowledge logic
    # ────────────────────────────────

    def send(self, payload: I | IcoRuntimeCommand | IcoRuntimeEvent) -> None:
        """Send a payload and wait for acknowledgment."""
        message = wrap_payload(payload)
        self._main_queue.put(message)
        self._wait_for_ack(message, timeout=self._timeout)

    def _wait_for_ack(
        self,
        pending_message: ChannelMessage[object],
        *,
        timeout: int = 5,
    ) -> None:
        """Wait for acknowledgment or handle runtime events from peer."""
        try:
            ack_message = self._ack_queue.get(timeout=timeout)

        except queue.Empty as e:
            raise TimeoutError(
                f"No ACK received for {type(pending_message.payload).__name__} within {timeout}s"
            ) from e

        if not ack_message.message_type == MessageType.acknowledge:
            raise RuntimeError(f"Expected ACK message, got {ack_message.message_type}")

        if ack_message.payload != pending_message.message_type:
            raise RuntimeError(
                f"Unexpected ACK: expected {pending_message.message_type}, got {ack_message.payload}"
            )

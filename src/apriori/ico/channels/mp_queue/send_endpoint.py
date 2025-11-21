from __future__ import annotations

import queue
from multiprocessing import Queue
from typing import TYPE_CHECKING, Generic, cast, final

from apriori.ico.core.operator import I
from apriori.ico.core.runtime.channel.channel import (
    IcoSendEndpoint,
)
from apriori.ico.core.runtime.channel.messages import (
    AcknowledgePayload,
    ChannelMessage,
    ChannelMessageType,
)
from apriori.ico.core.runtime.progress.mixin import ProgressMixin

if TYPE_CHECKING:
    ChannelQueue = Queue[ChannelMessage]
else:
    ChannelQueue = Queue  # noqa: F401


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

    _main_queue: ChannelQueue
    _ack_queue: ChannelQueue
    _timeout: int

    def __init__(
        self,
        main_queue: ChannelQueue,
        ack_queue: ChannelQueue,
        *,
        timeout: int = 5,
    ) -> None:
        IcoSendEndpoint[I].__init__(self)
        ProgressMixin.__init__(self)
        self._main_queue = main_queue
        self._ack_queue = ack_queue
        self._timeout = timeout

    # ────────────────────────────────
    # Send + Acknowledge logic
    # ────────────────────────────────

    def _send(self, message: ChannelMessage) -> None:
        """Send a payload and wait for acknowledgment."""
        self._main_queue.put(message)
        self._wait_for_ack(message, timeout=self._timeout)

    def _wait_for_ack(self, pending_message: ChannelMessage, timeout: int = 5) -> None:
        """Wait for acknowledgment or handle runtime events from peer."""
        try:
            message = self._ack_queue.get(timeout=timeout)

        except queue.Empty as e:
            raise TimeoutError(
                f"No ACK received for {type(pending_message.payload).__name__} within {timeout}s"
            ) from e

        if not message.message_type == ChannelMessageType.acknowledge:
            raise RuntimeError(f"Expected ACK message, got {message.message_type}")

        self._handle_ack(message, pending_message)

    def _handle_ack(
        self, message: ChannelMessage, pending_message: ChannelMessage
    ) -> None:
        """Confirm acknowledgment matches the pending message."""
        ack_payload = cast(AcknowledgePayload, message.unwrap())
        if ack_payload.ack_message_type != pending_message.message_type:
            raise RuntimeError(
                f"Unexpected ACK: expected {pending_message.message_type}, got {ack_payload.ack_message_type}"
            )

from __future__ import annotations

import queue
from collections.abc import Callable
from multiprocessing import Queue
from typing import TYPE_CHECKING, Generic, cast, final

from apriori.ico.core.runtime.channels.channel import (
    IcoSendEndpoint,
)
from apriori.ico.core.runtime.channels.messages import (
    AcknowledgePayload,
    ChannelMessage,
    ChannelMessagePayload,
    ChannelMessageType,
    InputPayload,
    RuntimeCommandPayload,
    RuntimeEventPayload,
)
from apriori.ico.core.runtime.progress.mixin import ProgressMixin
from apriori.ico.core.runtime.types import (
    IcoRuntimeCommandType,
    IcoRuntimeEventProtocol,
)
from apriori.ico.core.types import I, IcoNodeType

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
        name: str | None = None,
        timeout: int = 5,
    ) -> None:
        super().__init__(
            fn=self._send_fn,
            name=name or "mp_queue_send_endpoint",
            node_type=IcoNodeType.operator,
        )
        self._main_queue = main_queue
        self._ack_queue = ack_queue
        self._timeout = timeout

    # ────────────────────────────────
    # Main send function
    # ────────────────────────────────
    def _send_fn(self, item: I) -> None:
        """Send a single data item downstream."""
        self._send_input(item)

    def _send_input(self, item: I) -> None:
        """Handle sending of data items."""
        payload = InputPayload(item)
        self._send(payload)

    # ────────────────────────────────
    # Runtime command and event propagation
    # ────────────────────────────────

    def on_command(self, command: IcoRuntimeCommandType) -> None:
        """Handle sending of runtime commands."""
        payload = RuntimeCommandPayload(command)
        self._send(payload)

    def on_event(self, event: IcoRuntimeEventProtocol) -> None:
        """Handle sending of runtime events."""
        payload = RuntimeEventPayload(event)
        self._send(payload)

    # ────────────────────────────────
    # Core send logic
    # ────────────────────────────────

    def _send(self, payload: ChannelMessagePayload) -> None:
        """Send a payload and wait for acknowledgment."""
        message = payload.wrap()
        self._main_queue.put(message)
        self._wait_for_ack(message, timeout=self._timeout)

    def _wait_for_ack(self, pending_message: ChannelMessage, timeout: int = 5) -> None:
        """Wait for acknowledgment or handle runtime events from peer."""
        while True:
            try:
                message = self._ack_queue.get(timeout=timeout)

            except queue.Empty as e:
                raise TimeoutError(
                    f"No ACK received for {type(pending_message.payload).__name__} within {timeout}s"
                ) from e

            handler = self._ack_dispatch_table().get(message.message_type)
            if handler:
                handler(message, pending_message)
                return  # Exit once handler confirms success

            raise RuntimeError(
                f"Unexpected message type in ACK queue: {message.message_type}"
            )

    # ────────────────────────────────
    # ACK Handlers
    # ────────────────────────────────

    def _ack_dispatch_table(
        self,
    ) -> dict[ChannelMessageType, Callable[[ChannelMessage, ChannelMessage], None]]:
        """Mapping of message types to acknowledgment handlers."""
        return {
            ChannelMessageType.acknowledge: self._handle_ack,
            ChannelMessageType.runtime_event: self._handle_runtime_event,
        }

    def _handle_ack(
        self, message: ChannelMessage, pending_message: ChannelMessage
    ) -> None:
        """Confirm acknowledgment matches the pending message."""
        ack_payload = cast(AcknowledgePayload, message.unwrap())
        if ack_payload.ack_message_type != pending_message.message_type:
            raise RuntimeError(
                f"Unexpected ACK: expected {pending_message.message_type}, got {ack_payload.ack_message_type}"
            )

    def _handle_runtime_event(self, message: ChannelMessage, _: ChannelMessage) -> None:
        """Handle runtime events received while waiting for ACK."""
        payload = cast(RuntimeEventPayload, message.unwrap())
        event = payload.event
        event.raise_if_fault()
        raise RuntimeError(f"Unexpected runtime event during ACK wait: {event.type}")

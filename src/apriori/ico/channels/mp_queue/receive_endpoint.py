from __future__ import annotations

from multiprocessing import Queue
from typing import TYPE_CHECKING, Generic, cast, final

from apriori.ico.core.operator import O
from apriori.ico.core.runtime.channel.channel import IcoReceiveEndpoint
from apriori.ico.core.runtime.channel.messages import (
    AcknowledgePayload,
    ChannelMessage,
    ChannelMessageType,
    InputPayload,
    RuntimeCommandPayload,
    RuntimeEventPayload,
)
from apriori.ico.core.runtime.command import IcoRuntimeCommand
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.progress.mixin import ProgressMixin

if TYPE_CHECKING:
    ChannelQueue = Queue[ChannelMessage]
else:
    ChannelQueue = Queue  # noqa: F401


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

    _main_queue: ChannelQueue
    _ack_queue: ChannelQueue
    _timeout: float

    def __init__(
        self,
        main_queue: ChannelQueue,
        ack_queue: ChannelQueue,
        *,
        timeout: float = 5.0,
    ) -> None:
        IcoReceiveEndpoint[O].__init__(self)
        ProgressMixin.__init__(self)

        self._main_queue = main_queue
        self._ack_queue = ack_queue
        self._timeout = timeout  # seconds

    # ────────────────────────────────

    def receive(self) -> O | IcoRuntimeCommand | IcoRuntimeEvent:
        message = self._main_queue.get(timeout=self._timeout)

        # Dispatch based on message type
        match message.message_type:
            case ChannelMessageType.input:
                return self._handle_input(message)
            case ChannelMessageType.runtime_command:
                return self._handle_command(message)
            case ChannelMessageType.runtime_event:
                return self._handle_event(message)
            case _:
                raise RuntimeError(f"Unknown message type: {message.message_type}")

    # ────────────────────────────────
    # Handlers
    # ────────────────────────────────

    def _handle_input(self, message: ChannelMessage) -> O:
        """Handle normal data input message."""
        self._ack(ChannelMessageType.input)
        input_payload = cast(InputPayload, message.unwrap())
        return cast(O, input_payload.input)

    def _handle_command(self, message: ChannelMessage) -> IcoRuntimeCommand:
        """Handle runtime command (activate, reset, stop, etc.)."""
        payload = cast(RuntimeCommandPayload, message.unwrap())
        command = payload.command

        # Propagate command to connected runtime
        self._ack(ChannelMessageType.runtime_command)

        return command

    def _handle_event(self, message: ChannelMessage) -> IcoRuntimeEvent:
        """Handle runtime events (faults, metrics, progress, etc.)."""
        payload = cast(RuntimeEventPayload, message.unwrap())
        event = payload.event

        # Event is fire and forget, acknowledge immediately
        self._ack(ChannelMessageType.runtime_event)

        return event

    # ────────────────────────────────
    # Utilities
    # ────────────────────────────────

    def _ack(self, msg_type: ChannelMessageType) -> None:
        """Send acknowledgment to the sender."""
        self._ack_queue.put(AcknowledgePayload(msg_type).wrap())

    def _log_unknown(self, message: ChannelMessage) -> None:
        self.progress.log(
            f"{self}: Ignoring unknown message type {message.message_type}"
        )

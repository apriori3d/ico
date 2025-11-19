from __future__ import annotations

from collections.abc import Callable
from multiprocessing import Queue
from typing import TYPE_CHECKING, Generic, cast, final

from apriori.ico.core.runtime.channels.channel import IcoReceiveEndpoint
from apriori.ico.core.runtime.channels.messages import (
    AcknowledgePayload,
    ChannelMessage,
    ChannelMessageType,
    InputPayload,
    RuntimeCommandPayload,
    RuntimeEventPayload,
)
from apriori.ico.core.runtime.events import IcoRuntimeEvent
from apriori.ico.core.runtime.exceptions import IcoRuntimeError, IcoStopExecutionSignal
from apriori.ico.core.runtime.progress.mixin import ProgressMixin
from apriori.ico.core.runtime.types import (
    IcoRuntimeOperatorProtocol,
)
from apriori.ico.core.types import O

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

    runtime: IcoRuntimeOperatorProtocol | None

    _main_queue: ChannelQueue
    _ack_queue: ChannelQueue

    def __init__(
        self,
        main_queue: ChannelQueue,
        ack_queue: ChannelQueue,
        runtime: IcoRuntimeOperatorProtocol | None = None,
        name: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        super().__init__(
            fn=self._receive_fn,
            name=name or "mp_queue_receive_endpoint",
        )
        self.runtime = runtime
        self._main_queue = main_queue
        self._ack_queue = ack_queue
        self._timeout = timeout  # seconds

    # ────────────────────────────────
    # Main receive loop
    # ────────────────────────────────

    def _receive_fn(self, _: None = None) -> O:
        """Blocking receive for a single item.

        Waits for a message in the queue, routes it to a handler,
        and returns the next input item to the runtime contour flow.
        """
        while True:
            try:
                message = self._main_queue.get(timeout=self._timeout)

                # Dispatch based on message type
                handler = self._dispatch_table().get(message.message_type)

                if handler is None:
                    self._log_unknown(message)
                    continue

                result = handler(message)
                if result is not None:
                    return result

            except (IcoStopExecutionSignal, IcoRuntimeError):
                # Propagate runtime signals upward
                raise
            except Exception as e:
                # Wrap and send back fault event
                fault_event = IcoRuntimeEvent.exception(e)
                self._ack_queue.put(RuntimeEventPayload(fault_event).wrap())
                raise

    # ────────────────────────────────
    # Dispatch table
    # ────────────────────────────────

    def _dispatch_table(
        self,
    ) -> dict[ChannelMessageType, Callable[[ChannelMessage], None | O]]:
        """Return mapping of message types to handler methods."""
        return {
            ChannelMessageType.input: self._handle_input,
            ChannelMessageType.runtime_command: self._handle_command,
            ChannelMessageType.runtime_event: self._handle_event,
        }

    # ────────────────────────────────
    # Handlers
    # ────────────────────────────────

    def _handle_input(self, message: ChannelMessage) -> O:
        """Handle normal data input message."""
        self._ack(ChannelMessageType.input)
        input_payload = cast(InputPayload, message.unwrap())
        return cast(O, input_payload.input)

    def _handle_command(self, message: ChannelMessage) -> None:
        """Handle runtime command (activate, reset, stop, etc.)."""
        payload = cast(RuntimeCommandPayload, message.unwrap())
        command = payload.command

        # Propagate command to connected runtime
        self.on_command(command)

        self._ack(ChannelMessageType.runtime_command)

    def _handle_event(self, message: ChannelMessage) -> None:
        """Handle runtime events (faults, metrics, progress, etc.)."""
        payload = cast(RuntimeEventPayload, message.unwrap())
        event = payload.event
        # Event is fire and forget, acknowledge immediately
        self._ack(ChannelMessageType.runtime_event)

        if event.is_fault:
            event.raise_if_fault()

        # Propagate event to connected runtime
        self.on_event(event)

    # ────────────────────────────────
    # Utilities
    # ────────────────────────────────

    def _ack(self, msg_type: ChannelMessageType) -> None:
        """Send acknowledgment to the sender."""
        self._ack_queue.put(AcknowledgePayload(msg_type).wrap())

    def _log_unknown(self, message: ChannelMessage) -> None:
        self.progress.log(
            f"{self.name}: Ignoring unknown message type {message.message_type}"
        )

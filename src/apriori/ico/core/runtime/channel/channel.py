from __future__ import annotations

from abc import ABC, abstractmethod
from queue import Empty
from typing import Generic

from apriori.ico.core.operator import I, O
from apriori.ico.core.runtime.channel.messages import (
    AcknowledgeMessage,
    CommandAcknowledgeMessage,
    CommandMessage,
    DataMessage,
    EventMessage,
    RuntimeMessage,
)
from apriori.ico.core.runtime.command import IcoDeactivateCommand, IcoRuntimeCommand
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.node import IcoRuntimeNode


class IcoSendEndpoint(Generic[I], ABC):
    @abstractmethod
    def send(self, message: DataMessage[I] | RuntimeMessage) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class IcoReceiveEndpoint(Generic[O], ABC):
    @abstractmethod
    def receive(self) -> DataMessage[O] | RuntimeMessage: ...

    @abstractmethod
    def close(self) -> None: ...


class IcoChannel(Generic[I, O], ABC):
    __slots__ = (
        "sender",
        "receiver",
        "timeout",
        "runtime_port",
        "ignore_receive_timeouts",
        "accept_commands",
        "accept_events",
        "strict_accept",
        "_message_id",
    )

    sender: IcoSendEndpoint[I]
    receiver: IcoReceiveEndpoint[O]
    runtime_port: IcoRuntimeNode | None
    ignore_receive_timeouts: bool
    timeout: int
    accept_commands: bool
    accept_events: bool
    strict_accept: bool
    _message_id: int

    def __init__(
        self,
        sender: IcoSendEndpoint[I],
        receiver: IcoReceiveEndpoint[O],
        *,
        runtime_port: IcoRuntimeNode | None = None,
        timeout: int = 5,
        ignore_receive_timeouts: bool = True,
        accept_commands: bool = True,
        accept_events: bool = True,
        strict_accept: bool = False,
    ) -> None:
        self.sender = sender
        self.receiver = receiver
        self.timeout = timeout
        self.runtime_port = runtime_port
        self.ignore_receive_timeouts = ignore_receive_timeouts
        self.accept_commands = accept_commands
        self.accept_events = accept_events
        self.strict_accept = strict_accept
        self._message_id = 0

    @abstractmethod
    def invert(self) -> IcoChannel[O, I]: ...

    # ────────────────────────────────
    # Send + Acknowledge logic
    # ────────────────────────────────

    def send(self, payload: I) -> None:
        """Send a payload and wait for acknowledgment."""
        message = DataMessage(self._message_id, payload)
        self._send(message)
        self._wait_for_ack(message.id)

    def send_command(self, payload: IcoRuntimeCommand) -> IcoRuntimeCommand:
        """Send a payload and wait for acknowledgment."""
        message = CommandMessage(self._message_id, payload)
        self._send(message)

        # Runtime command may be updated by peer runtime
        next_command = self._wait_for_ack(message.id)
        assert next_command is not None
        return next_command

    def send_event(self, payload: IcoRuntimeEvent) -> None:
        """Send a payload and wait for acknowledgment."""
        message = EventMessage(self._message_id, payload)
        self._send(message)
        self._wait_for_ack(message.id)

    def _send(self, message: DataMessage[I] | RuntimeMessage) -> None:
        self._message_id += 1
        self.sender.send(message)

    def _wait_for_ack(self, pending_message_id: int) -> IcoRuntimeCommand | None:
        input_item = self.wait_for_message()

        if not isinstance(input_item, AcknowledgeMessage):
            raise RuntimeError(
                f"Unexpected response: expected ACK, got {type(input_item)}"
            )
        if input_item.message_id != pending_message_id:
            raise RuntimeError(
                f"Unexpected ACK: expected {pending_message_id}, got {input_item.message_id}"
            )
        if isinstance(input_item, CommandAcknowledgeMessage):
            return input_item.command

        return None

    def wait_for_item(self) -> O | None:
        message = self.wait_for_message()

        if message is None:
            return None

        if not isinstance(message, DataMessage):
            raise RuntimeError(
                f"Unexpected response: expected Data item, got {type(message)}"
            )
        return message.payload

    def wait_for_message(self) -> DataMessage[O] | AcknowledgeMessage | None:
        """Wait for output item or acknowledgment and handle incoming runtime commands/events from peer."""
        while True:
            try:
                input_message = self.receiver.receive()

            except (TimeoutError, Empty):
                if self.ignore_receive_timeouts:
                    continue
                else:
                    raise

            match input_message:
                case DataMessage():
                    self._send(
                        AcknowledgeMessage(
                            self._message_id, message_id=input_message.id
                        )
                    )
                    return input_message

                case CommandMessage():
                    self._handle_input_command(input_message)

                    if isinstance(input_message.command, IcoDeactivateCommand):
                        return None  # Exit on deactivate command

                case EventMessage():
                    self._handle_input_event(input_message)

                case AcknowledgeMessage():
                    return input_message

                case _:
                    raise RuntimeError(
                        f"Unexpected message type: {type(input_message)}"
                    )

    def _handle_input_command(self, message: CommandMessage) -> None:
        if not self.accept_commands:
            if self.strict_accept:
                raise RuntimeError("Runtime commands are not accepted.")
            return  # Ignore command, wait for actual data item

        if self.runtime_port is not None:
            next_command = self.runtime_port.broadcast_command(message.command)
        else:
            next_command = message.command

        self._send(
            CommandAcknowledgeMessage(
                self._message_id,
                message_id=message.id,
                command=next_command,
            )
        )

    def _handle_input_event(self, message: EventMessage) -> None:
        if not self.accept_events:
            if self.strict_accept:
                raise RuntimeError("Runtime events are not accepted.")
            return  # Ignore event, wait for actual data item

        if self.runtime_port is None:
            return

        self.runtime_port.bubble_event(message.event)

        self._send(AcknowledgeMessage(self._message_id, message.id))

    def close(self) -> None:
        """Close channel endpoints."""
        self.receiver.close()
        self.sender.close()

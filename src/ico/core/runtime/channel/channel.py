from __future__ import annotations

from abc import ABC, abstractmethod
from queue import Empty
from typing import Generic, Protocol

from ico.core.operator import I, O
from ico.core.runtime.channel.messages import (
    AcknowledgeMessage,
    CommandMessage,
    DataMessage,
    EventMessage,
    RuntimeMessage,
)
from ico.core.runtime.command import IcoDeactivateCommand, IcoRuntimeCommand
from ico.core.runtime.event import IcoFaultEvent, IcoRuntimeEvent
from ico.core.runtime.exceptions import IcoRuntimeError

# ────────────────────────────────────────────────
# Channel endpoint abstractions
# ────────────────────────────────────────────────


class IcoSendEndpoint(Generic[I], ABC):
    """Abstract interface for sending messages through communication channels.

    IcoSendEndpoint defines the sending side of bidirectional communication
    in distributed ICO execution. Enables pluggable transport mechanisms
    while maintaining type safety for message transmission.

    Transport Abstraction:
        • Process-agnostic: Works with multiprocessing, threading, networking
        • Type safety: Generic parameterization for payload types
        • Message envelope: Accepts both data and runtime control messages
        • Resource management: Cleanup through close() method

    Implementation Examples:
        • Multiprocessing queues for local distributed execution
        • Network sockets for remote distributed execution
        • Message brokers for enterprise distributed systems
        • In-memory queues for testing and development

    Generic Parameters:
        I: Type of data payloads being sent through this endpoint
    """

    @abstractmethod
    def send(self, message: DataMessage[I] | RuntimeMessage) -> None:
        """Send message through the communication endpoint.

        Args:
            message: Data or runtime message to transmit.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close endpoint and release communication resources."""
        ...


class IcoReceiveEndpoint(Generic[O], ABC):
    """Abstract interface for receiving messages through communication channels.

    IcoReceiveEndpoint defines the receiving side of bidirectional communication
    in distributed ICO execution. Provides transport abstraction while
    maintaining type safety for message reception.

    Transport Abstraction:
        • Blocking/non-blocking: Implementation-dependent receive behavior
        • Type safety: Generic parameterization for expected payload types
        • Message envelope: Returns both data and runtime control messages
        • Timeout handling: May raise TimeoutError or Empty exceptions

    Receive Patterns:
        • Blocking: Wait indefinitely for message arrival
        • Timeout: Wait with configurable timeout limits
        • Non-blocking: Return immediately with exception if no messages
        • Polling: Repeatedly check for message availability

    Generic Parameters:
        O: Type of data payloads expected from this endpoint
    """

    @abstractmethod
    def receive(self) -> DataMessage[O] | RuntimeMessage:
        """Receive message from communication endpoint.

        Returns:
            Data or runtime message received from remote endpoint.

        Raises:
            TimeoutError: If receive timeout expires.
            Empty: If no messages available (non-blocking mode).
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close endpoint and release communication resources."""
        ...


class IcoChannelRuntimePort(Protocol):
    """Protocol for connecting channels to runtime tree nodes.

    IcoChannelRuntimePort enables integration between communication channels
    and runtime tree infrastructure, allowing distributed runtime coordination
    through message-based command and event propagation.

    Runtime Integration:
        • Command handling: Process runtime commands from remote endpoints
        • Event handling: Process runtime events from remote endpoints
        • Distributed coordination: Bridge IPC and runtime tree semantics
        • Protocol-based: Flexible integration with various runtime nodes

    Usage Pattern:
        Runtime nodes (agents, workers) implement this protocol to enable
        channel-based distributed runtime coordination across processes.
    """

    def on_channel_command(self, command: IcoRuntimeCommand) -> None:
        """Handle runtime command received through channel.

        Args:
            command: Runtime command from remote process.
        """
        ...

    def on_channel_event(self, event: IcoRuntimeEvent) -> None:
        """Handle runtime event received through channel.

        Args:
            event: Runtime event from remote process.
        """
        ...


# ────────────────────────────────────────────────
# Bidirectional communication channel
# ────────────────────────────────────────────────


class IcoChannel(Generic[I, O], ABC):
    """Bidirectional communication channel for distributed ICO execution.

    IcoChannel provides sophisticated message-based communication between
    agent and worker processes, enabling distributed computation with
    unified runtime tree coordination and reliable message delivery.

    Channel Architecture:
        • Bidirectional Communication: Separate send/receive endpoints
        • Type Safety: Generic parameterization for input/output types
        • Multi-Protocol: Data messages + runtime control messages
        • Reliable Delivery: ACK/timeout mechanisms for critical messages
        • Runtime Integration: Direct connection to runtime tree nodes

    Communication Patterns:
        • Data Flow: Computation payloads (I → O) between processes
        • Command Distribution: Runtime commands from agent to worker
        • Event Propagation: Runtime events from worker to agent
        • Acknowledgment Protocol: Reliable delivery confirmation

    Message Processing:
        • Automatic ACK generation for received data messages
        • Runtime command forwarding to connected runtime port
        • Runtime event handling with fault propagation
        • Message ID tracking for delivery confirmation

    Configuration Options:
        • Timeout handling: Configurable receive timeouts
        • Message filtering: Accept/reject commands and events
        • Strict mode: Error vs ignore for unexpected messages
        • Runtime port: Optional connection to runtime tree node

    Distributed Execution Integration:
        Channels enable transparent distributed execution by maintaining
        unified communication semantics while bridging computation flow
        and runtime tree coordination across process boundaries.

    Generic Parameters:
        I: Type of input data sent through this channel
        O: Type of output data received through this channel

    Note:
        Channel direction is from sender perspective - agent sends I
        and receives O, while worker receives I and sends O.
    """

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
    runtime_port: IcoChannelRuntimePort | None
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
        runtime_port: IcoChannelRuntimePort | None = None,
        timeout: int = 5,
        ignore_receive_timeouts: bool = True,
        accept_commands: bool = True,
        accept_events: bool = True,
        strict_accept: bool = False,
    ) -> None:
        """Initialize bidirectional channel with endpoints and configuration.

        Args:
            sender: Endpoint for sending messages to remote process.
            receiver: Endpoint for receiving messages from remote process.
            runtime_port: Runtime node for forwarding commands/events (optional).
            timeout: Receive timeout in seconds for message operations.
            ignore_receive_timeouts: Continue waiting if receive times out.
            accept_commands: Process runtime commands from remote endpoint.
            accept_events: Process runtime events from remote endpoint.
            strict_accept: Raise errors vs ignore unexpected message types.

        Channel Configuration:
            • Runtime integration: Optional port for runtime tree coordination
            • Timeout behavior: Configurable handling of receive timeouts
            • Message filtering: Selective processing of command/event messages
            • Error handling: Strict vs permissive message type validation
        """
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
    def invert(self) -> IcoChannel[O, I]:
        """Create inverted channel with reversed input/output types.

        Returns:
            Channel with swapped send/receive endpoints and reversed types.

        Usage:
            Agent creates channel[I,O] and worker uses inverted channel[O,I]
            to maintain proper type relationships across process boundaries.
        """
        ...

    # ────────────────────────────────
    # Send + Acknowledge logic
    # ────────────────────────────────

    def send(self, payload: I, wait_for_ack: bool = True) -> None:
        """Send computation data payload with optional acknowledgment.

        Args:
            payload: Data item to send to remote process.
            wait_for_ack: Wait for delivery confirmation from receiver.

        Reliable Delivery:
            When wait_for_ack=True, blocks until receiver confirms receipt
            of the message, enabling reliable data transmission.
        """
        message = DataMessage(self._message_id, payload)
        self._send(message)

        if wait_for_ack:
            self._wait_for_ack(message.id)

    def send_command(
        self, payload: IcoRuntimeCommand, wait_for_ack: bool = True
    ) -> None:
        """Send runtime command with optional acknowledgment.

        Args:
            payload: Runtime command to send to remote process.
            wait_for_ack: Wait for delivery confirmation from receiver.

        Command Distribution:
            Enables distributed runtime tree coordination by forwarding
            lifecycle commands (activate, run, deactivate) to remote processes.
        """
        message = CommandMessage(self._message_id, payload)
        self._send(message)

        # Runtime command may be updated by peer runtime
        if wait_for_ack:
            self._wait_for_ack(message.id)

    def send_event(self, payload: IcoRuntimeEvent, wait_for_ack: bool = True) -> None:
        """Send runtime event with optional acknowledgment.

        Args:
            payload: Runtime event to send to remote process.
            wait_for_ack: Wait for delivery confirmation from receiver.

        Event Propagation:
            Enables distributed event bubbling by forwarding runtime events
            (state changes, errors, heartbeats) to remote processes.
        """
        message = EventMessage(self._message_id, payload)
        self._send(message)

        if wait_for_ack:
            self._wait_for_ack(message.id)

    def _send(self, message: DataMessage[I] | RuntimeMessage) -> None:
        """Internal method to send message with ID increment."""
        self._message_id += 1
        self.sender.send(message)

    def _wait_for_ack(self, pending_message_id: int) -> None:
        """Wait for acknowledgment message with matching ID.

        Args:
            pending_message_id: Expected acknowledgment message ID.

        Raises:
            RuntimeError: If wrong message type or mismatched ID received.
        """
        input_item = self.wait_for_message()

        if not isinstance(input_item, AcknowledgeMessage):
            raise RuntimeError(
                f"Unexpected response: expected ACK, got {type(input_item)}"
            )
        if input_item.message_id != pending_message_id:
            raise RuntimeError(
                f"Unexpected ACK: expected {pending_message_id}, got {input_item.message_id}"
            )

    def wait_for_item(self) -> O | None:
        """Wait for data payload from remote process.

        Returns:
            Data payload from remote process, or None on channel termination.

        Raises:
            RuntimeError: If non-data message received when data expected.

        Usage:
            Primary method for receiving computation data in distributed
            execution - blocks until data arrives or channel terminates.
        """
        message = self.wait_for_message()

        if message is None:
            return None

        if not isinstance(message, DataMessage):
            raise RuntimeError(
                f"Unexpected response: expected Data item, got {type(message)}"
            )
        return message.payload

    def wait_for_message(self) -> DataMessage[O] | AcknowledgeMessage | None:
        """Wait for message and handle runtime commands/events from peer.

        Core message processing loop that handles all incoming message types
        while waiting for specific expected messages (data or acknowledgments).

        Returns:
            DataMessage: Computation data from remote process
            AcknowledgeMessage: Delivery confirmation for sent message
            None: Channel termination signal (on deactivate command)

        Message Processing:
            • DataMessage: Auto-ACK and return for caller processing
            • CommandMessage: Forward to runtime port, ACK, continue waiting
            • EventMessage: Forward to runtime port, ACK, continue waiting
            • AcknowledgeMessage: Return immediately for sender confirmation
            • Deactivate command: Return None to signal termination

        Timeout Handling:
            Configurable behavior for receive timeouts - can ignore and retry
            or propagate timeout errors based on channel configuration.

        Error Handling:
            • Fault events converted to IcoRuntimeError exceptions
            • Unexpected message types raise RuntimeError
            • Timeout behavior controlled by ignore_receive_timeouts flag
        """
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
        """Handle runtime command received from remote process.

        Args:
            message: Command message from remote endpoint.

        Command Processing:
            • Configuration-based: Respects accept_commands setting
            • Runtime forwarding: Delegates to connected runtime port
            • Automatic ACK: Confirms command receipt to sender
            • Error handling: Strict vs permissive based on configuration
        """
        if not self.accept_commands:
            if self.strict_accept:
                raise RuntimeError("Runtime commands are not accepted.")
            return  # Ignore command, wait for actual data item

        if self.runtime_port is not None:
            self.runtime_port.on_channel_command(message.command)

        self._send(AcknowledgeMessage(self._message_id, message_id=message.id))

    def _handle_input_event(self, message: EventMessage) -> None:
        """Handle runtime event received from remote process.

        Args:
            message: Event message from remote endpoint.

        Event Processing:
            • Fault event handling: Convert to runtime errors immediately
            • Configuration-based: Respects accept_events setting
            • Runtime forwarding: Delegates to connected runtime port
            • Automatic ACK: Confirms event receipt to sender

        Special Cases:
            Fault events are converted to IcoRuntimeError exceptions
            after sending ACK, enabling distributed error propagation.
        """
        if isinstance(message.event, IcoFaultEvent):
            self._send(AcknowledgeMessage(self._message_id, message.id))
            raise IcoRuntimeError(
                f"Remote agent raise an exception: {message.event.info}"
            )

        if not self.accept_events:
            if self.strict_accept:
                raise RuntimeError("Runtime events are not accepted.")
            return  # Ignore event, wait for actual data item

        if self.runtime_port is None:
            return

        self.runtime_port.on_channel_event(message.event)
        self._send(AcknowledgeMessage(self._message_id, message.id))

    def close(self) -> None:
        """Close channel endpoints and release communication resources.

        Cleanup Method:
            Closes both sender and receiver endpoints, releasing any
            underlying transport resources (queues, sockets, etc.).
        """
        self.receiver.close()
        self.sender.close()

# apriori/ico/core/runtime/mp_queue_channel.py
from __future__ import annotations

from multiprocessing import Queue
from multiprocessing.context import SpawnContext
from typing import Generic, final

from ico.core.operator import I, O
from ico.core.runtime.channel.channel import (
    IcoChannel,
    IcoChannelRuntimePort,
    IcoReceiveEndpoint,
    IcoSendEndpoint,
)
from ico.core.runtime.channel.messages import DataMessage, RuntimeMessage

# ────────────────────────────────────────────────
# Multiprocessing queue-based endpoints
# ────────────────────────────────────────────────


@final
class MPQueueSendEndpoint(Generic[I], IcoSendEndpoint[I]):
    """Multiprocessing queue-based send endpoint for inter-process communication.

    MPQueueSendEndpoint provides message sending capabilities using
    Python's multiprocessing.Queue for cross-process data and runtime message
    transmission in ICO distributed execution environments.

    Queue-Based Architecture:
        • Thread-Safe: multiprocessing.Queue provides automatic synchronization
        • Process-Safe: Queue handles cross-process communication transparently
        • FIFO Ordering: Messages delivered in send order (First-In-First-Out)
        • Blocking Semantics: queue.put() blocks if queue is full (configurable)

    Message Types Supported:
        • DataMessage[I]: Computation data flowing between processes
        • RuntimeMessage: Commands, events, and control messages
        • Type Safety: Generic parameterization ensures type consistency

    Agent Integration:
        MPQueueSendEndpoint is used within IcoAgent/IcoAgentWorker for outgoing
        communication to remote processes. The endpoint handles serialization
        and cross-process transmission automatically within agent infrastructure.
    """

    __slots__ = "queue"

    queue: Queue[DataMessage[I] | RuntimeMessage]  # Underlying multiprocessing queue

    def __init__(self, main_queue: Queue[DataMessage[I] | RuntimeMessage]) -> None:
        """Initialize send endpoint with multiprocessing queue.

        Args:
            main_queue: Pre-configured multiprocessing.Queue for message transmission.

        Queue Configuration:
            • Queue should be created using appropriate multiprocessing context
            • Queue capacity can be limited (blocks on put() when full)
            • Queue handles serialization/deserialization automatically
        """
        self.queue = main_queue

    def send(self, message: DataMessage[I] | RuntimeMessage) -> None:
        """Send message through multiprocessing queue to receiving process.

        Args:
            message: Data or runtime message to send to remote process.

        Sending Process:
            1. Message serialization handled by multiprocessing.Queue
            2. Cross-process transmission via shared memory/pipes
            3. Blocking behavior if queue is full (configurable per queue)
            4. Automatic handling of process boundaries

        Thread Safety:
            Multiple threads can safely call send() on same endpoint.
            multiprocessing.Queue handles concurrent access automatically.
        """
        self.queue.put(message)

    def close(self) -> None:
        """Close queue and cleanup resources.

        Cleanup Process:
            • Close queue to prevent new messages
            • Wait for background thread to finish (join_thread())
            • Release system resources (memory, file descriptors)

        Note:
            Should be called when communication is complete to avoid
            resource leaks in long-running applications.
        """
        self.queue.close()
        self.queue.join_thread()


@final
class MPQueueReceiveEndpoint(Generic[O], IcoReceiveEndpoint[O]):
    """Multiprocessing queue-based receive endpoint for inter-process communication.

    MPQueueReceiveEndpoint provides reliable message reception capabilities using
    Python's multiprocessing.Queue for receiving data and runtime messages from
    remote processes in ICO distributed execution environments.

    Queue Reception Architecture:
        • Blocking Reception: queue.get() blocks until message available
        • FIFO Delivery: Messages received in same order as sent
        • Cross-Process: Receives messages from different process seamlessly
        • Deserialization: Automatic object reconstruction from queue data

    Message Processing Flow:
        Remote Process → Queue → get() → Deserialize → Local Process

    Agent Integration:
        MPQueueReceiveEndpoint is used within IcoAgent/IcoAgentWorker for incoming
        communication from remote processes. The endpoint handles deserialization
        and cross-process reception automatically within agent infrastructure.

    Exception Handling:
        Queue operations can raise exceptions in edge cases:
        • **EOFError**: Queue closed by sender
        • **OSError**: System-level queue errors
        • **pickle.PickleError**: Message deserialization failures
    """

    __slots__ = "queue"
    queue: Queue[DataMessage[O] | RuntimeMessage]  # Underlying multiprocessing queue

    def __init__(self, main_queue: Queue[DataMessage[O] | RuntimeMessage]) -> None:
        """Initialize receive endpoint with multiprocessing queue.

        Args:
            main_queue: Pre-configured multiprocessing.Queue for message reception.

        Queue Configuration:
            • Same queue instance used by corresponding send endpoint
            • Queue handles cross-process synchronization automatically
            • Queue capacity affects blocking behavior on sender side
        """
        self.queue = main_queue

    def receive(self) -> DataMessage[O] | RuntimeMessage:
        """Receive next message from multiprocessing queue (blocking).

        Returns:
            Next available message from remote process (DataMessage or RuntimeMessage).

        Reception Process:
            1. Block until message available in queue
            2. Deserialize message from queue storage
            3. Reconstruct typed message object
            4. Return message for processing

        Blocking Behavior:
            • get() blocks indefinitely until message available
            • Use queue.get(timeout=N) for timeout-based reception
            • No messages = indefinite wait (by design)

        Thread Safety:
            Multiple threads can safely call receive() on same endpoint.
            Each thread will receive different messages (round-robin).
        """
        return self.queue.get()

    def close(self) -> None:
        """Close queue and cleanup resources.

        Cleanup Process:
            • Close queue to prevent new messages
            • Wait for background thread to finish (join_thread())
            • Release system resources (memory, file descriptors)

        Coordination:
            Close should be coordinated between send and receive endpoints
            to avoid messages lost during shutdown.
        """
        self.queue.close()
        self.queue.join_thread()


# ────────────────────────────────────────────────
# Bidirectional multiprocessing channel
# ────────────────────────────────────────────────


class MPChannel(Generic[I, O], IcoChannel[I, O]):
    """Multiprocessing-based bidirectional channel for ICO agent/worker communication.

    MPChannel provides complete bidirectional inter-process communication using
    Python's multiprocessing.Queue infrastructure, enabling reliable data flow
    and runtime coordination between ICO Agent and Worker processes.

    Bidirectional Architecture:
        • **Send Capability**: Outgoing messages via MPQueueSendEndpoint[I]
        • **Receive Capability**: Incoming messages via MPQueueReceiveEndpoint[O]
        • **Runtime Integration**: Full IcoChannel compliance with command/event handling
        • **Channel Inversion**: Support for symmetric communication patterns

    Channel Pair Pattern:
        MPChannel supports bidirectional communication by creating inverted pairs:

        **Agent Side**: MPChannel[TaskData, ResultData]
        • Sends: TaskData to worker
        • Receives: ResultData from worker

        **Worker Side**: MPChannel[ResultData, TaskData] (inverted)
        • Sends: ResultData to agent
        • Receives: TaskData from agent

    Queue Creation Strategy:
        MPChannel automatically creates queues if not provided:
        • **Auto-Creation**: Uses provided SpawnContext to create fresh queues
        • **Manual Injection**: Accepts pre-configured endpoints for custom setups
        • **Context Management**: Ensures proper multiprocessing context usage

    Agent Integration:
        MPChannel is the primary communication infrastructure for ICO agent/worker
        distributed execution, providing reliable bidirectional IPC with automatic
        queue management and runtime message coordination.

    Channel Inversion Benefits:
        • **Symmetric Design**: Both processes use same channel interface
        • **Type Safety**: Generic parameters automatically inverted
        • **Runtime Coordination**: Command/event acceptance automatically flipped
        • **Resource Sharing**: Underlying queues shared between process pair

    Multiprocessing Context Integration:
        • **Spawn Context**: Preferred for cross-platform compatibility
        • **Fork Context**: Available on Unix systems (faster startup)
        • **Queue Management**: Context ensures proper queue creation and cleanup
        • **Process Safety**: All operations safe across process boundaries
    """

    __slots__ = "mp_context"

    mp_context: SpawnContext  # Multiprocessing context for queue creation

    def __init__(
        self,
        mp_context: SpawnContext,
        send_endpoint: MPQueueSendEndpoint[I] | None = None,
        receive_endpoint: MPQueueReceiveEndpoint[O] | None = None,
        *,
        runtime_port: IcoChannelRuntimePort | None = None,
        timeout: int = 5,
        ignore_receive_timeouts: bool = True,
        accept_commands: bool = True,
        accept_events: bool = True,
        strict_accept: bool = False,
    ) -> None:
        """Initialize bidirectional multiprocessing channel with automatic queue creation.

        Args:
            mp_context: Multiprocessing context (spawn/fork) for queue creation.
            send_endpoint: Optional pre-configured send endpoint (auto-created if None).
            receive_endpoint: Optional pre-configured receive endpoint (auto-created if None).
            runtime_port: Optional runtime port for command/event coordination.
            timeout: Timeout in seconds for blocking operations.
            ignore_receive_timeouts: Whether to ignore timeout exceptions on receive.
            accept_commands: Whether this channel accepts runtime commands.
            accept_events: Whether this channel accepts runtime events.
            strict_accept: Whether to enforce strict command/event acceptance rules.

        Initialization Process:
            1. **Endpoint Creation**: Auto-create missing endpoints using mp_context
            2. **Queue Allocation**: Create fresh multiprocessing.Queue instances
            3. **Parent Initialization**: Call IcoChannel.__init__ with full configuration
            4. **Context Storage**: Store mp_context for invert() operations

        Auto-Creation Strategy:
            When endpoints not provided:
            • **Send Queue**: ctx.Queue() for outgoing messages
            • **Receive Queue**: ctx.Queue() for incoming messages
            • **Independent Queues**: Separate queues for bidirectional communication
            • **Type Safety**: Generic parameters preserved in queue typing

        Runtime Configuration:
            • **Command Flow**: Agent typically accepts_commands=True, sends to worker
            • **Event Flow**: Worker typically accepts_events=False, sends to agent
            • **Bidirectional**: Both data flows work regardless of command/event settings
            • **Timeout Handling**: Configurable timeout behavior for robust communication

        Example - Custom Endpoint Configuration:
            ```python
            import multiprocessing as mp

            ctx = mp.get_context('spawn')

            # Create shared queues manually
            task_queue = ctx.Queue(maxsize=100)  # Limited capacity
            result_queue = ctx.Queue()

            # Create endpoints with shared queues
            send_ep = MPQueueSendEndpoint[Task](task_queue)
            recv_ep = MPQueueReceiveEndpoint[Result](result_queue)

            # Create channel with pre-configured endpoints
            channel = MPChannel[Task, Result](
                mp_context=ctx,
                send_endpoint=send_ep,
                receive_endpoint=recv_ep,
                timeout=30,
                accept_commands=True
            )
            ```
        """
        if not send_endpoint:
            sender_queue: Queue[DataMessage[I] | RuntimeMessage] = mp_context.Queue()
            send_endpoint = MPQueueSendEndpoint[I](sender_queue)

        if not receive_endpoint:
            receiver_queue: Queue[DataMessage[O] | RuntimeMessage] = mp_context.Queue()
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

    def invert(self) -> MPChannel[O, I]:
        """Create inverted channel pair for symmetric agent/worker communication.

        Returns:
            Inverted MPChannel with swapped send/receive queues and flipped runtime settings.

        Raises:
            AssertionError: If current endpoints are not MPQueue endpoints.

        Channel Pair Creation:
            Original and inverted channels share the same underlying queues but
            with swapped send/receive roles, enabling symmetric process communication:

            **Original Channel (Agent)**: MPChannel[Task, Result]
            • Sends via: task_queue (to worker)
            • Receives via: result_queue (from worker)
            • Commands: accept_commands=True (can send to worker)
            • Events: accept_events=False (receives from worker)

            **Inverted Channel (Worker)**: MPChannel[Result, Task]
            • Sends via: result_queue (to agent) - same queue, opposite direction
            • Receives via: task_queue (from agent) - same queue, opposite direction
            • Commands: accept_commands=False (receives from agent)
            • Events: accept_events=True (can send to agent)

        Queue Sharing Architecture:
            ```
            Agent Process          Shared Queues          Worker Process
            ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
            │ Send: Task  │────→ │ task_queue  │ ←──── │ Recv: Task  │
            │ Recv: Result│ ←──── │result_queue │────→ │ Send: Result│
            └─────────────┘       └─────────────┘       └─────────────┘
            ```

        Runtime Setting Inversion:
            • **accept_commands**: Flipped (only one side should accept)
            • **accept_events**: Flipped (opposite of commands for coordination)
            • **strict_accept**: Preserved (same validation rules)
            • **timeout**: Inherited from parent channel

        Benefits of Inversion Pattern:
            • **Symmetric Interface**: Both processes use same MPChannel API
            • **Automatic Configuration**: Runtime settings auto-flipped appropriately
            • **Resource Efficiency**: Shared queues minimize memory usage
            • **Type Safety**: Generic parameters automatically swapped
            • **Zero Copy**: No data duplication between channel pair
        """
        assert isinstance(self.sender, MPQueueSendEndpoint)
        assert isinstance(self.receiver, MPQueueReceiveEndpoint)

        return MPChannel[O, I](
            mp_context=self.mp_context,
            send_endpoint=MPQueueSendEndpoint[O](self.receiver.queue),
            receive_endpoint=MPQueueReceiveEndpoint[I](self.sender.queue),
            accept_commands=not self.accept_commands,
            accept_events=not self.accept_events,
            strict_accept=self.strict_accept,
        )

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Generic

from apriori.ico.core.operator import I, IcoOperator, O
from apriori.ico.core.runtime.channel.channel import IcoChannel
from apriori.ico.core.runtime.command import (
    IcoActivateCommand,
    IcoDeactivateCommand,
    IcoRuntimeCommand,
)
from apriori.ico.core.runtime.event import IcoFaultEvent, IcoRuntimeEvent
from apriori.ico.core.runtime.node import IcoRemotePlaceholderNode, IcoRuntimeNode
from apriori.ico.core.runtime.state import (
    BaseStateModel,
    IcoRuntimeState,
    ReadyState,
)
from apriori.ico.core.runtime.utils import discover_and_connect_runtime_nodes
from apriori.ico.core.signature import IcoSignature
from apriori.ico.core.signature_utils import infer_from_flow_factory

# ────────────────────────────────────────────────
# Agent-specific runtime states
# ────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class PendingState(IcoRuntimeState):
    """Agent worker state before activation completion.

    PendingState represents agent workers that are initializing but not yet
    ready for execution. Used during agent startup and resource allocation.
    """

    name: ClassVar[str] = "Pending"


@dataclass(slots=True, frozen=True)
class WaitingState(ReadyState):
    """Agent state waiting for input/output data exchange.

    WaitingState indicates the agent is ready and waiting for data items
    to arrive through the communication channel. Inherits ready status
    while indicating waiting behavior.
    """

    name: ClassVar[str] = "Waiting"


@dataclass(slots=True, frozen=True)
class SendingState(ReadyState):
    """Agent state actively sending data through communication channel.

    SendingState indicates the agent is transmitting data items through
    the inter-process communication channel. Maintains ready status
    while performing data transmission.
    """

    name: ClassVar[str] = "Sending"


# ────────────────────────────────────────────────
# Agent state management models
# ────────────────────────────────────────────────


class AgentStateModel(BaseStateModel):
    """State model for runtime agents with communication states.

    AgentStateModel extends base state management with specialized states
    for inter-process communication and coordination. Manages the additional
    states required for distributed execution scenarios.

    Agent-Specific States:
        • waiting(): Transition to WaitingState (waiting for data)
        • sending(): Transition to SendingState (transmitting data)

    State Transitions:
        • Standard transitions from BaseStateModel (activate/deactivate)
        • Communication transitions for data exchange coordination
        • Validated transitions ensuring proper ready-state prerequisites
    """

    def waiting(self) -> None:
        """Transition to waiting state for data exchange.

        Raises:
            RuntimeError: If not currently in a ready-compatible state.
        """
        if not self.state.is_ready:
            raise RuntimeError(
                "Cannot transition to Waiting state from non-Ready state."
            )
        self.update_state(WaitingState())

    def sending(self) -> None:
        """Transition to sending state for data transmission.

        Raises:
            RuntimeError: If not currently in a ready-compatible state.
        """
        if not self.state.is_ready:
            raise RuntimeError(
                "Cannot transition to Sending state from non-Ready state."
            )
        self.update_state(SendingState())


class AgentWorkerStateModel(AgentStateModel):
    """Extended state model for agent worker processes.

    AgentWorkerStateModel adds pending state management for agent workers
    that initialize in separate processes and require additional startup
    coordination before becoming ready for execution.

    Worker-Specific States:
        • pending(): Transition to PendingState (initialization phase)
        • Inherits all AgentStateModel communication states

    Usage:
        Used by IcoAgentWorker instances running in separate processes
        to coordinate their initialization and execution lifecycle.
    """

    def pending(self) -> None:
        """Transition to pending state during worker initialization.

        Raises:
            RuntimeError: If currently in ready state (invalid transition).
        """
        if self.state.is_ready:
            raise RuntimeError("Cannot transition to Pending state from Ready state.")
        self.update_state(PendingState())


# ────────────────────────────────────────────────
# Distributed execution agent
# ────────────────────────────────────────────────


class IcoAgent(
    Generic[I, O],
    IcoOperator[I, O],
    IcoRuntimeNode,
    ABC,
):
    """Distributed execution agent integrating computation flow with runtime tree.

    IcoAgent provides an abstraction for distributed computation
    that seamlessly integrates with both ICO computation flows and runtime trees.
    It manages remote execution through inter-process communication channels
    while maintaining unified runtime tree coordination.

    Architecture Integration:
        • Computation Flow: Implements IcoOperator[I, O] for data processing
        • Runtime Tree: Inherits IcoRuntimeNode for lifecycle management
        • Distributed Execution: Manages remote worker processes via channels
        • Type Safety: Generic parameterization ensures type consistency

    Distributed Execution Model:
        • Portal Pattern: Local agent acts as portal to remote computation
        • Worker Process: Remote IcoAgentWorker executes actual computation
        • Channel Communication: Bidirectional data and control flow
        • Runtime Coordination: Unified command/event propagation

    Agent Lifecycle:
        1. Creation: Agent created with flow_factory and channel configuration
        2. Activation: Worker process spawned, channel established
        3. Execution: Data items sent/received through channel portal
        4. Deactivation: Worker process terminated, resources cleaned up

    Runtime Tree Integration:
        • Worker Placeholder: IcoRemotePlaceholderNode represents remote worker
        • Command Propagation: Runtime commands forwarded to remote worker
        • Event Bubbling: Worker events bubbled through placeholder
        • State Coordination: Agent states synchronized with runtime tree

    Channel Communication:
        • Bidirectional Channels: Data flows in both directions
        • Command Forwarding: Runtime commands sent to worker
        • Event Bubbling: Worker events received and propagated
        • Error Handling: Fault events enable distributed error recovery

    State Management:
        • AgentStateModel: Extended states for communication phases
        • Waiting State: Agent waiting for worker response
        • Sending State: Agent transmitting data to worker
        • Fault Handling: Automatic fault state on communication errors

    Implementation Requirements:
        Subclasses must implement:
        • get_remote_runtime_factory(): Create remote worker instance
        • _activate_worker(): Setup worker process and communication
        • _deactivate_worker(): Cleanup worker process and resources

    Example Usage:
        • MPAgent: Multiprocessing implementation using multiprocessing channels
        • NetworkAgent: Distributed execution over network protocols (TBD)
        • ContainerAgent: Kubernetes/Docker container execution (TBD)

    Note:
        IcoAgent bridges local computation flows with distributed runtime
        execution, enabling transparent scalability while maintaining
        unified runtime tree coordination and type safety.
    """

    channel: IcoChannel[I, O] | None
    flow_factory: Callable[[], IcoOperator[I, O]]
    subtree_factory: Callable[[], IcoRuntimeNode]

    # Placeholder for worker in runtime tree (actual worker exists in separate process).
    _worker_placeholder: IcoRuntimeNode

    def __init__(
        self,
        *,
        channel: IcoChannel[I, O] | None = None,
        flow_factory: Callable[[], IcoOperator[I, O]],
        name: str | None = None,
        runtime_children: Sequence[IcoRuntimeNode] | None = None,
        state_model: BaseStateModel | None = None,
    ) -> None:
        """Initialize agent with dual inheritance and worker placeholder setup.

        Initializes both computation flow operator and runtime node infrastructure,
        establishing the foundation for distributed execution coordination.

        Args:
            channel: Communication channel to remote worker (may be created later).
            flow_factory: Factory function to create computation flow for remote execution.
            name: Optional name for both operator identity and runtime node.
            runtime_children: Additional runtime children beyond worker placeholder.
            state_model: Custom state model (defaults to AgentStateModel).

        Architecture Setup:
            • Dual initialization: Both IcoOperator and IcoRuntimeNode
            • Portal function: _portal_fn handles data exchange
            • Worker placeholder: Added to runtime children for tree coordination
            • Channel integration: Prepared for communication establishment
        """
        # Note: pylance cannot infer IcoOperator.__init__ from Generic inheritance, but mypy can.
        IcoOperator.__init__(  # pyright: ignore[reportUnknownMemberType]
            self, fn=self._portal_fn, name=name
        )

        IcoRuntimeNode.__init__(
            self,
            runtime_name=name,
            runtime_children=runtime_children,
            state_model=state_model or AgentStateModel(),
        )
        self.channel = channel
        self.flow_factory = flow_factory
        self._worker_placeholder = IcoRemotePlaceholderNode()
        self._runtime_children.append(self._worker_placeholder)

    # ──────── Worker management ────────

    def get_remote_flow_factory(self) -> Callable[[], IcoOperator[I, O]]:
        """Get factory function for creating remote computation flow.

        Returns:
            Factory function that creates the computation flow to be executed
            in the remote worker process.
        """
        return self.flow_factory

    @abstractmethod
    def get_remote_runtime_factory(self) -> Callable[[], IcoAgentWorker[I, O]]:
        """Create factory function for remote worker runtime node.

        Abstract method that subclasses must implement to specify how
        remote worker instances are created for distributed execution.

        Returns:
            Factory function that creates IcoAgentWorker instances
            configured with appropriate channels and computation flows.
        """
        ...

    def _activate_worker(self) -> None:
        """Activate remote worker process and establish communication.

        Abstract hook for subclasses to implement worker process creation,
        channel establishment, and communication initialization.

        Implementation Requirements:
            • Spawn or connect to remote worker process
            • Establish bidirectional communication channel
            • Configure worker with computation flow and runtime tree
        """
        pass

    def _deactivate_worker(self) -> None:
        """Deactivate remote worker process and cleanup resources.

        Abstract hook for subclasses to implement worker process termination,
        channel cleanup, and resource deallocation.

        Implementation Requirements:
            • Gracefully terminate remote worker process
            • Close communication channels and cleanup IPC resources
            • Release any allocated system resources
        """
        pass

    # ──────── Data flow function ────────

    def _portal_fn(self, input: I) -> O:
        """Portal function implementing data exchange with remote worker.

        Core function that handles the distributed execution by coordinating
        data transmission to remote worker and receiving computed results.
        Manages agent state transitions during the communication process.

        Args:
            input: Data item to send to remote worker for processing.

        Returns:
            Processed result received from remote worker.

        Communication Flow:
            1. Transition to SendingState and transmit input item
            2. Transition to WaitingState and wait for processed result
            3. Return to ReadyState upon successful completion
            4. Transition to FaultState on any communication error

        State Management:
            • sending(): During data transmission to worker
            • waiting(): While waiting for worker response
            • ready(): Upon successful completion
            • fault(): On any communication or processing error

        Raises:
            Exception: Re-raises any communication or worker processing errors
                     after transitioning to fault state.
        """
        assert self.channel is not None
        assert isinstance(self.state_model, AgentStateModel)

        try:
            # Send item to agent process
            self.state_model.sending()
            self.channel.send(input)

            # Wait for result from agent process
            self.state_model.waiting()
            output = self.channel.wait_for_item()

            self.state_model.ready()

            # MPProcess should always receive an output here
            assert output is not None
            return output

        except Exception:
            self.state_model.fault()
            raise

    # ──────── Channel message handling ────────

    def on_channel_command(self, command: IcoRuntimeCommand) -> None:
        """Handle commands received through communication channel.

        Commands should not be received by the agent from the worker.
        Worker-to-agent communication uses events, not commands.

        Raises:
            Exception: Always raises as channel commands are invalid for agents.
        """
        raise Exception("Channel commands should not be sent to the agent runtime.")

    def on_channel_event(self, event: IcoRuntimeEvent) -> None:
        """Handle events received from remote worker through channel.

        Processes events from the remote worker and integrates them into the
        local runtime tree by bubbling them through the worker placeholder.

        Args:
            event: Runtime event received from remote worker process.

        Event Integration:
            • Events bubble through worker placeholder for proper tree routing
            • Maintains event trace consistency across process boundaries
            • Enables distributed event propagation and monitoring
        """
        # Ensure events from the worker are bubbled correctly in the runtime tree
        self.bubble_event(event, from_child=self._worker_placeholder)

    # ─────── Runtime API ────────

    def on_command(self, command: IcoRuntimeCommand) -> None:
        """Handle runtime commands with distributed execution coordination.

        Orchestrates command processing between local agent and remote worker,
        managing worker lifecycle and ensuring proper command propagation
        across process boundaries.

        Args:
            command: Runtime command to process and potentially forward.

        Command Coordination:
            • Pre-order commands: Process locally then forward to worker
            • Post-order commands: Forward to worker then process locally
            • Activate: Spawn worker before forwarding command
            • Deactivate: Forward command then cleanup worker

        Worker Lifecycle Management:
            • Activation: Creates worker process and establishes channel
            • Command forwarding: All commands sent to remote worker
            • Deactivation: Graceful worker shutdown and resource cleanup
        """
        is_ready = self.state.is_ready
        if command.broadcast_order == "pre":
            super().on_command(command)

        match command:
            case IcoActivateCommand():
                assert command.broadcast_order == "pre"

                if not is_ready:
                    # Spawn an agent in pre-order, before sending a command downstream
                    # assert self.channel is None
                    self._activate_worker()

                assert self.channel is not None
                self.channel.send_command(command)

            case IcoDeactivateCommand():
                if self.state.is_ready or self.state.is_fault:
                    assert command.broadcast_order == "post"
                    assert self.channel is not None

                    self.channel.send_command(command)

                    # Deactivate worker in post-order, to ensure proper shutdown
                    self._deactivate_worker()

                    # Close channel queues
                    self.channel.close()
                    # self.channel = None

            case _:
                if is_ready:
                    assert self.channel is not None
                    self.channel.send_command(command)

        if command.broadcast_order == "post":
            super().on_command(command)

    @property
    def signature(self) -> IcoSignature:
        signature = super().signature

        if not signature.infered:
            signature_from_factory = infer_from_flow_factory(self.flow_factory)
            signature = signature_from_factory or IcoSignature(
                i=type(Any), c=None, o=type(Any), infered=False
            )

        return signature


# ────────────────────────────────────────────────
# Remote worker process runtime node
# ────────────────────────────────────────────────


class IcoAgentWorker(
    Generic[I, O],
    IcoRuntimeNode,
    ABC,
):
    """Remote worker process runtime node for distributed execution.

    IcoAgentWorker represents the remote execution component of an agent,
    running in a separate process and executing the actual computation flow.
    It coordinates with local IcoAgent through bidirectional channels.

    Worker Architecture:
        • Process Isolation: Runs in separate process for parallel execution
        • Computation Flow: Executes actual IcoOperator computation
        • Runtime Integration: Participates in distributed runtime tree
        • Channel Communication: Bidirectional data and control exchange

    Execution Model:
        • Event Loop: Continuous run_loop() processing input items
        • Blocking Receive: Waits for input items from local agent
        • Flow Processing: Executes computation through embedded flow
        • Result Transmission: Sends output back to local agent

    Runtime Tree Integration:
        • Remote Participation: Part of distributed runtime tree
        • Command Reception: Receives runtime commands from agent
        • Event Generation: Bubbles events back to agent
        • State Management: AgentWorkerStateModel with pending state

    Communication Coordination:
        • Channel Runtime Port: Worker serves as runtime endpoint
        • Command Broadcasting: Distributes commands to embedded flow
        • Event Forwarding: Sends events upstream to agent
        • Error Propagation: Fault events transmitted for recovery

    Lifecycle Management:
        1. Initialization: Worker created with flow and channel
        2. Pending State: Worker initializing but not yet active
        3. Event Loop: Continuous processing of input items
        4. Termination: Graceful shutdown on deactivate command

    Error Handling:
        • Exception Capture: Catches computation and communication errors
        • Fault Events: Generates IcoFaultEvent for error propagation
        • Error Recovery: Enables distributed fault tolerance
        • Loop Continuation: Recovers from individual item processing errors

    Note:
        IcoAgentWorker enables true distributed execution while maintaining
        unified runtime tree semantics and enabling fault-tolerant computing.
    """

    flow: IcoOperator[I, O]
    channel: IcoChannel[O, I]

    def __init__(
        self,
        *,
        channel: IcoChannel[O, I],
        flow_factory: Callable[[], IcoOperator[I, O]],
        runtime_parent: IcoRuntimeNode | None = None,
        runtime_children: Sequence[IcoRuntimeNode] | None = None,
        state_model: BaseStateModel | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize agent worker with computation flow and communication channel.

        Sets up the worker runtime node with embedded computation flow and
        establishes bidirectional communication with the parent agent process.

        Args:
            channel: Bidirectional communication channel (reversed direction from agent).
            flow_factory: Factory function to create the computation flow.
            runtime_parent: Parent runtime node (typically not used in separate process).
            runtime_children: Additional runtime children for worker's subtree.
            state_model: Custom state model (defaults to AgentWorkerStateModel).
            name: Optional name for worker identification.

        Worker Setup:
            • Runtime tree initialization with AgentWorkerStateModel
            • Computation flow creation and runtime integration
            • Channel runtime port configuration for command/event handling
            • Automatic discovery and connection of runtime nodes in flow
        """
        IcoRuntimeNode.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            runtime_name=name,
            runtime_parent=runtime_parent,
            runtime_children=runtime_children,
            state_model=state_model or AgentWorkerStateModel(),
        )
        self.flow = flow_factory()
        self.channel = channel

        # Connect to runtime port to enable command and event handling for remote runtime
        channel.runtime_port = self

        discover_and_connect_runtime_nodes(self, self.flow)

    def run_loop(self) -> None:
        """Main execution loop for remote worker process.

        Implements the core event loop for distributed execution, continuously
        processing input items received from the parent agent and returning
        computed results. Manages worker lifecycle and handles errors gracefully.

        Execution Flow:
            1. Pending State: Worker initializes and waits for activation
            2. Item Processing Loop:
                - Wait for input item from channel (blocking)
                - Process item through embedded computation flow
                - Send result back through channel
                - Repeat until termination signal
            3. Termination: Exit loop on deactivate command

        State Transitions:
            • pending(): Initial state during worker startup
            • running(): During computation flow execution
            • sending(): While transmitting results upstream
            • waiting(): Between items, ready for next input
            • idle(): Upon receiving termination signal
            • fault(): On any processing or communication error

        Error Handling:
            • Exception capture for each item processing
            • Fault event generation for error propagation
            • Loop continuation for recovery from individual item failures
            • Graceful degradation without worker termination

        Termination Conditions:
            • Deactivate command: Graceful shutdown requested
            • None item received: Channel closed by agent
            • Communication failure: Channel broken or process stopped

        Note:
            This method blocks indefinitely and should be run as the main
            function in the worker process. It provides the execution
            foundation for distributed ICO computation.
        """
        assert isinstance(self.state_model, AgentWorkerStateModel)

        # Agent is now pending activation
        self.state_model.pending()

        while True:
            try:
                # Blocks internally until new input arrives in the input channel.
                input_item = self.channel.wait_for_item()

                if input_item is None:
                    self.state_model.idle()
                    break  # Exit loop on deactivate command

                # Process input item through flow
                self.state_model.running()
                output = self.flow(input_item)

                # Send output item upstream
                self.state_model.sending()
                self.channel.send(output)

                # Wait for the next item
                self.state_model.waiting()

            except Exception as e:
                # Report runtime errors downstream to output channel and terminate
                self.state_model.fault()
                self.bubble_event(IcoFaultEvent.create(e))
                continue

    # ──────── Channel message handling ────────

    def on_channel_command(self, command: IcoRuntimeCommand) -> None:
        """Handle runtime commands received from parent agent through channel.

        Processes runtime commands received from the parent agent and broadcasts
        them to the embedded computation flow and runtime subtree.

        Args:
            command: Runtime command from parent agent to process locally.

        Command Processing:
            • Broadcast to subtree: Commands propagated to embedded flow
            • Local state management: Commands trigger worker state transitions
            • Distributed coordination: Enables unified runtime tree control
        """
        self.broadcast_command(command)

    def on_channel_event(self, event: IcoRuntimeEvent) -> None:
        """Handle events that should not be received by worker from agent.

        Agent-to-worker communication uses commands, not events.
        Events flow from worker to agent, not the reverse direction.

        Raises:
            Exception: Always raises as channel events are invalid for workers.
        """
        raise Exception(
            "Channel events should not be sent to the agent worker runtime."
        )

    # ─────── Runtime API ────────

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        """Handle runtime events and forward them to parent agent.

        Processes events generated within the worker's runtime subtree
        and forwards them to the parent agent through the communication channel.

        Args:
            event: Runtime event generated within worker's subtree.

        Returns:
            Event for continued local processing (same as input).

        Event Forwarding:
            • Channel transmission: Events sent to parent agent
            • Local processing: Standard event bubbling within worker
            • Distributed propagation: Events cross process boundaries
        """
        # Send event to upstream runtime
        self.channel.send_event(event)
        return super().on_event(event)

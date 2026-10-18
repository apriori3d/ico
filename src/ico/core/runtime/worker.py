from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Generic

from ico.core.operator import I, IcoOperator, IcoOperatorProtocol, O
from ico.core.probe import IcoRunnerProbe
from ico.core.runtime.channel.channel import IcoChannel
from ico.core.runtime.command import (
    IcoActivateCommand,
    IcoDeactivateCommand,
    IcoRuntimeCommand,
)
from ico.core.runtime.event import IcoFaultEvent, IcoRuntimeEvent
from ico.core.runtime.node import IcoRemotePlaceholderNode, IcoRuntimeNode
from ico.core.runtime.state import (
    BaseStateModel,
    IcoRuntimeState,
    ReadyState,
)
from ico.core.runtime.utils import discover_and_connect_runtime_nodes
from ico.core.signature import IcoSignature
from ico.core.signature_utils import infer_from_flow_factory

# ────────────────────────────────────────────────
# Worker-specific runtime states
# ────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class PendingState(IcoRuntimeState):
    """Runner state before activation completion.

    PendingState represents runners that are initializing but not yet
    ready for execution. Used during runner startup and resource allocation.
    """

    name: ClassVar[str] = "Pending"


@dataclass(slots=True, frozen=True)
class WaitingState(ReadyState):
    """Worker state waiting for input/output data exchange.

    WaitingState indicates the worker is ready and waiting for data items
    to arrive through the communication channel. Inherits ready status
    while indicating waiting behavior.
    """

    name: ClassVar[str] = "Waiting"


@dataclass(slots=True, frozen=True)
class SendingState(ReadyState):
    """Worker state actively sending data through communication channel.

    SendingState indicates the worker is transmitting data items through
    the inter-process communication channel. Maintains ready status
    while performing data transmission.
    """

    name: ClassVar[str] = "Sending"


# ────────────────────────────────────────────────
# Worker state management models
# ────────────────────────────────────────────────


class WorkerStateModel(BaseStateModel):
    """State model for runtime workers with communication states.

    WorkerStateModel extends base state management with specialized states
    for inter-process communication and coordination. Manages the additional
    states required for distributed execution scenarios.

    Worker-Specific States:
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


class RunnerStateModel(WorkerStateModel):
    """Extended state model for runner processes.

    RunnerStateModel adds pending state management for runners
    that initialize in separate processes and require additional startup
    coordination before becoming ready for execution.

    Runner-Specific States:
        • pending(): Transition to PendingState (initialization phase)
        • Inherits all WorkerStateModel communication states

    Usage:
        Used by IcoRunner instances running in separate processes
        to coordinate their initialization and execution lifecycle.
    """

    def pending(self) -> None:
        """Transition to pending state during runner initialization.

        Raises:
            RuntimeError: If currently in ready state (invalid transition).
        """
        if self.state.is_ready:
            raise RuntimeError("Cannot transition to Pending state from Ready state.")
        self.update_state(PendingState())


# ────────────────────────────────────────────────
# Distributed execution worker
# ────────────────────────────────────────────────


class IcoWorkerDied(RuntimeError):
    """The process of a worker's runner is gone (a crash, a kill, an exit).

    Raised by every call of the worker once its runner is dead, so that a
    crashed process ends the wait for its result instead of blocking it
    forever. ``IcoAsyncStream`` treats it as an item error (``on_error``)
    and stops giving items to that worker.

    Attributes:
        exitcode: Exit code of the process; negative for a signal (as in
            ``multiprocessing``), ``None`` when unknown.
    """

    def __init__(self, exitcode: int | None, name: str | None = None) -> None:
        self.exitcode = exitcode
        where = f"worker '{name}'" if name else "worker"
        super().__init__(f"{where} process {_exit_reason(exitcode)}")

    @property
    def signal(self) -> str | None:
        """Name of the signal that killed the process (e.g. ``SIGSEGV``)."""
        if self.exitcode is None or self.exitcode >= 0:
            return None
        return _signal_name(-self.exitcode)


def _signal_name(number: int) -> str:
    import signal

    try:
        return signal.Signals(number).name
    except ValueError:
        return f"signal {number}"


def _exit_reason(exitcode: int | None) -> str:
    if exitcode is None:
        return "is gone"
    if exitcode < 0:
        return f"died: {_signal_name(-exitcode)} (exit code {exitcode})"
    return f"exited with code {exitcode}"


class IcoWorker(
    Generic[I, O],
    IcoOperator[I, O],
    IcoRuntimeNode,
    ABC,
):
    """Distributed execution worker integrating computation flow with runtime tree.

    IcoWorker provides an abstraction for distributed computation
    that seamlessly integrates with both ICO computation flows and runtime trees.
    It manages remote execution through inter-process communication channels
    while maintaining unified runtime tree coordination.

    Architecture Integration:
        • Computation Flow: Implements IcoOperator[I, O] for data processing
        • Runtime Tree: Inherits IcoRuntimeNode for lifecycle management
        • Distributed Execution: Manages remote runner processes via channels
        • Type Safety: Generic parameterization ensures type consistency

    Distributed Execution Model:
        • Portal Pattern: Local worker acts as portal to remote computation
        • Runner Process: Remote IcoRunner executes actual computation
        • Channel Communication: Bidirectional data and control flow
        • Runtime Coordination: Unified command/event propagation

    Worker Lifecycle:
        1. Creation: Worker created with flow_factory and channel configuration
        2. Activation: Runner process spawned, channel established
        3. Execution: Data items sent/received through channel portal
        4. Deactivation: Runner process terminated, resources cleaned up

    Runtime Tree Integration:
        • Runner Placeholder: IcoRemotePlaceholderNode represents remote runner
        • Command Propagation: Runtime commands forwarded to remote runner
        • Event Bubbling: Runner events bubbled through placeholder
        • State Coordination: Worker states synchronized with runtime tree

    Channel Communication:
        • Bidirectional Channels: Data flows in both directions
        • Command Forwarding: Runtime commands sent to runner
        • Event Bubbling: Runner events received and propagated
        • Error Handling: Fault events enable distributed error recovery

    State Management:
        • WorkerStateModel: Extended states for communication phases
        • Waiting State: Worker waiting for runner response
        • Sending State: Worker transmitting data to runner
        • Fault Handling: Automatic fault state on communication errors

    Implementation Requirements:
        Subclasses must implement:
        • get_remote_runtime_factory(): Create remote runner instance
        • _activate_runner(): Setup runner process and communication
        • _deactivate_runner(): Cleanup runner process and resources

    Example Usage:
        • MPWorker: Multiprocessing implementation using multiprocessing channels
        • NetworkWorker: Distributed execution over network protocols (TBD)
        • ContainerWorker: Kubernetes/Docker container execution (TBD)

    Note:
        IcoWorker bridges local computation flows with distributed runtime
        execution, enabling transparent scalability while maintaining
        unified runtime tree coordination and type safety.
    """

    channel: IcoChannel[I, O] | None
    flow_factory: Callable[[], IcoOperatorProtocol[I, O]]
    runner_probe: IcoRunnerProbe | None = None
    """Hook for tools (profilers): when set, the runner runs
    ``runner_probe.wrap(flow)`` instead of the flow it built from
    ``flow_factory``; ``None`` normally (see :mod:`ico.core.probe`)."""
    subtree_factory: Callable[[], IcoRuntimeNode]

    # Placeholder for runner in runtime tree (actual runner exists in separate process).
    _runner_placeholder: IcoRuntimeNode

    def __init__(
        self,
        *,
        channel: IcoChannel[I, O] | None = None,
        flow_factory: Callable[[], IcoOperatorProtocol[I, O]],
        name: str | None = None,
        runtime_children: Sequence[IcoRuntimeNode] | None = None,
        state_model: BaseStateModel | None = None,
    ) -> None:
        """Initialize worker with dual inheritance and runner placeholder setup.

        Initializes both computation flow operator and runtime node infrastructure,
        establishing the foundation for distributed execution coordination.

        Args:
            channel: Communication channel to remote runner (may be created later).
            flow_factory: Factory function to create computation flow for remote execution.
            name: Optional name for both operator identity and runtime node.
            runtime_children: Additional runtime children beyond runner placeholder.
            state_model: Custom state model (defaults to WorkerStateModel).

        Architecture Setup:
            • Dual initialization: Both IcoOperator and IcoRuntimeNode
            • Portal function: _portal_fn handles data exchange
            • Runner placeholder: Added to runtime children for tree coordination
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
            state_model=state_model or WorkerStateModel(),
        )
        self.channel = channel
        self.flow_factory = flow_factory
        self._runner_placeholder = IcoRemotePlaceholderNode()
        self._runtime_children.append(self._runner_placeholder)

    # ──────── Runner management ────────

    def get_remote_flow_factory(self) -> Callable[[], IcoOperatorProtocol[I, O]]:
        """Get factory function for creating remote computation flow.

        Returns:
            Factory function that creates the computation flow to be executed
            in the remote runner process.
        """
        return self.flow_factory

    @abstractmethod
    def get_remote_runtime_factory(self) -> Callable[[], IcoRunner[I, O]]:
        """Create factory function for remote runner runtime node.

        Abstract method that subclasses must implement to specify how
        remote runner instances are created for distributed execution.

        Returns:
            Factory function that creates IcoRunner instances
            configured with appropriate channels and computation flows.
        """
        ...

    def _activate_runner(self) -> None:
        """Activate remote runner process and establish communication.

        Abstract hook for subclasses to implement runner process creation,
        channel establishment, and communication initialization.

        Implementation Requirements:
            • Spawn or connect to remote runner process
            • Establish bidirectional communication channel
            • Configure runner with computation flow and runtime tree
        """
        pass

    def _deactivate_runner(self) -> None:
        """Deactivate remote runner process and cleanup resources.

        Abstract hook for subclasses to implement runner process termination,
        channel cleanup, and resource deallocation.

        Implementation Requirements:
            • Gracefully terminate remote runner process
            • Close communication channels and cleanup IPC resources
            • Release any allocated system resources
        """
        pass

    def runner_alive(self) -> bool:
        """Whether the remote runner can still receive commands; subclasses
        with a runner process report its state."""
        return True

    # ──────── Data flow function ────────

    def _portal_fn(self, input: I) -> O:
        """Portal function implementing data exchange with remote runner.

        Core function that handles the distributed execution by coordinating
        data transmission to remote runner and receiving computed results.
        Manages worker state transitions during the communication process.

        Args:
            input: Data item to send to remote runner for processing.

        Returns:
            Processed result received from remote runner.

        Communication Flow:
            1. Transition to SendingState and transmit input item
            2. Transition to WaitingState and wait for processed result
            3. Return to ReadyState upon successful completion
            4. Transition to FaultState on any communication error

        State Management:
            • sending(): During data transmission to runner
            • waiting(): While waiting for runner response
            • ready(): Upon successful completion
            • fault(): On any communication or processing error

        Raises:
            Exception: Re-raises any communication or runner processing errors
                     after transitioning to fault state.
        """
        assert self.channel is not None
        assert isinstance(self.state_model, WorkerStateModel)

        try:
            # Send item to worker process
            self.state_model.sending()
            self.channel.send(input)

            # Wait for result from worker process
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

        Commands should not be received by the worker from the runner.
        Runner-to-worker communication uses events, not commands.

        Raises:
            Exception: Always raises as channel commands are invalid for workers.
        """
        raise Exception("Channel commands should not be sent to the worker runtime.")

    def on_channel_event(self, event: IcoRuntimeEvent) -> None:
        """Handle events received from remote runner through channel.

        Processes events from the remote runner and integrates them into the
        local runtime tree by bubbling them through the runner placeholder.

        Args:
            event: Runtime event received from remote runner process.

        Event Integration:
            • Events bubble through runner placeholder for proper tree routing
            • Maintains event trace consistency across process boundaries
            • Enables distributed event propagation and monitoring
        """
        # Ensure events from the runner are bubbled correctly in the runtime tree
        self.bubble_event(event, from_child=self._runner_placeholder)

    # ─────── Runtime API ────────

    def on_command(self, command: IcoRuntimeCommand) -> None:
        """Handle runtime commands with distributed execution coordination.

        Orchestrates command processing between local worker and remote runner,
        managing runner lifecycle and ensuring proper command propagation
        across process boundaries.

        Args:
            command: Runtime command to process and potentially forward.

        Command Coordination:
            • Pre-order commands: Process locally then forward to runner
            • Post-order commands: Forward to runner then process locally
            • Activate: Spawn runner before forwarding command
            • Deactivate: Forward command then cleanup runner

        Runner Lifecycle Management:
            • Activation: Creates runner process and establishes channel
            • Command forwarding: All commands sent to remote runner
            • Deactivation: Graceful runner shutdown and resource cleanup
        """
        is_ready = self.state.is_ready
        if command.broadcast_order == "pre":
            super().on_command(command)

        match command:
            case IcoActivateCommand():
                assert command.broadcast_order == "pre"

                if not is_ready:
                    # Spawn a worker in pre-order, before sending a command downstream
                    # assert self.channel is None
                    self._activate_runner()

                assert self.channel is not None
                self.channel.send_command(command)

            case IcoDeactivateCommand():
                if self.state.is_ready or self.state.is_fault:
                    assert command.broadcast_order == "post"
                    assert self.channel is not None

                    if self.runner_alive():  # a dead runner cannot acknowledge
                        self.channel.send_command(command)

                    # Deactivate runner in post-order, to ensure proper shutdown
                    self._deactivate_runner()

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
# Remote runner process runtime node
# ────────────────────────────────────────────────


class IcoRunner(
    Generic[I, O],
    IcoRuntimeNode,
    ABC,
):
    """Remote runner process runtime node for distributed execution.

    IcoRunner represents the remote execution component of a worker,
    running in a separate process and executing the actual computation flow.
    It coordinates with local IcoWorker through bidirectional channels.

    Runner Architecture:
        • Process Isolation: Runs in separate process for parallel execution
        • Computation Flow: Executes actual IcoOperator computation
        • Runtime Integration: Participates in distributed runtime tree
        • Channel Communication: Bidirectional data and control exchange

    Execution Model:
        • Event Loop: Continuous run_loop() processing input items
        • Blocking Receive: Waits for input items from local worker
        • Flow Processing: Executes computation through embedded flow
        • Result Transmission: Sends output back to local worker

    Runtime Tree Integration:
        • Remote Participation: Part of distributed runtime tree
        • Command Reception: Receives runtime commands from worker
        • Event Generation: Bubbles events back to worker
        • State Management: RunnerStateModel with pending state

    Communication Coordination:
        • Channel Runtime Port: Runner serves as runtime endpoint
        • Command Broadcasting: Distributes commands to embedded flow
        • Event Forwarding: Sends events upstream to worker
        • Error Propagation: Fault events transmitted for recovery

    Lifecycle Management:
        1. Initialization: Runner created with flow and channel
        2. Pending State: Runner initializing but not yet active
        3. Event Loop: Continuous processing of input items
        4. Termination: Graceful shutdown on deactivate command

    Error Handling:
        • Exception Capture: Catches computation and communication errors
        • Fault Events: Generates IcoFaultEvent for error propagation
        • Error Recovery: Enables distributed fault tolerance
        • Loop Continuation: Recovers from individual item processing errors

    Note:
        IcoRunner enables true distributed execution while maintaining
        unified runtime tree semantics and enabling fault-tolerant computing.
    """

    flow: IcoOperatorProtocol[I, O]
    channel: IcoChannel[O, I]

    def __init__(
        self,
        *,
        channel: IcoChannel[O, I],
        flow_factory: Callable[[], IcoOperatorProtocol[I, O]],
        runtime_parent: IcoRuntimeNode | None = None,
        runtime_children: Sequence[IcoRuntimeNode] | None = None,
        state_model: BaseStateModel | None = None,
        name: str | None = None,
        probe: IcoRunnerProbe | None = None,
    ) -> None:
        """Initialize runner with computation flow and communication channel.

        Sets up the runner runtime node with embedded computation flow and
        establishes bidirectional communication with the parent worker process.

        Args:
            channel: Bidirectional communication channel (reversed direction from worker).
            flow_factory: Factory function to create the computation flow.
            runtime_parent: Parent runtime node (typically not used in separate process).
            runtime_children: Additional runtime children for runner's subtree.
            state_model: Custom state model (defaults to RunnerStateModel).
            name: Optional name for runner identification.
            probe: The worker's ``runner_probe``: wraps the built flow.

        Runner Setup:
            • Runtime tree initialization with RunnerStateModel
            • Computation flow creation and runtime integration
            • Channel runtime port configuration for command/event handling
            • Automatic discovery and connection of runtime nodes in flow
        """
        IcoRuntimeNode.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            runtime_name=name,
            runtime_parent=runtime_parent,
            runtime_children=runtime_children,
            state_model=state_model or RunnerStateModel(),
        )
        flow = flow_factory()
        self.flow = flow if probe is None else probe.wrap(flow)
        self.channel = channel

        # Connect to runtime port to enable command and event handling for remote runtime
        channel.runtime_port = self

        discover_and_connect_runtime_nodes(self, self.flow)

    def run_loop(self) -> None:
        """Main execution loop for remote runner process.

        Implements the core event loop for distributed execution, continuously
        processing input items received from the parent worker and returning
        computed results. Manages runner lifecycle and handles errors gracefully.

        Execution Flow:
            1. Pending State: Runner initializes and waits for activation
            2. Item Processing Loop:
                - Wait for input item from channel (blocking)
                - Process item through embedded computation flow
                - Send result back through channel
                - Repeat until termination signal
            3. Termination: Exit loop on deactivate command

        State Transitions:
            • pending(): Initial state during runner startup
            • running(): During computation flow execution
            • sending(): While transmitting results upstream
            • waiting(): Between items, ready for next input
            • idle(): Upon receiving termination signal
            • fault(): On any processing or communication error

        Error Handling:
            • Exception capture for each item processing
            • Fault event generation for error propagation
            • Loop continuation for recovery from individual item failures
            • Graceful degradation without runner termination

        Termination Conditions:
            • Deactivate command: Graceful shutdown requested
            • None item received: Channel closed by worker
            • Communication failure: Channel broken or process stopped

        Note:
            This method blocks indefinitely and should be run as the main
            function in the runner process. It provides the execution
            foundation for distributed ICO computation.
        """
        assert isinstance(self.state_model, RunnerStateModel)

        # Worker is now pending activation
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
        """Handle runtime commands received from parent worker through channel.

        Processes runtime commands received from the parent worker and broadcasts
        them to the embedded computation flow and runtime subtree.

        Args:
            command: Runtime command from parent worker to process locally.

        Command Processing:
            • Broadcast to subtree: Commands propagated to embedded flow
            • Local state management: Commands trigger runner state transitions
            • Distributed coordination: Enables unified runtime tree control
        """
        self.broadcast_command(command)

    def on_channel_event(self, event: IcoRuntimeEvent) -> None:
        """Handle events that should not be received by runner from worker.

        Worker-to-runner communication uses commands, not events.
        Events flow from runner to worker, not the reverse direction.

        Raises:
            Exception: Always raises as channel events are invalid for runners.
        """
        raise Exception("Channel events should not be sent to the runner runtime.")

    # ─────── Runtime API ────────

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        """Handle runtime events and forward them to parent worker.

        Processes events generated within the runner's runtime subtree
        and forwards them to the parent worker through the communication channel.

        Args:
            event: Runtime event generated within runner's subtree.

        Returns:
            Event for continued local processing (same as input).

        Event Forwarding:
            • Channel transmission: Events sent to parent worker
            • Local processing: Standard event bubbling within runner
            • Distributed propagation: Events cross process boundaries
        """
        # Send event to upstream runtime
        self.channel.send_event(event)
        return super().on_event(event)

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, final

from ico.core.runtime.command import (
    IcoActivateCommand,
    IcoDeactivateCommand,
    IcoRuntimeCommand,
)
from ico.core.runtime.event import IcoRuntimeEvent
from ico.core.tree_utils import TreePathIndex


@dataclass(slots=True, frozen=True)
class IcoRuntimeState:
    """Base class for runtime execution lifecycle states.

    Runtime State Architecture:
        • States manage execution lifecycle: resource allocation → execution → cleanup
        • State transitions triggered by runtime commands (activate, run, deactivate)
        • State changes bubble up as events for monitoring and coordination

    Core State Lifecycle:
        1. Idle: No resources allocated, node inactive
        2. Ready: Resources allocated, prepared for execution
        3. Running: Active execution in progress
        4. Fault: Error occurred, requires intervention

    """

    name: ClassVar[str] = "base_state"

    @property
    def is_ready(self) -> bool:
        """Check if node is in ready state (prepared for execution).

        Returns:
            True if node has allocated resources and is ready to run.
        """
        return isinstance(self, ReadyState)

    @property
    def is_fault(self) -> bool:
        """Check if node is in fault state (error occurred).

        Returns:
            True if node encountered an error and requires intervention.
        """
        return isinstance(self, FaultState)

    def __str__(self) -> str:
        """String representation showing the state name."""
        return type(self).name


@dataclass(slots=True, frozen=True)
class IdleState(IcoRuntimeState):
    """Initial state - no resources allocated, node inactive.

    IdleState represents runtime nodes before activation or after deactivation.
    No resources are allocated and no execution can occur. This is the default
    state for newly created runtime nodes.
    """

    name: ClassVar[str] = "Idle"


@dataclass(slots=True, frozen=True)
class ReadyState(IcoRuntimeState):
    """Prepared state - resources allocated, ready for execution.

    ReadyState indicates that runtime node has successfully allocated all
    required resources and is prepared to begin execution when run command
    is received. This state enables execution coordination across runtime tree.
    """

    name: ClassVar[str] = "Ready"


@dataclass(slots=True, frozen=True)
class RunningState(ReadyState):
    """Active execution state - node is currently running.

    RunningState represents active execution phase where runtime node is
    performing its execution logic. Inherits from ReadyState to maintain
    ready status while indicating active execution.
    """

    name: ClassVar[str] = "Running"


@dataclass(slots=True, frozen=True)
class FaultState(IcoRuntimeState):
    """Error state - runtime fault occurred, intervention required.

    FaultState indicates that runtime node encountered an error during
    activation, execution, or deactivation. Node requires intervention
    to recover and cannot proceed with normal execution.
    """

    name: ClassVar[str] = "Fault"


StateTransitionMap = dict[type[IcoRuntimeCommand], type[IcoRuntimeState]]
"""Type alias for command-to-state transition mapping.

Maps runtime command types to their corresponding target state types,
enabling configurable state transition behavior for different runtime nodes.
Used by BaseStateModel and subclasses to define lifecycle transitions.
"""


class BaseStateModel:
    """State management model for runtime node execution lifecycle.

    BaseStateModel orchestrates state transitions in response to runtime commands,
    providing lifecycle management for execution infrastructure.

    State Transition Architecture:
        • Command-driven transitions: Commands trigger automatic state changes
        • Lifecycle enforcement: Validates legal state transitions (ready → running)
        • Error handling: Transitions to fault state on runtime errors
        • Event generation: State changes bubble as events for monitoring

    Default State Transitions:
        • IcoActivateCommand → ReadyState (allocate resources)
        • IcoDeactivateCommand → IdleState (release resources)
        • Manual transitions: ready() → ReadyState, running() → RunningState
        • Error transitions: fault() → FaultState (on any error)

    Lifecycle Flow:
        1. Creation: Node starts in IdleState
        2. Activation: activate() command → ReadyState (resources allocated)
        3. Execution: run() command → RunningState (execution begins)
        4. Completion: Node remains in RunningState until deactivation
        5. Deactivation: deactivate() command → IdleState (resources released)
        6. Error Recovery: fault() → FaultState (requires intervention)

    Subclass Customization:
        Subclasses can override transitions dict to customize state behavior
        for specialized runtime nodes (agents, tools, specific operators).
    """

    transitions: ClassVar[StateTransitionMap] = {
        IcoActivateCommand: ReadyState,
        IcoDeactivateCommand: IdleState,
    }

    __slots__ = ("_state",)

    _state: IcoRuntimeState

    def __init__(self) -> None:
        """Initialize state model with idle state (no resources allocated)."""
        self._state = IdleState()

    @property
    def state(self) -> IcoRuntimeState:
        """Current execution lifecycle state of the runtime node.

        Returns:
            Current state representing execution phase and resource status.
        """
        return self._state

    def update_state(self, new_state: IcoRuntimeState) -> None:
        """Update current state to new execution phase.

        Args:
            new_state: New runtime state for this node.

        Note:
            Called by transition methods to update internal state.
            State changes automatically bubble as events.
        """
        self._state = new_state

    def update(self, command: IcoRuntimeCommand) -> bool:
        """Process runtime command and trigger appropriate state transition.

        Automatically transitions state based on received runtime commands
        using the configured transitions mapping. Enables command-driven
        lifecycle management across the runtime tree.

        Args:
            command: Runtime command to process (activate, deactivate, etc.).

        Returns:
            True if state was updated based on command, False otherwise.

        Note:
            Subclasses can override transitions dict to customize which
            commands trigger which state changes for specialized nodes.
        """
        new_state: IcoRuntimeState | None = None

        for command_cls, state_cls in type(self).transitions.items():
            if isinstance(command, command_cls):
                new_state = state_cls()

        if new_state and new_state != self._state:
            self.update_state(new_state)
            return True

        return False

    def idle(self) -> None:
        """Transition to idle state (no resources allocated).

        Manually transitions node to idle state, typically used during
        cleanup or after deactivation. Resources should be released
        before calling this method.
        """
        self.update_state(IdleState())

    def ready(self) -> None:
        """Transition to ready state (prepared for execution).

        Manually transitions node to ready state, indicating that all
        required resources have been allocated and node is prepared
        for execution.

        Raises:
            RuntimeError: If current state is not already ready-compatible.
        """
        if not self.state.is_ready:
            raise RuntimeError("Cannot transition to Ready state from non-Ready state.")

        self.update_state(ReadyState())

    def running(self) -> None:
        """Transition to running state (active execution).

        Manually transitions node to running state, indicating that
        execution has begun. Node must be in ready state before
        transitioning to running.

        Raises:
            RuntimeError: If current state is not ready for execution.
        """
        if not self.state.is_ready:
            raise RuntimeError(
                "Cannot transition to Running state from non-Ready state."
            )
        self.update_state(RunningState())

    def fault(self) -> None:
        """Transition to fault state (error recovery required).

        Manually transitions node to fault state, indicating that
        a runtime error occurred and intervention is required.
        Can be called from any state.
        """
        self.update_state(FaultState())


# ────────────────────────────────────────────────
# State collection API
# ────────────────────────────────────────────────


@final
@dataclass(slots=True, frozen=True)
class IcoStateRequestCommand(IcoRuntimeCommand):
    """Runtime command requesting current state from nodes.

    IcoStateRequestCommand is broadcast through the runtime tree to collect
    current state information from all nodes. Each node responds by bubbling
    an IcoStateEvent containing its current execution state.

    State Collection Flow:
        1. Root broadcasts IcoStateRequestCommand down the tree
        2. Each node receives command and responds with IcoStateEvent
        3. State events bubble up carrying current execution states
        4. Monitoring tools collect aggregated state information

    Note:
        Enables runtime tree monitoring and debugging by providing
        comprehensive visibility into execution states across all nodes.
    """

    @staticmethod
    def create() -> IcoStateRequestCommand:
        """Create new state request command.

        Returns:
            Command instance for broadcasting state collection requests.
        """
        return IcoStateRequestCommand()


@final
@dataclass(slots=True, frozen=True)
class IcoStateEvent(IcoRuntimeEvent):
    """Runtime event carrying node state information.

    IcoStateEvent bubbles runtime state information up the tree in response
    to state requests or automatic state changes. Carries tree path trace
    for debugging and state aggregation.

    State Event Architecture:
        • Automatic: Generated when nodes change state during lifecycle
        • On-demand: Generated in response to IcoStateRequestCommand
        • Traceable: Includes tree path for node identification
        • Aggregatable: Enables runtime tree state monitoring

    Integration:
        • Runtime monitoring tools subscribe to state events
        • State events enable distributed execution coordination
        • Event traces support debugging and performance analysis
    """

    state: IcoRuntimeState

    @staticmethod
    def create(state: IcoRuntimeState) -> IcoStateEvent:
        """Create new state event with runtime state.

        Args:
            state: Current runtime state to include in event.

        Returns:
            Event instance for bubbling state information.
        """
        return IcoStateEvent(state=state, trace=TreePathIndex())

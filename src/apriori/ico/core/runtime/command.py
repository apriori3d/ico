from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, final

CommandBroadcastOrder = Literal["pre", "post"]
"""Broadcast order for runtime commands through the tree.

Defines the traversal order for command propagation:
- 'pre': Parent processes command before children (default)
- 'post': Children process command before parent (e.g., deactivation)
"""


@dataclass(slots=True, frozen=True)
class IcoRuntimeCommand:
    """Base class for runtime lifecycle control commands.

    IcoRuntimeCommand defines the interface for controlling execution phases
    across the runtime tree. Commands broadcast downward to coordinate
    resource allocation, execution timing, and cleanup across all nodes.

    Command Architecture:
        • Broadcast propagation: Commands flow from root to leaves
        • State transitions: Commands trigger automatic state changes
        • Resource coordination: Commands synchronize resource lifecycle
        • Execution control: Commands manage distributed execution timing

    Core Commands:
        • activate: Allocate resources and prepare for execution
        • run: Begin execution phase across runtime tree
        • deactivate: Release resources and cleanup

    Broadcast Order:
        Most commands use 'pre' order (parent → children) to ensure
        proper initialization. Deactivation uses 'post' order (children → parent)
        to ensure cleanup happens in reverse dependency order.
    """

    broadcast_order: ClassVar[CommandBroadcastOrder] = "pre"


@final
@dataclass(slots=True, frozen=True)
class IcoActivateCommand(IcoRuntimeCommand):
    """Command to activate runtime nodes and allocate resources.

    IcoActivateCommand broadcasts through the runtime tree to initiate
    resource allocation and prepare all nodes for execution. Triggers
    state transitions from Idle → Ready across the runtime tree.

    Activation Phase:
        • Resource allocation: Memory, connections, handles
        • Initialization: Setup computations, load models, prepare data
        • State transition: Idle → Ready for all nodes
        • Dependency preparation: Ensure all prerequisites are met
    """

    @staticmethod
    def create() -> IcoActivateCommand:
        """Create new activation command for broadcasting.

        Returns:
            Command instance for resource allocation phase.
        """
        return IcoActivateCommand()


@final
@dataclass(slots=True, frozen=True)
class IcoRunCommand(IcoRuntimeCommand):
    """Command to begin execution across runtime nodes.

    IcoRunCommand broadcasts through the runtime tree to initiate
    execution phase. Triggers state transitions from Ready → Running
    and begins actual computation execution.

    Execution Phase:
        • Computation start: Begin data processing, model inference
        • State transition: Ready → Running for all nodes
        • Execution coordination: Synchronize distributed execution
        • Progress tracking: Enable monitoring and progress reporting
    """

    @staticmethod
    def create() -> IcoRunCommand:
        """Create new run command for broadcasting.

        Returns:
            Command instance for execution initiation.
        """
        return IcoRunCommand()


@final
@dataclass(slots=True, frozen=True)
class IcoDeactivateCommand(IcoRuntimeCommand):
    """Command to deactivate runtime nodes and release resources.

    IcoDeactivateCommand broadcasts through the runtime tree to initiate
    cleanup and resource release. Uses 'post' broadcast order to ensure
    children are cleaned up before parents.

    Deactivation Phase:
        • Resource cleanup: Release memory, close connections, free handles
        • State transition: Running/Ready → Idle for all nodes
        • Dependency cleanup: Ensure proper teardown order
        • Result collection: Gather final outputs before cleanup

    Broadcast Order:
        Uses 'post' order (children first) to ensure proper cleanup
        sequence - dependencies are released before dependents.
    """

    broadcast_order: ClassVar[CommandBroadcastOrder] = "post"

    @staticmethod
    def create() -> IcoDeactivateCommand:
        """Create new deactivation command for broadcasting.

        Returns:
            Command instance for resource cleanup phase.
        """
        return IcoDeactivateCommand()

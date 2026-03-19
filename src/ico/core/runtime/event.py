from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import final

from ico.core.tree_utils import TreePathIndex

# ───── Event class ─────


@dataclass(slots=True, frozen=True)
class IcoRuntimeEvent:
    """Base class for runtime events that bubble up the execution tree.

    IcoRuntimeEvent provides the foundation for all runtime feedback that flows
    upward from child nodes to parents, enabling monitoring, coordination,
    and error handling across the distributed runtime tree.

    Event Architecture:
        • Bubble propagation: Events flow from leaves to root
        • Tree tracing: Events carry path traces for debugging and routing
        • Event filtering: Nodes can handle and stop event propagation
        • Monitoring integration: Events enable runtime tree observation

    Event Types:
        • State events: Node lifecycle changes (idle, ready, running, fault)
        • Heartbeat events: Periodic health and status updates
        • Fault events: Error conditions requiring intervention
        • Progress events: Execution progress and completion status

    Tree Path Tracing:
        Each event carries a TreePathIndex that tracks its origin path
        through the runtime tree, enabling precise event routing and debugging.
    """

    trace: TreePathIndex


@final
@dataclass(slots=True, frozen=True)
class IcoHeartbeatEvent(IcoRuntimeEvent):
    """Periodic health check event for runtime tree monitoring.

    IcoHeartbeatEvent provides regular status updates from runtime nodes,
    enabling monitoring tools to verify that nodes are alive and responsive.
    Used for distributed system health monitoring and failure detection.

    Heartbeat Architecture:
        • Periodic generation: Nodes emit heartbeats at regular intervals
        • Health monitoring: Confirms node responsiveness and availability
        • Failure detection: Missing heartbeats indicate node problems
        • Distributed coordination: Enables cluster health awareness

    Usage:
        Monitoring tools subscribe to heartbeat events to track runtime
        tree health and detect failed or unresponsive nodes for recovery.
    """

    @staticmethod
    def create() -> IcoHeartbeatEvent:
        """Create new heartbeat event for status monitoring.

        Returns:
            Heartbeat event with empty trace (to be populated during bubbling).
        """
        return IcoHeartbeatEvent(trace=TreePathIndex())


@final
@dataclass(slots=True, frozen=True)
class IcoFaultEvent(IcoRuntimeEvent):
    """Error event carrying exception details for fault handling.

    IcoFaultEvent transports error information from failed runtime nodes
    up the tree, enabling centralized error handling, logging, and recovery.
    Includes comprehensive exception metadata for debugging and analysis.

    Fault Event Architecture:
        • Exception capture: Automatically extracts error details
        • Stack trace preservation: Maintains full debugging information
        • Error classification: Includes exception type and module information
        • Recovery coordination: Enables parent nodes to handle child failures

    Error Information:
        • Exception type and module for error classification
        • Error message and string representation
        • Full stack trace for debugging
        • Tree path trace for error source identification

    Fault Handling:
        Parent nodes receive fault events and can implement recovery
        strategies, error logging, or escalation to higher-level handlers.
    """

    info: dict[str, object]

    @staticmethod
    def create(e: Exception) -> IcoFaultEvent:
        """Create fault event from exception with comprehensive error details.

        Args:
            e: Exception that occurred in runtime execution.

        Returns:
            Fault event containing extracted exception metadata and trace.
        """
        # Extract exception metadata
        exc_type = type(e)
        tb = e.__traceback__

        info: dict[str, object] = {
            "type": f"{exc_type.__module__}.{exc_type.__name__}",
            "message": str(e),
            "repr": repr(e),
            "traceback": traceback.format_exception(exc_type, e, tb),
        }

        return IcoFaultEvent(info=info, trace=TreePathIndex())

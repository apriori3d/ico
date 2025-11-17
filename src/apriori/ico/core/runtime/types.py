from __future__ import annotations

from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

from typing_extensions import Self

from apriori.ico.core.runtime.progress.types import ProgressProtocol
from apriori.ico.core.types import IcoOperatorProtocol

# ──── Runtime Commands ────


class IcoRuntimeCommandType(Enum):
    """
    Runtime commands controlling activation and resource lifecycle
    across agents and runtime contours.

    Commands:
        • activate   - Allocate resources and prepare for execution
        • reset      - Reinitialize internal state (weights, cache, buffers)
        • deactivate - Release resources and mark as inactive
        • pause      - Temporarily suspend execution without teardown
        • resume     - Resume execution after pause
        • stop       - Stop signal for iterative or streaming operators
    """

    activate = auto()
    reset = auto()
    deactivate = auto()
    pause = auto()
    resume = auto()
    stop = auto()


# ──── State for runtime operators ────


class IcoRuntimeStateType(Enum):
    """
    Current runtime state of an agent and connected contour.

    States:
        • inactive - Operator uninitialized or fully released
        • ready    - Initialized and ready to run
        • running  - Currently executing or active
        • paused   - Temporarily suspended, resources preserved
        • error    - Faulted state after unrecoverable failure
    """

    inactive = auto()
    ready = auto()
    running = auto()
    paused = auto()
    error = auto()


# ──── Event types for runtime signaling ────


class IcoRuntimeEventType(Enum):
    fault = auto()
    heartbeat = auto()


class IcoRuntimeEventProtocol(Protocol):
    type: IcoRuntimeEventType
    meta: dict[Any, Any]


# ──── Protocols for runtime operators ────


class IcoRuntimeStateProtocol(Protocol):
    # ─── Properties ───

    @property
    def state(self) -> IcoRuntimeStateType:
        """Current runtime state."""
        ...

    @property
    def last_event(self) -> IcoRuntimeEventProtocol | None:
        """Last received runtime event."""


class IcoRuntimeFlowProtocol(Protocol):
    """Operator responsible for pushing runtime command downstream and runtime events upstream."""

    def on_command(self, command: IcoRuntimeCommandType) -> None: ...

    def on_event(self, event: IcoRuntimeEventProtocol) -> None: ...


@runtime_checkable
class IcoRuntimeTreeProtocol(Protocol):
    runtime_children: list[IcoRuntimeTreeProtocol]

    @property
    def runtime_parent(self) -> IcoRuntimeTreeProtocol | None: ...

    @runtime_parent.setter
    def runtime_parent(self, value: IcoRuntimeTreeProtocol | None) -> None: ...

    # ─── Runtime Connection ───

    def connect_runtime(self, runtime: IcoRuntimeTreeProtocol) -> None: ...

    def disconnect_runtime(self, runtime: IcoRuntimeTreeProtocol) -> None: ...

    # ─── Command & Event Propagation ───

    def broadcast_command(self, command: IcoRuntimeCommandType) -> None: ...

    def bubble_event(self, event: IcoRuntimeEventProtocol) -> None: ...

    # ─── Progress ───

    def attach_progress(self, progress: ProgressProtocol) -> Self: ...


@runtime_checkable
class IcoRuntimeOperatorProtocol(
    IcoRuntimeStateProtocol,
    IcoRuntimeTreeProtocol,
    IcoRuntimeFlowProtocol,
    IcoOperatorProtocol[None, None],
    Protocol,
): ...


@runtime_checkable
class ConnectedToIcoRuntime(Protocol):
    runtime: IcoRuntimeOperatorProtocol | None
    """Get the associated runtime protocol."""

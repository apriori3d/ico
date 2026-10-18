from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from ico.core.runtime.command import IcoRuntimeCommand
from ico.core.runtime.event import IcoRuntimeEvent

# ──────────────────────────────────────────────────────────────
# Channel Message System for Inter-Process Communication
# ──────────────────────────────────────────────────────────────

# Type variable for payloads
P = TypeVar("P", covariant=True)
"""Covariant type variable for message payload types.

Enables type-safe message passing across process boundaries while maintaining
variance relationships for generic payload types.
"""


@dataclass(slots=True, frozen=True)
class ChannelMessage:
    """Base class for all inter-process communication messages.

    ChannelMessage provides the foundation for distributed communication
    between worker processes and runner processes. Every message carries
    a unique identifier for tracking, acknowledgment, and error handling.

    Message Architecture:
        • Unified envelope for all IPC communication
        • Message ID tracking for reliable delivery
        • Hierarchical message types for different communication patterns
        • Serialization-friendly immutable design

    Message Flow:
        Worker ↔️ Runner communication through bidirectional channels with
        typed message envelopes ensuring reliable distributed coordination.

    Usage:
        Base class for all message types - not instantiated directly.
        Subclasses implement specific communication patterns.
    """

    id: int


@dataclass(slots=True, frozen=True)
class DataMessage(Generic[P], ChannelMessage):
    """Message carrying computation data payloads between processes.

    DataMessage transports actual computation data (input/output items)
    between worker and runner processes. Generic parameterization ensures
    type safety across process boundaries.

    Data Flow Pattern:
        • Worker → Runner: Input data items for remote processing
        • Runner → Worker: Computed output data items
        • Type preservation: Generic payload maintains type information
        • Serialization: Payload must be serializable for IPC transport

    Generic Parameters:
        P: Type of the payload data being transmitted

    Usage:
        Primary mechanism for computation data exchange in distributed
        ICO execution across worker/runner process boundaries.
    """

    payload: P


@dataclass(slots=True, frozen=True)
class RuntimeMessage(ChannelMessage):
    """Base class for runtime control messages in distributed execution.

    RuntimeMessage represents non-data communication for coordinating
    runtime tree operations across process boundaries. Enables unified
    runtime management in distributed ICO execution.

    Runtime Control Categories:
        • Commands: Lifecycle control (activate, run, deactivate)
        • Events: Status feedback (state changes, errors, heartbeats)
        • Acknowledgments: Message delivery confirmation

    Distributed Runtime Integration:
        Maintains runtime tree semantics across process boundaries by
        transmitting runtime coordination messages through channels.
    """

    pass


@dataclass(slots=True, frozen=True)
class CommandMessage(RuntimeMessage):
    """Message carrying runtime commands to remote processes.

    CommandMessage enables distributed runtime tree coordination by
    transmitting lifecycle commands from worker to runner processes.

    Command Distribution:
        • Worker broadcasts commands to local runtime tree
        • CommandMessage forwards commands to remote runner process
        • Runner receives and broadcasts commands to remote runtime subtree
        • Maintains unified command propagation across process boundaries

    Supported Commands:
        • IcoActivateCommand: Resource allocation and runner preparation
        • IcoRunCommand: Execution initiation across distributed nodes
        • IcoDeactivateCommand: Graceful shutdown and cleanup

    Usage:
        Integral part of distributed runtime tree lifecycle management,
        ensuring coordinated execution phases across all processes.
    """

    command: IcoRuntimeCommand


@dataclass(slots=True, frozen=True)
class EventMessage(RuntimeMessage):
    """Message carrying runtime events from remote processes.

    EventMessage enables distributed event bubbling by transmitting
    runtime feedback from runner processes back to worker processes.

    Event Propagation:
        • Runner runtime tree generates events (state changes, errors)
        • EventMessage transmits events to worker process
        • Worker receives and bubbles events through local runtime tree
        • Maintains unified event propagation across process boundaries

    Event Types:
        • IcoStateEvent: Runner state changes and lifecycle updates
        • IcoFaultEvent: Error conditions requiring worker intervention
        • IcoHeartbeatEvent: Runner health and liveness indicators

    Usage:
        Critical for distributed error handling, monitoring, and runtime
        tree coordination across worker/runner process boundaries.
    """

    event: IcoRuntimeEvent
    ack: bool = True
    """Whether the sender waits for an acknowledgment; telemetry (progress,
    profiles, answers to queries) is sent without one."""


@dataclass(slots=True, frozen=True)
class AcknowledgeMessage(RuntimeMessage):
    """Message confirming receipt of other messages for reliable delivery.

    AcknowledgeMessage provides delivery confirmation in distributed
    communication, enabling reliable message passing and error detection
    in worker/runner coordination.

    Acknowledgment Protocol:
        • Message sender transmits message with unique ID
        • Message receiver sends AcknowledgeMessage with matching ID
        • Sender confirms successful delivery and can proceed
        • Missing acknowledgments indicate communication failures

    Reliability Features:
        • Message delivery confirmation for critical communications
        • Timeout detection for failed or lost messages
        • Retry mechanisms for unreliable network conditions
        • Error propagation for permanent communication failures

    Usage:
        Optional reliability layer for critical runtime messages where
        delivery confirmation is required for distributed coordination.
    """

    message_id: int

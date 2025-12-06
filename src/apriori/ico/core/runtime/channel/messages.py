from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from apriori.ico.core.runtime.command import IcoRuntimeCommand
from apriori.ico.core.runtime.event import IcoRuntimeEvent

# ──────────────────────────────────────────────────────────────
# Channel Message class
# ──────────────────────────────────────────────────────────────

# Type variable for payloads
P = TypeVar("P", covariant=True)


@dataclass(slots=True, frozen=True)
class ChannelMessage:
    """Unified envelope exchanged between runtime endpoints."""

    id: int


@dataclass(slots=True, frozen=True)
class DataMessage(Generic[P], ChannelMessage):
    payload: P


@dataclass(slots=True, frozen=True)
class RuntimeMessage(ChannelMessage):
    pass


@dataclass(slots=True, frozen=True)
class CommandMessage(RuntimeMessage):
    command: IcoRuntimeCommand


@dataclass(slots=True, frozen=True)
class EventMessage(RuntimeMessage):
    event: IcoRuntimeEvent


@dataclass(slots=True, frozen=True)
class AcknowledgeMessage(RuntimeMessage):
    message_id: int


@dataclass(slots=True, frozen=True)
class CommandAcknowledgeMessage(AcknowledgeMessage):
    command: IcoRuntimeCommand

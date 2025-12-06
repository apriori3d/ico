from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar

from apriori.ico.core.runtime.command import IcoRuntimeCommand
from apriori.ico.core.runtime.event import IcoRuntimeEvent

# ──────────────────────────────────────────────────────────────
# Channel Message class
# ──────────────────────────────────────────────────────────────

# Type variable for payloads
P = TypeVar("P", covariant=True)


@dataclass(slots=True, frozen=True)
class ChannelMessage(Generic[P]):
    """Unified envelope exchanged between runtime endpoints."""

    id: int
    payload: P


@dataclass(slots=True, frozen=True)
class DataMessage(Generic[P], ChannelMessage[P]):
    pass


@dataclass(slots=True, frozen=True)
class CommandMessage(ChannelMessage[IcoRuntimeCommand]):
    pass


@dataclass(slots=True, frozen=True)
class EventMessage(ChannelMessage[IcoRuntimeEvent]):
    pass


@dataclass(slots=True, frozen=True)
class AcknowledgeMessage(ChannelMessage[int]):
    pass


@dataclass(slots=True, frozen=True)
class CommandAcknowledgeMessage(AcknowledgeMessage):
    command: IcoRuntimeCommand


# Non-generic message types
RuntimeMessageTypes: TypeAlias = CommandMessage | EventMessage
SystemMessageTypes: TypeAlias = RuntimeMessageTypes | AcknowledgeMessage

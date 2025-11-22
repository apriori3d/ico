from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic, TypeVar

from apriori.ico.core.runtime.command import IcoRuntimeCommand
from apriori.ico.core.runtime.event import IcoRuntimeEvent

# ──────────────────────────────────────────────────────────────
# Message categories
# ──────────────────────────────────────────────────────────────


class MessageType(Enum):
    """Categories of messages exchanged between runtime endpoints.
    Used to acknowladge receipt of specific message types."""

    input = auto()
    runtime_command = auto()
    runtime_event = auto()
    acknowledge = auto()


# ──────────────────────────────────────────────────────────────
# Channel Message class
# ──────────────────────────────────────────────────────────────

# Type variable for payloads
P = TypeVar("P", covariant=True)


@dataclass(slots=True)
class ChannelMessage(Generic[P]):
    """Unified envelope exchanged between runtime endpoints."""

    __message_type__: MessageType
    payload: P

    def __init__(self, payload: P) -> None:
        self.payload = payload

    @property
    def message_type(self) -> MessageType:
        return self.__message_type__


class InputChannelMessage(ChannelMessage[P]):
    __message_type__: MessageType = MessageType.input


class CommandChannelMessage(ChannelMessage[IcoRuntimeCommand]):
    __message_type__: MessageType = MessageType.runtime_command


class EventChannelMessage(ChannelMessage[IcoRuntimeEvent]):
    __message_type__: MessageType = MessageType.runtime_event


class AcknowledgeChannelMessage(ChannelMessage[MessageType]):
    __message_type__: MessageType = MessageType.acknowledge


def wrap_payload(
    payload: P | IcoRuntimeCommand | IcoRuntimeEvent,
) -> ChannelMessage[P | IcoRuntimeCommand | IcoRuntimeEvent]:
    if isinstance(payload, IcoRuntimeCommand):
        return CommandChannelMessage(payload)
    elif isinstance(payload, IcoRuntimeEvent):
        return EventChannelMessage(payload)
    else:
        return InputChannelMessage(payload)

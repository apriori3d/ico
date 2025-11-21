from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, ClassVar

from apriori.ico.core.runtime.command import IcoRuntimeCommand
from apriori.ico.core.runtime.event import IcoRuntimeEvent

# ──────────────────────────────────────────────────────────────
# Message categories
# ──────────────────────────────────────────────────────────────


class ChannelMessageType(Enum):
    """Categories of messages exchanged between runtime endpoints."""

    input = auto()
    runtime_command = auto()
    runtime_event = auto()
    acknowledge = auto()
    # error = auto()
    # system = auto()


# ──────────────────────────────────────────────────────────────
# Decorator for assigning message types
# ──────────────────────────────────────────────────────────────


def message(t: ChannelMessageType) -> Callable[[type[Any]], type[Any]]:
    """Decorator that assigns a ChannelMessageType to a payload class."""

    def decorator(cls: type[Any]) -> type[Any]:
        cls.__message_type__ = t
        return cls

    return decorator


# ──────────────────────────────────────────────────────────────
# Base payload
# ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ChannelMessagePayload:
    """Base class for all message payloads."""

    __message_type__: ClassVar[ChannelMessageType]

    @property
    def message_type(self) -> ChannelMessageType:
        return self.__message_type__

    def wrap(self) -> ChannelMessage:
        """Wrap this payload instance into a ChannelMessage."""
        return ChannelMessage(self.message_type, self)


# ──────────────────────────────────────────────────────────────
# Concrete payloads
# ──────────────────────────────────────────────────────────────


@message(ChannelMessageType.input)
@dataclass(slots=True)
class InputPayload(ChannelMessagePayload):
    """Data payload carrying one input item."""

    input: Any


@message(ChannelMessageType.runtime_command)
@dataclass(slots=True)
class RuntimeCommandPayload(ChannelMessagePayload):
    """Payload carrying a runtime command (activate/reset/stop)."""

    command: IcoRuntimeCommand


@message(ChannelMessageType.runtime_event)
@dataclass(slots=True)
class RuntimeEventPayload(ChannelMessagePayload):
    """Payload carrying runtime events (faults, progress, etc.)."""

    event: IcoRuntimeEvent


@message(ChannelMessageType.acknowledge)
@dataclass(slots=True)
class AcknowledgePayload(ChannelMessagePayload):
    """Acknowledgment of a received message."""

    ack_message_type: ChannelMessageType


# ──────────────────────────────────────────────────────────────
# Message wrapper
# ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ChannelMessage:
    """Unified envelope exchanged between runtime endpoints."""

    message_type: ChannelMessageType
    payload: ChannelMessagePayload

    def unwrap(self) -> ChannelMessagePayload:
        """Safely access payload of the expected type."""
        if self.message_type != self.payload.message_type:
            raise TypeError(
                f"Expected {self.payload.message_type}, got {self.message_type}"
            )
        return self.payload

from __future__ import annotations

import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple, TypeAlias, final

# ──── Event types for runtime signaling ────


class IcoRuntimeEventType(Enum):
    fault = auto()
    heartbeat = auto()
    tool = auto()
    print = auto()
    log = auto()
    progress = auto()


# ───── Define payload types ─────


class PrintEventMeta(NamedTuple):
    message: str


class ProgressEventMeta(NamedTuple):
    total: int
    advance: int


# Union of all possible meta payloads
RuntimeEventMeta: TypeAlias = PrintEventMeta | ProgressEventMeta | Mapping[str, object]


# ───── Event class ─────


@dataclass(slots=True, frozen=True)
class IcoRuntimeEvent:
    pass


@final
@dataclass(slots=True, frozen=True)
class IcoHearbeatEvent(IcoRuntimeEvent):
    pass


@final
@dataclass(slots=True, frozen=True)
class IcoFaultEvent(IcoRuntimeEvent):
    info: dict[str, object]

    @staticmethod
    def exception(e: Exception) -> IcoFaultEvent:
        # Extract exception metadata
        exc_type = type(e)
        tb = e.__traceback__

        info: dict[str, object] = {
            "type": f"{exc_type.__module__}.{exc_type.__name__}",
            "message": str(e),
            "repr": repr(e),
            "traceback": traceback.format_exception(exc_type, e, tb),
        }

        return IcoFaultEvent(info=info)

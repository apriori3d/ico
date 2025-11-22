from __future__ import annotations

import traceback
from collections.abc import Mapping
from enum import Enum, auto

from apriori.ico.core.runtime.exceptions import IcoRuntimeError

# ──── Event types for runtime signaling ────


class IcoRuntimeEventType(Enum):
    fault = auto()
    heartbeat = auto()


class IcoRuntimeEvent:
    __slots__ = ("type", "meta")

    type: IcoRuntimeEventType
    meta: Mapping[str, object]

    def __init__(
        self,
        type: IcoRuntimeEventType,
        meta: Mapping[str, object] | None = None,
    ) -> None:
        self.type = type
        self.meta = meta or {}

    @staticmethod
    def heartbeat() -> IcoRuntimeEvent:
        return IcoRuntimeEvent(type=IcoRuntimeEventType.heartbeat)

    @staticmethod
    def exception(e: Exception) -> IcoRuntimeEvent:
        # Extract exception metadata
        exc_type = type(e)
        tb = e.__traceback__

        meta = {
            "type": f"{exc_type.__module__}.{exc_type.__name__}",
            "message": str(e),
            "repr": repr(e),
            "traceback": traceback.format_exception(exc_type, e, tb),
        }

        return IcoRuntimeEvent(type=IcoRuntimeEventType.fault, meta=meta)

    @property
    def is_fault(self) -> bool:
        return self.type == IcoRuntimeEventType.fault

    def raise_if_fault(self) -> None:
        if self.type == IcoRuntimeEventType.fault:
            msg = self.meta.get("message", "Unknown fault")
            raise IcoRuntimeError(f"IcoRuntimeEvent fault: {msg}")

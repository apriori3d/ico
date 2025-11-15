from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any

from apriori.ico.core.runtime.exceptions import IcoRuntimeError
from apriori.ico.core.runtime.types import IcoRuntimeEventType


@dataclass(slots=True)
class IcoRuntimeEvent:
    type: IcoRuntimeEventType
    meta: dict[Any, Any] = field(default_factory=dict)

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

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import final

from apriori.ico.core.tree_utils import TreePathIndex

# ───── Event class ─────


@dataclass(slots=True, frozen=True)
class IcoRuntimeEvent:
    trace: TreePathIndex


@final
@dataclass(slots=True, frozen=True)
class IcoHeartbeatEvent(IcoRuntimeEvent):
    @staticmethod
    def create() -> IcoHeartbeatEvent:
        return IcoHeartbeatEvent(trace=TreePathIndex())


@final
@dataclass(slots=True, frozen=True)
class IcoFaultEvent(IcoRuntimeEvent):
    info: dict[str, object]

    @staticmethod
    def create(e: Exception) -> IcoFaultEvent:
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

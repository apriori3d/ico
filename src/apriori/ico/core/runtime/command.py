from __future__ import annotations

from enum import Enum, auto


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


class IcoRuntimeCommand:
    __slots__ = ("type", "meta")

    type: IcoRuntimeCommandType
    meta: dict[str, str]

    def __init__(
        self,
        type: IcoRuntimeCommandType,
        meta: dict[str, str] | None = None,
    ) -> None:
        self.type = type
        self.meta = meta or {}

    @staticmethod
    def activate() -> IcoRuntimeCommand:
        return IcoRuntimeCommand(type=IcoRuntimeCommandType.activate)

    @staticmethod
    def deactivate() -> IcoRuntimeCommand:
        return IcoRuntimeCommand(type=IcoRuntimeCommandType.deactivate)

    @staticmethod
    def pause() -> IcoRuntimeCommand:
        return IcoRuntimeCommand(type=IcoRuntimeCommandType.pause)

    @staticmethod
    def resume() -> IcoRuntimeCommand:
        return IcoRuntimeCommand(type=IcoRuntimeCommandType.resume)

    @staticmethod
    def reset() -> IcoRuntimeCommand:
        return IcoRuntimeCommand(type=IcoRuntimeCommandType.reset)

    @staticmethod
    def stop() -> IcoRuntimeCommand:
        return IcoRuntimeCommand(type=IcoRuntimeCommandType.stop)

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, final

# class IcoRuntimeCommandType(Enum):
#     activate = auto()
#     run = auto()
#     reset = auto()
#     deactivate = auto()
#     pause = auto()
#     resume = auto()
#     stop = auto()


CommandBroadcastOrder = Literal["pre", "post"]


@dataclass(slots=True, frozen=True)
class IcoRuntimeCommand:
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

    broadcast_order: ClassVar[CommandBroadcastOrder] = "pre"


@final
@dataclass(slots=True, frozen=True)
class IcoActivateCommand(IcoRuntimeCommand):
    @staticmethod
    def create() -> IcoActivateCommand:
        return IcoActivateCommand()


@final
@dataclass(slots=True, frozen=True)
class IcoRunCommand(IcoRuntimeCommand):
    @staticmethod
    def create() -> IcoActivateCommand:
        return IcoActivateCommand()


@final
@dataclass(slots=True, frozen=True)
class IcoDeactivateCommand(IcoRuntimeCommand):
    broadcast_order: ClassVar[CommandBroadcastOrder] = "post"

    @staticmethod
    def create() -> IcoDeactivateCommand:
        return IcoDeactivateCommand()

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal, final


class IcoRuntimeCommandType(Enum):
    activate = auto()
    run = auto()
    reset = auto()
    deactivate = auto()
    pause = auto()
    resume = auto()
    stop = auto()


CommandBroadcastOrder = Literal["Pre-order", "Post-order"]


@dataclass(slots=True)
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

    broadcast_order: CommandBroadcastOrder = field(init=False)

    def __post_init__(self):
        self.broadcast_order = "Pre-order"


@final
@dataclass(slots=True)
class IcoActivateCommand(IcoRuntimeCommand):
    pass


@final
@dataclass(slots=True)
class IcoRunCommand(IcoRuntimeCommand):
    pass


@final
@dataclass(slots=True)
class IcoDeactivateCommand(IcoRuntimeCommand):
    def __post_init__(self):
        self.broadcast_order = "Post-order"


@final
@dataclass(slots=True)
class IcoPauseCommand(IcoRuntimeCommand):
    pass


@final
@dataclass(slots=True)
class IcoResumeCommand(IcoRuntimeCommand):
    pass


@final
@dataclass(slots=True)
class IcoResetCommand(IcoRuntimeCommand):
    pass


@final
@dataclass(slots=True)
class IcoStopCommand(IcoRuntimeCommand):
    pass

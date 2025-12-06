from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import final


class IcoRuntimeCommandType(Enum):
    activate = auto()
    run = auto()
    reset = auto()
    deactivate = auto()
    pause = auto()
    resume = auto()
    stop = auto()


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

    pass


@final
@dataclass(slots=True, frozen=True)
class IcoActivateCommand(IcoRuntimeCommand):
    pass


@final
@dataclass(slots=True, frozen=True)
class IcoRunCommand(IcoRuntimeCommand):
    pass


@final
@dataclass(slots=True, frozen=True)
class IcoDeactivateCommand(IcoRuntimeCommand):
    pass


@final
@dataclass(slots=True, frozen=True)
class IcoPauseCommand(IcoRuntimeCommand):
    pass


@final
@dataclass(slots=True, frozen=True)
class IcoResumeCommand(IcoRuntimeCommand):
    pass


@final
@dataclass(slots=True, frozen=True)
class IcoResetCommand(IcoRuntimeCommand):
    pass


@final
@dataclass(slots=True, frozen=True)
class IcoStopCommand(IcoRuntimeCommand):
    pass

from dataclasses import dataclass
from typing import ClassVar

from apriori.ico.core.runtime.command import (
    IcoActivateCommand,
    IcoDeactivateCommand,
    IcoRuntimeCommand,
)


@dataclass(slots=True, frozen=True)
class IcoRuntimeState:
    name: ClassVar[str] = "base_state"

    def is_ready(self) -> bool:
        return isinstance(self, ReadyState)

    def __str__(self) -> str:
        return type(self).name


@dataclass(slots=True, frozen=True)
class IdleState(IcoRuntimeState):
    name: ClassVar[str] = "Idle"


@dataclass(slots=True, frozen=True)
class ReadyState(IcoRuntimeState):
    name: ClassVar[str] = "Ready"


@dataclass(slots=True, frozen=True)
class RunningState(ReadyState):
    name: ClassVar[str] = "Running"


@dataclass(slots=True, frozen=True)
class FaultState(IcoRuntimeState):
    name: ClassVar[str] = "Fault"


StateTransitionMap = dict[type[IcoRuntimeCommand], type[IcoRuntimeState]]


class BaseStateModel:
    """Base state model for ICO runtime nodes. Node becomes ready after activation."""

    transitions: ClassVar[StateTransitionMap] = {
        IcoActivateCommand: ReadyState,
        IcoDeactivateCommand: IdleState,
    }

    __slots__ = ("state",)

    state: IcoRuntimeState

    def __init__(self) -> None:
        self.state = IdleState()

    def update(self, command: IcoRuntimeCommand) -> None:
        new_state: IcoRuntimeState | None = None

        for command_cls, state_cls in type(self).transitions.items():
            if isinstance(command, command_cls):
                new_state = state_cls()

        if new_state:
            self.state = new_state

    def idle(self) -> None:
        self.state = IdleState()

    def ready(self) -> None:
        if not self.state.is_ready():
            raise RuntimeError("Cannot transition to Ready state from non-Ready state.")

        self.state = ReadyState()

    def running(self) -> None:
        if not self.state.is_ready():
            raise RuntimeError(
                "Cannot transition to Running state from non-Ready state."
            )
        self.state = RunningState()

    def fault(self) -> None:
        self.state = FaultState()

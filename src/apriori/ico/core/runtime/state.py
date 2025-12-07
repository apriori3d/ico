from dataclasses import dataclass

from apriori.ico.core.runtime.command import (
    IcoActivateCommand,
    IcoDeactivateCommand,
    IcoRuntimeCommand,
)


@dataclass(slots=True, frozen=True)
class IcoRuntimeState:
    name: str = "base_state"

    def is_ready(self) -> bool:
        return isinstance(self, ReadyState)

    def __str__(self) -> str:
        return type(self).name


class InactiveState(IcoRuntimeState):
    name: str = "Inactive"


class ActiveState(IcoRuntimeState):
    name: str = "Active"


class ReadyState(ActiveState):
    name: str = "Ready"


class WaitingState(ReadyState):
    name: str = "Waiting"


class RunningState(ReadyState):
    name: str = "Running"


class SendingState(ReadyState):
    name: str = "Sending"


class FaultState(IcoRuntimeState):
    name: str = "Fault"


class BaseStateModel:
    """Base state model for ICO runtime nodes. Node becomes ready after activation."""

    transitions: dict[type[IcoRuntimeCommand], type[IcoRuntimeState]] = {
        IcoActivateCommand: ReadyState,
        IcoDeactivateCommand: InactiveState,
    }

    __slots__ = ("state",)

    state: IcoRuntimeState

    def __init__(self) -> None:
        self.state = InactiveState()

    def update(self, command: IcoRuntimeCommand) -> None:
        new_state: IcoRuntimeState | None = None

        for command_cls, state_cls in type(self).transitions.items():
            if isinstance(command, command_cls):
                new_state = state_cls()

        if new_state:
            self.state = new_state

    def inactive(self) -> None:
        self.state = InactiveState()

    def waiting(self) -> None:
        if not self.state.is_ready():
            raise RuntimeError(
                "Cannot transition to Waiting state from non-Ready state."
            )
        self.state = WaitingState()

    def running(self) -> None:
        if not self.state.is_ready():
            raise RuntimeError(
                "Cannot transition to Running state from non-Ready state."
            )
        self.state = RunningState()

    def sending(self) -> None:
        if not self.state.is_ready():
            raise RuntimeError(
                "Cannot transition to Sending state from non-Ready state."
            )
        self.state = SendingState()

    def ready(self) -> None:
        if not self.state.is_ready():
            raise RuntimeError("Cannot transition to Ready state from non-Ready state.")

        self.state = ReadyState()

    def fault(self) -> None:
        self.state = FaultState()

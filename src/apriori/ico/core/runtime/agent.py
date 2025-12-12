from abc import ABC
from dataclasses import dataclass
from typing import ClassVar

from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.core.runtime.state import (
    BaseStateModel,
    IcoRuntimeState,
    ReadyState,
)


@dataclass(slots=True, frozen=True)
class PendingState(IcoRuntimeState):
    name: ClassVar[str] = "Pending"


@dataclass(slots=True, frozen=True)
class WaitingState(ReadyState):
    name: ClassVar[str] = "Waiting"


@dataclass(slots=True, frozen=True)
class SendingState(ReadyState):
    name: ClassVar[str] = "Sending"


class AgentStateModel(BaseStateModel):
    """State model for runtime agents."""

    def pending(self) -> None:
        if self.state.is_ready():
            raise RuntimeError("Cannot transition to Pending state from Ready state.")
        self.state = PendingState()

    def waiting(self) -> None:
        if not self.state.is_ready():
            raise RuntimeError(
                "Cannot transition to Waiting state from non-Ready state."
            )
        self.state = WaitingState()

    def sending(self) -> None:
        if not self.state.is_ready():
            raise RuntimeError(
                "Cannot transition to Sending state from non-Ready state."
            )
        self.state = SendingState()


class IcoAgentNode(IcoRuntimeNode, ABC):
    runtime_type_name: ClassVar[str] = "Agent"

    def __init__(self, *, name: str | None = None) -> None:
        IcoRuntimeNode.__init__(
            self,
            runtime_name=name,
            state_model=AgentStateModel(),
        )

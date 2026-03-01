from __future__ import annotations

from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.runtime.command import IcoRuntimeCommand
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.core.runtime.runtime import IcoRuntime
from apriori.ico.core.runtime.state import BaseStateModel, IcoRuntimeState


class RecordingStateModel(BaseStateModel):
    """
    Test-only state model that records all state transitions.
    """

    recorded_states: list[type[IcoRuntimeState]]

    def __init__(self) -> None:
        super().__init__()
        self.recorded_states = [type(self._state)]

    def update_state(self, new_state: IcoRuntimeState) -> None:
        super().update_state(new_state)
        self.recorded_states.append(type(new_state))


class RecordingRuntimeNode(IcoRuntimeNode):
    """
    Test-only runtime node that records received commands + events.
    """

    recorded_commands: list[type[IcoRuntimeCommand]]
    recorded_events: list[type[IcoRuntimeEvent]]

    def __init__(
        self,
        *,
        name: str | None = None,
        runtime_children: list[IcoRuntimeNode] | None = None,
    ) -> None:
        super().__init__(
            runtime_children=runtime_children,
            runtime_name=name,
            state_model=RecordingStateModel(),
        )
        self.recorded_commands = []
        self.recorded_events = []

    @property
    def recorded_states(self) -> list[type[IcoRuntimeState]]:
        assert isinstance(self.state_model, RecordingStateModel)
        return self.state_model.recorded_states

    def on_command(self, command: IcoRuntimeCommand) -> None:
        self.recorded_commands.append(type(command))
        super().on_command(command)

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        self.recorded_events.append(type(event))
        return super().on_event(event)


class RecordingRuntime(IcoRuntime):
    recorded_commands: list[type[IcoRuntimeCommand]]
    recorded_events: list[type[IcoRuntimeEvent]]

    def __init__(
        self,
        closure: IcoOperator[None, None],
    ):
        super().__init__(
            closure=closure,
            name="state_recording_runtime",
            state_model=RecordingStateModel(),
        )
        self.recorded_commands = []
        self.recorded_events = []

    @property
    def recorded_states(self) -> list[type[IcoRuntimeState]]:
        assert isinstance(self.state_model, RecordingStateModel)
        return self.state_model.recorded_states

    def on_command(self, command: IcoRuntimeCommand) -> None:
        self.recorded_commands.append(type(command))
        super().on_command(command)

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        self.recorded_events.append(type(event))
        return super().on_event(event)

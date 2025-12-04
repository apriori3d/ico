from __future__ import annotations

from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.runtime.command import IcoRuntimeCommand, IcoRuntimeCommandType
from apriori.ico.core.runtime.contour import IcoRuntimeContour
from apriori.ico.core.runtime.event import IcoRuntimeEvent, IcoRuntimeEventType
from apriori.ico.core.runtime.node import IcoRuntimeNode, IcoRuntimeState


class RecordingRuntimeNode(IcoRuntimeNode):
    """
    Test-only runtime node that records received commands + events.
    """

    recorded_commands: list[IcoRuntimeCommandType]
    recorded_events: list[IcoRuntimeEventType]
    recorded_states: list[IcoRuntimeState]

    def __init__(
        self,
        *,
        name: str | None = None,
        runtime_children: list[IcoRuntimeNode] | None = None,
    ) -> None:
        super().__init__(
            runtime_children=runtime_children,
            name=name,
        )
        self.recorded_states = [self._state]
        self.recorded_commands = []
        self.recorded_events = []

    @property
    def state(self) -> IcoRuntimeState:
        return self._state

    @state.setter
    def state(self, state: IcoRuntimeState) -> None:
        self._state = state
        self.recorded_states.append(state)

    def on_command(self, command: IcoRuntimeCommand) -> IcoRuntimeCommand | None:
        super().on_command(command)
        self.recorded_commands.append(command.type)

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        super().on_event(event)
        self.recorded_events.append(event.type)


class RecordingContour(IcoRuntimeContour):
    recorded_commands: list[IcoRuntimeCommandType]
    recorded_events: list[IcoRuntimeEventType]
    recorded_states: list[IcoRuntimeState]

    def __init__(
        self,
        closure: IcoOperator[None, None],
    ):
        super().__init__(closure=closure, name="state_recording_contour")
        self.recorded_states = [self._state]
        self.recorded_commands = []
        self.recorded_events = []

    @property
    def state(self) -> IcoRuntimeState:
        return self._state

    @state.setter
    def state(self, state: IcoRuntimeState) -> None:
        self._state = state
        self.recorded_states.append(state)

    def on_command(self, command: IcoRuntimeCommand) -> IcoRuntimeCommand | None:
        super().on_command(command)
        self.recorded_commands.append(command.type)

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        super().on_event(event)
        self.recorded_events.append(event.type)

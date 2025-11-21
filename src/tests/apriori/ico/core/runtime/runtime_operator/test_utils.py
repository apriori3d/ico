from __future__ import annotations

from collections.abc import Callable, Sequence

from apriori.ico.core.runtime.operator import IcoRuntimeOperator
from apriori.ico.core.runtime.types import (
    IcoRuntimeCommandType,
    IcoRuntimeEvent,
    IcoRuntimeStateType,
)


class StateRecordingRuntime(IcoRuntimeOperator):
    states: list[IcoRuntimeStateType]

    def __init__(
        self,
        fn: Callable[[None], None],
        *,
        children: Sequence[IcoNode] | None = None,
        name: str | None = None,
    ) -> None:
        self.states = []
        super().__init__(fn=fn, children=children, name=name)

    def _set_state(self, state: IcoRuntimeStateType) -> None:
        self._state = state
        self.states.append(state)


class ControlRecordingRuntime(IcoRuntimeOperator):
    """
    Test-only runtime node that records received commands + events.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        runtime_children: list[IcoRuntimeOperator] | None = None,
    ) -> None:
        super().__init__(
            fn=lambda _: None,
            runtime_children=runtime_children,
            name=name,
        )
        self.received_commands: list[IcoRuntimeCommandType] = []
        self.received_events: list[IcoRuntimeEvent] = []

    def on_command(self, command: IcoRuntimeCommandType) -> None:
        super().on_command(command)
        self.received_commands.append(command)

    def on_event(self, event: IcoRuntimeEvent) -> None:
        super().on_event(event)
        self.received_events.append(event)

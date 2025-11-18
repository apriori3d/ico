from __future__ import annotations

from typing import Any

from apriori.ico.core.runtime.types import (
    IcoRuntimeCommandType,
    IcoRuntimeEventProtocol,
    IcoRuntimeStateType,
)

COMMAND_TO_STATE = {
    IcoRuntimeCommandType.activate: IcoRuntimeStateType.ready,
    IcoRuntimeCommandType.reset: IcoRuntimeStateType.ready,
    IcoRuntimeCommandType.deactivate: IcoRuntimeStateType.inactive,
    IcoRuntimeCommandType.pause: IcoRuntimeStateType.paused,
    IcoRuntimeCommandType.resume: IcoRuntimeStateType.ready,
}


class IcoRuntimeStateMixin:
    _state: IcoRuntimeStateType
    _last_command: IcoRuntimeCommandType | None
    _last_event: IcoRuntimeEventProtocol | None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._set_state(IcoRuntimeStateType.inactive)
        self._last_command = None
        self._last_event = None

    # ─── Properties ───

    @property
    def state(self) -> IcoRuntimeStateType:
        """Current runtime state of the operator."""
        return self._state

    def _set_state(self, state: IcoRuntimeStateType) -> None:
        self._state = state

    @property
    def last_command(self) -> IcoRuntimeCommandType | None:
        """Last received runtime command."""
        return self._last_command

    @property
    def last_event(self) -> IcoRuntimeEventProtocol | None:
        """Last received runtime event."""
        return self._last_event

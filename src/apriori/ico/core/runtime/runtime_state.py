from __future__ import annotations

from collections.abc import Callable

from apriori.ico.core.runtime.events import IcoRuntimeEvent
from apriori.ico.core.runtime.types import (
    IcoRuntimeCommandType,
    IcoRuntimeProtocol,
    IcoRuntimeStateType,
)

COMMAND_TO_STATE = {
    IcoRuntimeCommandType.activate: IcoRuntimeStateType.ready,
    IcoRuntimeCommandType.reset: IcoRuntimeStateType.ready,
    IcoRuntimeCommandType.deactivate: IcoRuntimeStateType.inactive,
    IcoRuntimeCommandType.pause: IcoRuntimeStateType.paused,
    IcoRuntimeCommandType.resume: IcoRuntimeStateType.running,
}


class IcoRuntimeStateMixin:
    _state: IcoRuntimeStateType
    _last_command: IcoRuntimeCommandType | None
    _last_event: IcoRuntimeEvent | None

    def __init__(self) -> None:
        super().__init__()
        if not isinstance(self, IcoRuntimeProtocol):
            raise TypeError(
                "IcoRuntimeLifecycleMixin can only be used with IcoRuntimeProtocol instances"
            )
        self._state = IcoRuntimeStateType.inactive
        self._last_command = None
        self._last_event = None

    # ─── Runtime State tracking ───

    def _track(self, fn: Callable[[None], None]) -> None:
        """Execute function while managing runtime state transitions."""
        try:
            self._state = IcoRuntimeStateType.running
            fn(None)
            self._state = IcoRuntimeStateType.ready
        except Exception:
            self._state = IcoRuntimeStateType.error
            raise

    # ─── Properties ───

    @property
    def state(self) -> IcoRuntimeStateType:
        """Current runtime state of the operator."""
        return self._state

    @property
    def last_command(self) -> IcoRuntimeCommandType | None:
        """Last received runtime command."""
        return self._last_command

    @property
    def last_event(self) -> IcoRuntimeEvent | None:
        """Last received runtime event."""
        return self._last_event

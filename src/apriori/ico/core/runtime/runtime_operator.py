from __future__ import annotations

from collections.abc import Callable, Sequence

from typing_extensions import Self

from apriori.ico.core.dsl.operator import IcoOperator
from apriori.ico.core.runtime.runtime_state import (
    COMMAND_TO_STATE,
    IcoRuntimeStateMixin,
)
from apriori.ico.core.runtime.runtime_tree import IcoRuntimeTreeMixin
from apriori.ico.core.runtime.types import (
    IcoRuntimeCommandType,
    IcoRuntimeEventProtocol,
    IcoRuntimeStateType,
)
from apriori.ico.core.types import IcoNodeProtocol, IcoNodeType


class IcoRuntimeOperator(
    IcoOperator[None, None],
    IcoRuntimeStateMixin,
    IcoRuntimeTreeMixin,
):
    def __init__(
        self,
        fn: Callable[[None], None] | None = None,
        *,
        children: Sequence[IcoNodeProtocol] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(
            fn=fn or self._noop_fn,
            name=name,
            node_type=IcoNodeType.runtime,
            children=children,
        )

    # ─── Execution ───

    def __call__(self, item: None) -> None:
        try:
            self._state = IcoRuntimeStateType.running
            super().__call__(None)
            self._state = IcoRuntimeStateType.ready
        except Exception:
            self._state = IcoRuntimeStateType.error
            raise

    def _noop_fn(self, _: None) -> None:
        pass

    def run(self) -> Self:
        """Execute the contour by calling itself."""
        self(None)
        return self

    # ─── Command Handling ───

    def broadcast_command(self, command: IcoRuntimeCommandType) -> None:
        super().broadcast_command(command)
        self.on_command(command)

    def on_command(self, command: IcoRuntimeCommandType) -> None:
        """
        Handle a single runtime command.

        Subclasses may override to implement additional behavior
        (e.g., resource allocation, reset hooks, or teardown logic).
        """
        self._state = COMMAND_TO_STATE.get(command, self._state)
        self._last_command = command

    # ─── Event Handling ───

    def bubble_event(self, event: IcoRuntimeEventProtocol) -> None:
        self.on_event(event)
        super().bubble_event(event)

    def on_event(self, event: IcoRuntimeEventProtocol) -> None:
        """
        Handle a single runtime event.

        Subclasses may override to implement additional behavior
        (e.g., logging, metrics, or alerting).
        """
        self._last_event = event

    # ─── Declarative runtime control ───

    def activate(self) -> Self:
        """Broadcast 'activate' event through the entire flow."""
        self.broadcast_command(IcoRuntimeCommandType.activate)
        return self

    def reset(self) -> Self:
        """Broadcast 'reset' event through the entire flow."""
        self.broadcast_command(IcoRuntimeCommandType.reset)
        return self

    def deactivate(self) -> Self:
        """Broadcast 'deactivate' event through the entire flow."""
        self.broadcast_command(IcoRuntimeCommandType.deactivate)
        return self

    def pause(self) -> Self:
        """Broadcast 'pause' event through the entire flow."""
        self.broadcast_command(IcoRuntimeCommandType.pause)
        return self

    def resume(self) -> Self:
        """Broadcast 'resume' event through the entire flow."""
        self.broadcast_command(IcoRuntimeCommandType.resume)
        return self

    def stop(self) -> Self:
        """Broadcast 'stop' event through the entire flow."""
        self.broadcast_command(IcoRuntimeCommandType.stop)
        return self

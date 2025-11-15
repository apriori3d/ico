from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from typing_extensions import Self

from apriori.ico.core.dsl.operator import O2
from apriori.ico.core.runtime.events import IcoRuntimeEvent
from apriori.ico.core.runtime.runtime_hierarchy import IcoRuntimeHierarchyMixin
from apriori.ico.core.runtime.runtime_state import (
    COMMAND_TO_STATE,
    IcoRuntimeStateMixin,
)
from apriori.ico.core.runtime.types import IcoRuntimeCommandType, IcoRuntimeProtocol
from apriori.ico.core.types import IcoOperatorProtocol, NodeType


class IcoRuntimeOperator(
    IcoRuntimeStateMixin,
    IcoRuntimeHierarchyMixin,
    IcoRuntimeProtocol,
):
    name: str
    children: list[IcoOperatorProtocol[Any, Any]]
    parent: IcoOperatorProtocol[Any, Any] | None
    node_type: NodeType
    fn: Callable[[None], None]

    def __init__(self) -> None:
        super().__init__()
        self.name = self.__class__.__name__
        self.parent = None
        self.children = []
        self.node_type = NodeType.runtime
        self.fn = self._noop_fn

    # ─── Command Handling ───

    def broadcast_command(self, command):
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

    def bubble_event(self, event):
        self.on_event(event)
        super().bubble_event(event)

    def on_event(self, event: IcoRuntimeEvent) -> None:
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

    # ─── Execution ───

    def _noop_fn(self, _: None) -> None:
        pass

    def run(self) -> Self:
        """Execute the contour by calling itself."""
        return self

    # ─── Declarative sync execution path ───

    def __call__(self, _: None) -> None:
        raise RuntimeError(
            "IcoRuntimeMixin does not implement data flow and has ICO form () → ()"
        )

    # ─── Imperative async execution path ───

    async def run_async(self, item: None) -> None:
        raise RuntimeError(
            "IcoRuntimeMixin does not implement data flow and has ICO form () → ()"
        )

    # ─── Operator composition ───

    def chain(
        self, other: IcoOperatorProtocol[None, O2]
    ) -> IcoOperatorProtocol[None, O2]:
        raise RuntimeError(
            "IcoRuntimeMixin does not implement data flow and has ICO form () → ()"
        )

    def __or__(
        self, other: IcoOperatorProtocol[None, O2]
    ) -> IcoOperatorProtocol[None, O2]:
        raise RuntimeError(
            "IcoRuntimeMixin does not implement data flow and has ICO form () → ()"
        )

    def map(self) -> IcoOperatorProtocol[Iterator[None], Iterator[None]]:
        raise RuntimeError(
            "IcoRuntimeMixin does not implement data flow and has ICO form () → ()"
        )

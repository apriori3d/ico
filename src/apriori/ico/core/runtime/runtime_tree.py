from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from typing_extensions import Self

from apriori.ico.core.runtime.progress.types import ProgressProtocol, SupportsProgress
from apriori.ico.core.runtime.types import (
    IcoRuntimeCommandType,
    IcoRuntimeEventProtocol,
    IcoRuntimeOperatorProtocol,
    IcoRuntimeTreeProtocol,
)


class IcoRuntimeTreeMixin:
    runtime_children: list[IcoRuntimeTreeProtocol]
    _runtime_parent: IcoRuntimeTreeProtocol | None

    def __init__(
        self,
        runtime_parent: IcoRuntimeTreeProtocol | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.runtime_children = []
        self._runtime_parent = runtime_parent

    @property
    def runtime_parent(self) -> IcoRuntimeTreeProtocol | None:
        return self._runtime_parent

    @runtime_parent.setter
    def runtime_parent(self, value: IcoRuntimeTreeProtocol | None) -> None:
        self._runtime_parent = value

    # ─── Command & Event Propagation ───

    def broadcast_command(
        self,
        command: IcoRuntimeCommandType,
    ) -> None:
        """
        Recursively propagate a runtime command through the operator tree.

        Each node implementing `SupportsIcoRuntime` receives `on_command(command)`.
        """
        for child in self.runtime_children:
            child.broadcast_command(command)

    def bubble_event(self, event: IcoRuntimeEventProtocol) -> None:
        """
        Propagate a runtime event upward until a contour or agent host is reached.
        """
        if self._runtime_parent:
            self._runtime_parent.bubble_event(event)

    def connect_runtime(self, runtime: IcoRuntimeTreeProtocol) -> None:
        """Connect to a runtime host for command propagation."""
        if runtime not in self.runtime_children:
            self.runtime_children.append(runtime)
        runtime.runtime_parent = self

    def disconnect_runtime(self, runtime: IcoRuntimeTreeProtocol) -> None:
        """Disconnect from a runtime host."""
        if runtime in self.runtime_children:
            self.runtime_children.remove(runtime)
        runtime.runtime_parent = None

    # ─── Progress ───

    def attach_progress(self, progress: ProgressProtocol) -> Self:
        """
        Bind a shared progress relay to all progress-capable nodes.

        Returns:
            Self — allows chaining: contour.bind_progress().ready().run().idle()
        """
        self.progress = progress
        for runtime in iterate_nodes(self):
            if isinstance(runtime, SupportsProgress):
                runtime.progress = self.progress

        return self


def iterate_nodes(
    node: Any, strict: bool = True
) -> Iterator[IcoRuntimeOperatorProtocol]:
    """Recursively yield all children operators in the flow tree."""
    if strict and not isinstance(node, IcoRuntimeOperatorProtocol):
        raise TypeError(
            f"Operator {node} in runtime should be an instance of IcoRuntimeProtocol"
        )
    yield node
    for c in node.runtime_children:
        yield from iterate_nodes(c, strict)

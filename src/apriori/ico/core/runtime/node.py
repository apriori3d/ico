from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from typing import TypeAlias

from typing_extensions import Protocol, Self, runtime_checkable

from apriori.ico.core.runtime.command import (
    IcoActivateCommand,
    IcoDeactivateCommand,
    IcoRunCommand,
    IcoRuntimeCommand,
)
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.state import (
    BaseStateModel,
    IcoRuntimeState,
    IcoStateEvent,
    IcoStateRequestCommand,
)
from apriori.ico.core.tree_utils import TraversalInfo, TreeWalker

# ────────────────────────────────────────────────
# Runtime node
# ────────────────────────────────────────────────


class IcoRuntimeNode(ABC):
    """Structural attributes for graph representation of ICO operators."""

    __slots__ = (
        "state_model",
        "runtime_name",
        "_runtime_children",
        "_runtime_parent",
    )

    state_model: BaseStateModel
    runtime_name: str | None
    _runtime_children: list[IcoRuntimeNode]
    _runtime_parent: IcoRuntimeNode | None

    def __init__(
        self,
        runtime_name: str | None = None,
        runtime_parent: IcoRuntimeNode | None = None,
        runtime_children: Sequence[IcoRuntimeNode] | None = None,
        state_model: BaseStateModel | None = None,
    ) -> None:
        self.runtime_name = runtime_name
        self._runtime_parent = runtime_parent
        self._runtime_children = (
            list(runtime_children) if runtime_children is not None else []
        )
        self.state_model = state_model or BaseStateModel()

        for child in self._runtime_children:
            child._runtime_parent = self

    # ────────────────────────────────────────────────
    # Properties
    # ────────────────────────────────────────────────

    @property
    def runtime_parent(self) -> IcoRuntimeNode | None:
        """Parent node in the runtime tree."""
        return self._runtime_parent

    @property
    def runtime_children(self) -> list[IcoRuntimeNode]:
        """Children nodes in the runtime tree."""
        return self._runtime_children

    @property
    def state(self) -> IcoRuntimeState:
        """Current runtime state of the operator."""
        return self.state_model.state

    # ────────────────────────────────────────────────
    # Commands
    # ────────────────────────────────────────────────

    def on_command(self, command: IcoRuntimeCommand) -> None:
        """
        Handle a single runtime command.

        Subclasses may override to implement additional behavior
        (e.g., resource allocation, reset hooks, or teardown logic).
        """
        self.state_model.update(command)

        if isinstance(command, IcoStateRequestCommand):
            # Respond with current state
            self.bubble_event(
                IcoStateEvent.create(state=self.state),
            )

    def broadcast_command(self, command: IcoRuntimeCommand) -> None:
        def visit_fn(node_info: BroadcastTraversalInfo) -> None:
            assert node_info.context is not None
            node_info.node.on_command(node_info.context)

        walker = create_broadcast_walker(command)
        walker.walk(
            self,
            visit_fn=visit_fn,
            order=command.broadcast_order,
        )

    # ────────────────────────────────────────────────
    # Events
    # ────────────────────────────────────────────────

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        """
        Handle a single runtime event.

        Subclasses may override to implement additional behavior
        (e.g., logging, metrics, or alerting).
        """
        return event

    def bubble_event(
        self,
        event: IcoRuntimeEvent,
        from_child: IcoRuntimeNode | None = None,
    ) -> None:
        """
        Propagate a runtime event upward until a contour or agent host is reached.
        """
        if from_child is not None:
            child_index = self._runtime_children.index(from_child)
            event = replace(event, trace=event.trace.add_child(child_index))

        # If event is handled and should not propagate further, stop here
        if not self.on_event(event):
            return None

        if self._runtime_parent:
            self._runtime_parent.bubble_event(event, from_child=self)

    # ────────────────────────────────────────────────
    # Runtime Tree Management
    # ────────────────────────────────────────────────

    def add_runtime_children(self, *children: IcoRuntimeNode) -> Self:
        """Connect to a runtime host for command propagation."""
        for child in children:
            if child.runtime_parent is not None:
                raise ValueError(
                    f"Child node {child} already has a runtime parent "
                    f"{child.runtime_parent}"
                )
            if child not in self._runtime_children:
                self._runtime_children.append(child)
            child._runtime_parent = self
        return self

    def remove_runtime_child(self, *children: IcoRuntimeNode) -> Self:
        """Disconnect from a runtime host."""
        for child in children:
            if child.runtime_parent is not self:
                raise ValueError(f"Child node {child} is not a runtime child of {self}")
            if child in self._runtime_children:
                self._runtime_children.remove(child)
            child._runtime_parent = None
        return self

    # ────────────────────────────────────────────────
    # Runtime Commands API
    # ────────────────────────────────────────────────

    def activate(self) -> Self:
        """Broadcast 'activate' event through the entire flow."""
        self.broadcast_command(IcoActivateCommand.create())
        return self

    def run(self) -> Self:
        """Broadcast 'run' event through the entire flow."""
        self.broadcast_command(IcoRunCommand.create())
        return self

    def deactivate(self) -> Self:
        """Broadcast 'deactivate' event through the entire flow."""
        self.broadcast_command(IcoDeactivateCommand.create())
        return self

    # ────────────────────────────────────────────────
    # Describe util interface
    # ────────────────────────────────────────────────

    def describe(self) -> None:
        from apriori.ico.describe.describer import describe as describe_util

        describe_util(self)

    # ────────────────────────────────────────────────
    # Hirarchy Iteration
    # ────────────────────────────────────────────────

    def iterate_nodes(self) -> Iterator[IcoRuntimeNode]:
        """Recursively yield all children operators in the flow tree."""
        yield self
        for c in self.runtime_children:
            yield from c.iterate_nodes()

    def iterate_parents(self) -> Iterator[IcoRuntimeNode]:
        """Recursively yield all parent operators in the flow tree."""
        if self.runtime_parent is None:
            return

        yield self.runtime_parent
        yield from self.runtime_parent.iterate_parents()


# ────────────────────────────────────────────────
# Remote runtime factory protocol
# ────────────────────────────────────────────────


class IcoRemotePlaceholderNode(IcoRuntimeNode):
    """Placeholder node representing a remote runtime node in the local runtime tree.
    Need to be able to use correct tree path index in command and events."""

    pass


@runtime_checkable
class HasRemoteRuntime(Protocol):
    def get_remote_runtime_factory(self) -> Callable[[], IcoRuntimeNode]: ...


# ────────────────────────────────────────────────
# Tree walker api
# ────────────────────────────────────────────────

# ──────── Command broadcast walker ────────

BroadcastTreeWalker: TypeAlias = TreeWalker[IcoRuntimeNode, IcoRuntimeCommand]
BroadcastTraversalInfo: TypeAlias = TraversalInfo[IcoRuntimeNode, IcoRuntimeCommand]


def create_broadcast_walker(command: IcoRuntimeCommand) -> BroadcastTreeWalker:
    def _get_children(node: IcoRuntimeNode) -> Sequence[IcoRuntimeNode]:
        return [
            c
            for c in node.runtime_children
            if not isinstance(c, IcoRemotePlaceholderNode)
        ]

    return BroadcastTreeWalker(
        get_children_fn=_get_children,
        initial_context=command,
    )


# ──────── Runtime tree walker ────────

RuntimeTreeWalker: TypeAlias = TreeWalker[IcoRuntimeNode, None]
RuntimeTraversalInfo: TypeAlias = TraversalInfo[IcoRuntimeNode, None]


def create_runtime_walker(expand_remote_runtimes: bool = False) -> RuntimeTreeWalker:
    """Get a tree walker for ICO runtime nodes."""

    def _get_children(node: IcoRuntimeNode) -> Sequence[IcoRuntimeNode]:
        children = list(node.runtime_children)

        if expand_remote_runtimes and isinstance(node, HasRemoteRuntime):
            factory = node.get_remote_runtime_factory()
            children.append(factory())

        return children

    return RuntimeTreeWalker(get_children_fn=_get_children)

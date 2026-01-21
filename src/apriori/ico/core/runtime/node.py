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
from apriori.ico.core.runtime.state import BaseStateModel, IcoRuntimeState
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

    def _on_broadcast_visit(self, node_info: BroadcastTraversalInfo) -> None:
        assert node_info.context is not None
        self.on_command(node_info.context)

    def broadcast_command(self, command: IcoRuntimeCommand) -> None:
        walker = create_broadcast_walker(command)
        walker.walk(
            self,
            visit_fn=self._on_broadcast_visit,
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
    ) -> IcoRuntimeEvent | None:
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
            if child not in self._runtime_children:
                self._runtime_children.append(child)
            child._runtime_parent = self
        return self

    def remove_runtime_child(self, child: IcoRuntimeNode) -> Self:
        """Disconnect from a runtime host."""
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
# Sub-tree factory protocol
# ────────────────────────────────────────────────


@runtime_checkable
class HasSubTreeFactory(Protocol):
    subtree_factory: Callable[[], IcoRuntimeNode]


# ────────────────────────────────────────────────
# Tree walker api
# ────────────────────────────────────────────────


BroadcastTreeWalker: TypeAlias = TreeWalker[IcoRuntimeNode, IcoRuntimeCommand]
BroadcastTraversalInfo: TypeAlias = TraversalInfo[IcoRuntimeNode, IcoRuntimeCommand]


def create_broadcast_walker(
    command: IcoRuntimeCommand,
) -> BroadcastTreeWalker:
    return BroadcastTreeWalker(
        get_children_fn=lambda node: node.runtime_children,
        initial_context=command,
    )


RuntimeTreeWalker: TypeAlias = TreeWalker[IcoRuntimeNode, None]
RuntimeTraversalInfo: TypeAlias = TraversalInfo[IcoRuntimeNode, None]


def create_runtime_tree_walker(
    *,
    include_agent_worker: bool = True,
) -> RuntimeTreeWalker:
    """Get a tree walker for ICO runtime nodes."""
    return RuntimeTreeWalker(
        get_children_fn=lambda node: node.runtime_children,
        get_lazy_subtree_fn=_create_worker_node if include_agent_worker else None,
    )


def _create_worker_node(node: IcoRuntimeNode) -> Sequence[IcoRuntimeNode] | None:
    if isinstance(node, HasSubTreeFactory):
        return [node.subtree_factory()]
    return None

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import (
    Protocol,
    TypeAlias,
)

from typing_extensions import runtime_checkable

from apriori.ico.core.signature import IcoSignature
from apriori.ico.core.tree_utils import TraversalInfo, TreeWalker


class IcoNode:
    """Structural attributes for graph representation of ICO operators."""

    name: str | None
    parent: IcoNode | None
    children: Sequence[IcoNode]

    def __init__(
        self,
        name: str | None = None,
        parent: IcoNode | None = None,
        children: Sequence[IcoNode] | None = None,
    ) -> None:
        self.name = name
        self.parent = parent
        self.children = children or []

        for child in self.children:
            child.parent = self

    def __str__(self) -> str:
        return type(self).__name__

    @property
    def signature(self) -> IcoSignature:
        return IcoSignature(i=None, c=None, o=None, infered=False)

    # ────────────────────────────────────────────────
    # Describe util interface
    # ────────────────────────────────────────────────

    def describe(self) -> None:
        from apriori.ico.describe.describer import describe

        describe(self)


# ────────────────────────────────────────────────
# Remote flow protocol
# ────────────────────────────────────────────────


@runtime_checkable
class HasRemoteFlow(Protocol):
    def get_remote_flow_factory(self) -> Callable[[], IcoNode]: ...


# ────────────────────────────────────────────────
# Iteration api
# ────────────────────────────────────────────────


def iterate_nodes(
    node: IcoNode,
) -> Iterator[IcoNode]:
    """Recursively yield all children operators in the flow tree."""
    yield node
    for c in node.children:
        yield from iterate_nodes(c)


def iterate_parents(
    node: IcoNode,
) -> Iterator[IcoNode]:
    """Recursively yield all parent operators in the flow tree."""
    if node.parent is None:
        return

    yield node.parent
    yield from iterate_parents(node.parent)


# ────────────────────────────────────────────────
# Tree walker api
# ────────────────────────────────────────────────


FlowTreeWalker: TypeAlias = TreeWalker[IcoNode, None]
FlowTraversalInfo: TypeAlias = TraversalInfo[IcoNode, None]


def create_flow_walker(expand_remote_flows: bool = False) -> FlowTreeWalker:
    """Get a tree walker for ICO runtime nodes."""

    def _get_children(node: IcoNode) -> Sequence[IcoNode]:
        children = list(node.children)

        if expand_remote_flows and isinstance(node, HasRemoteFlow):
            factory = node.get_remote_flow_factory()
            children.append(factory())

        return children

    return FlowTreeWalker(get_children_fn=_get_children)

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import (
    Protocol,
    TypeAlias,
)

from typing_extensions import runtime_checkable

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

    # ────────────────────────────────────────────────
    # Describe util interface
    # ────────────────────────────────────────────────

    def describe(self) -> None:
        from apriori.ico.describe.describer import describe

        describe(self)


# ────────────────────────────────────────────────
# Sub-flow factory protocol
# ────────────────────────────────────────────────


@runtime_checkable
class HasSubflowFactory(Protocol):
    def get_subflow_factory(self) -> Callable[[], IcoNode] | None: ...


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


def create_flow_walker(*, visit_subflows: bool = True) -> FlowTreeWalker:
    """Get a tree walker for ICO runtime nodes."""
    return FlowTreeWalker(
        get_children_fn=lambda node: node.children,
        get_lazy_subtree_fn=_create_node_subflow if visit_subflows else None,
        subtree_policy="children_or_subtree",
    )


def _create_node_subflow(node: IcoNode) -> Sequence[IcoNode] | None:
    if isinstance(node, HasSubflowFactory) and node.get_subflow_factory is not None:
        return [node.get_subflow_factory()]

    return None

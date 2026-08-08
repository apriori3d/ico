from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import (
    Protocol,
    TypeAlias,
    runtime_checkable,
)

from ico.core.tree_utils import TraversalInfo, TreeWalker


@runtime_checkable
class IcoNodeProtocol(Protocol):
    name: str | None
    parent: IcoNodeProtocol | None
    children: Sequence[IcoNodeProtocol]

    def describe(self) -> None: ...


class IcoNode:
    """Structural attributes for graph representation of ICO operators.

    IcoNode serves as the base class for all ICO framework elements, providing
    hierarchical tree structure with parent-child relationships. This enables
    graph traversal, composition analysis, and visualization of computation flows.

    Attributes:
        name: Optional human-readable identifier for the node.
        parent: Reference to the parent node in the tree structure.
        children: Sequence of child nodes in the tree structure.
    """

    name: str | None
    parent: IcoNodeProtocol | None
    children: Sequence[IcoNodeProtocol]

    def __init__(
        self,
        name: str | None = None,
        parent: IcoNodeProtocol | None = None,
        children: Sequence[IcoNodeProtocol] | None = None,
    ) -> None:
        """Initialize an ICO node with optional name and tree relationships.

        Args:
            name: Optional human-readable identifier for the node.
            parent: Optional parent node to establish hierarchy.
            children: Optional sequence of child nodes to connect.

        Note:
            Setting children automatically updates their parent references.
        """
        self.name = name
        self.parent = parent
        self.children = children or []

        for child in self.children:
            child.parent = self

    def __str__(self) -> str:
        """Return a string representation of the node.

        Returns:
            The class name of the node.
        """
        return type(self).__name__

    # ────────────────────────────────────────────────
    # Describe utility interface
    # ────────────────────────────────────────────────

    def describe(self) -> None:
        """Display a rich visual description of this node and its structure.

        Uses the ICO describe system to render tree structure, signatures,
        and other relevant information to the console.
        """
        from ico.describe.describer import describe

        describe(self)


# ────────────────────────────────────────────────
# Remote flow protocol
# ────────────────────────────────────────────────


@runtime_checkable
class HasRemoteFlow(Protocol):
    """Protocol for nodes that can provide remote flow factories.

    Nodes implementing this protocol can expand their structure by creating
    remote flow instances, enabling distributed computation patterns.
    """

    def get_remote_flow_factory(self) -> Callable[[], IcoNode]: ...


# ────────────────────────────────────────────────
# Iteration api
# ────────────────────────────────────────────────


def iterate_nodes(
    node: IcoNodeProtocol,
) -> Iterator[IcoNodeProtocol]:
    """Recursively yield all children operators in the flow tree."""
    yield node
    for c in node.children:
        yield from iterate_nodes(c)


def iterate_parents(
    node: IcoNodeProtocol,
) -> Iterator[IcoNodeProtocol]:
    """Recursively yield all parent operators in the flow tree."""
    if node.parent is None:
        return

    yield node.parent
    yield from iterate_parents(node.parent)


# ────────────────────────────────────────────────
# Tree walker api
# ────────────────────────────────────────────────


FlowTreeWalker: TypeAlias = TreeWalker[IcoNodeProtocol, None]
FlowTraversalInfo: TypeAlias = TraversalInfo[IcoNodeProtocol, None]


def create_flow_walker(expand_remote_flows: bool = False) -> FlowTreeWalker:
    """Create a tree walker for ICO flow nodes.

    Args:
        expand_remote_flows: If True, expand nodes with remote flows by
            calling their factory functions to include remote children.

    Returns:
        A configured FlowTreeWalker that can traverse ICO node hierarchies.
    """

    def _get_children(node: IcoNodeProtocol) -> Sequence[IcoNodeProtocol]:
        children = list(node.children)

        if expand_remote_flows and isinstance(node, HasRemoteFlow):
            factory = node.get_remote_flow_factory()
            children.append(factory())

        return children

    return FlowTreeWalker(get_children_fn=_get_children)

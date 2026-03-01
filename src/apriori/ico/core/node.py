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
    parent: IcoNode | None
    children: Sequence[IcoNode]

    def __init__(
        self,
        name: str | None = None,
        parent: IcoNode | None = None,
        children: Sequence[IcoNode] | None = None,
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

    @property
    def signature(self) -> IcoSignature:
        """Get the ICO type signature for this node.

        Base implementation returns an untyped signature. Subclasses should
        override this to provide specific input/output type information.

        Returns:
            IcoSignature with None types and infered=False.
        """
        return IcoSignature(i=None, c=None, o=None, infered=False)

    # ────────────────────────────────────────────────
    # Describe utility interface
    # ────────────────────────────────────────────────

    def describe(self) -> None:
        """Display a rich visual description of this node and its structure.

        Uses the ICO describe system to render tree structure, signatures,
        and other relevant information to the console.
        """
        from apriori.ico.describe.describer import describe

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
    """Create a tree walker for ICO flow nodes.

    Args:
        expand_remote_flows: If True, expand nodes with remote flows by
            calling their factory functions to include remote children.

    Returns:
        A configured FlowTreeWalker that can traverse ICO node hierarchies.
    """

    def _get_children(node: IcoNode) -> Sequence[IcoNode]:
        children = list(node.children)

        if expand_remote_flows and isinstance(node, HasRemoteFlow):
            factory = node.get_remote_flow_factory()
            children.append(factory())

        return children

    return FlowTreeWalker(get_children_fn=_get_children)

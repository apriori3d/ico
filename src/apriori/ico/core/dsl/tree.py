# ────────────────────────────────────────────────
# Tree traversal Utilities
# ────────────────────────────────────────────────


from collections.abc import Iterator

from apriori.ico.core.types import IcoNodeProtocol


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

from __future__ import annotations

from collections.abc import Iterator, Sequence


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

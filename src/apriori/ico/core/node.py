from __future__ import annotations

from collections.abc import Iterator, Sequence


class IcoNode:
    """Structural attributes for graph representation of ICO operators."""

    name: str
    parent: IcoNode | None
    children: Sequence[IcoNode]
    ico_form_target: object | None

    def __init__(
        self,
        name: str | None = None,
        ico_form_target: object | None = None,
        parent: IcoNode | None = None,
        children: Sequence[IcoNode] | None = None,
    ) -> None:
        self.name = name or self.__class__.__name__
        self.ico_form_target = ico_form_target
        self.parent = parent
        self.children = children or []

        for child in self.children:
            child.parent = self


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

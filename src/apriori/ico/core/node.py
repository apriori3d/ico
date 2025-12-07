from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import ClassVar


class IcoNode:
    """Structural attributes for graph representation of ICO operators."""

    type_name: ClassVar[str] = "Node"

    name: str | None
    parent: IcoNode | None
    children: Sequence[IcoNode]
    original_fn: object | None

    def __init__(
        self,
        name: str | None = None,
        original_fn: object | None = None,
        parent: IcoNode | None = None,
        children: Sequence[IcoNode] | None = None,
    ) -> None:
        self.name = name
        self.original_fn = original_fn
        self.parent = parent
        self.children = children or []

        for child in self.children:
            child.parent = self

    def __str__(self) -> str:
        return self.name or self.type_name

    # ────────────────────────────────────────────────
    # Describe util interface
    # ────────────────────────────────────────────────

    def describe(
        self,
        *,
        show_ico_form: bool = True,
        include_runtime: bool = False,
    ) -> None:
        from apriori.ico.core.meta.describer import describe as describe_util

        describe_util(
            self,
            show_ico_form=show_ico_form,
            include_runtime=include_runtime,
        )


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

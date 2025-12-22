from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar, final


@final
@dataclass(slots=True, frozen=True)
class TreePathIndex:
    path: tuple[int, ...] = ()

    def child(self, index: int) -> TreePathIndex:
        return TreePathIndex(path=self.path + (index,))

    def __str__(self) -> str:
        return ".".join(map(str, self.path))


T = TypeVar("T")
TraversalOrder = Literal["pre", "post"]


@final
class TreeTraversal(Generic[T]):
    """
    Depth-first traversal of a hierarchical tree structure.

    Supports pre-order and post-order traversal via either:
    - direct visiting (`__call__`)
    - iteration (`iterate()`)

    Example:
        traverser = HirachyTraverse(children=get_children, visit=print, order="pre")
        traverser(root)
    """

    __slots__ = ("stack", "children", "visit", "order")

    children: Callable[[T], list[T]]
    visit: Callable[[T], None]
    order: TraversalOrder
    stack: list[T]

    def __init__(
        self,
        children: Callable[[T], list[T]],
        visit: Callable[[T], None],
        order: TraversalOrder = "pre",
    ) -> None:
        self.children = children
        self.visit = visit
        self.order = order

    def __call__(self, root: T) -> None:
        match self.order:
            case "pre":
                self._visit_dfs_pre(root)
            case "post":
                self._visit_dfs_post(root)
            case _:
                raise ValueError(f"Unsupported traversal order: {self.order}")

    def _visit_dfs_pre(self, node: T) -> None:
        self.visit(node)

        for child in self.children(node):
            self._visit_dfs_pre(child)

    def _visit_dfs_post(self, node: T) -> None:
        for child in self.children(node):
            self._visit_dfs_post(child)

        self.visit(node)

    def iterate(self, node: T) -> Iterator[T]:
        match self.order:
            case "pre":
                yield from self.iterate_pre(node)
            case "post":
                yield from self.iterate_post(node)
            case _:
                raise ValueError(f"Unsupported traversal order: {self.order}")

    def iterate_pre(self, node: T) -> Iterator[T]:
        yield node

        for child in self.children(node):
            yield from self.iterate_pre(child)

    def iterate_post(self, node: T) -> Iterator[T]:
        for child in self.children(node):
            yield from self.iterate_post(child)

        yield node

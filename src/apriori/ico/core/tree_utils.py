from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar, final


@final
@dataclass(slots=True, frozen=True)
class TreePathIndex:
    path_index: tuple[int, ...] = ()

    def add_child(self, index: int) -> TreePathIndex:
        return TreePathIndex(path_index=self.path_index + (index,))

    def reverse(self) -> TreePathIndex:
        return TreePathIndex(path_index=self.path_index[::-1])

    def __str__(self) -> str:
        return ".".join(map(str, self.path_index))


T = TypeVar("T")
C = TypeVar("C")
TraversalOrder = Literal["pre", "post", "pre_post"]


@dataclass(slots=True)
class TraversalInfo(Generic[T, C]):
    node: T
    path: TreePathIndex
    total_siblings: int
    current_order: TraversalOrder = "pre"
    context: C | None = None
    visit_children: bool = True

    @property
    def is_last(self) -> bool:
        if self.is_root:
            return False
        return self.path.path_index[-1] == self.total_siblings - 1

    @property
    def is_root(self) -> bool:
        return len(self.path.path_index) == 0

    @property
    def node_path(self) -> tuple[T, TreePathIndex]:
        return (self.node, self.path)


LazySubtreeVisitingPolicy = Literal[
    "children_only",
    "subtree_only",
    "children_or_subtree",
    "subtree_or_children",
    "children_and_subtree",
]


@final
class TreeWalker(Generic[T, C]):
    __slots__ = (
        "get_children_fn",
        "get_lazy_subtree_fn",
        "filter_fn",
        "initial_context",
        "subtree_policy",
    )

    get_children_fn: Callable[[T], Sequence[T]]
    get_lazy_subtree_fn: Callable[[T], Sequence[T] | None] | None
    filter_fn: Callable[[T], bool] | None
    initial_context: C | None
    subtree_policy: LazySubtreeVisitingPolicy

    def __init__(
        self,
        get_children_fn: Callable[[T], Sequence[T]],
        get_lazy_subtree_fn: Callable[[T], Sequence[T] | None] | None = None,
        filter_fn: Callable[[T], bool] | None = None,
        initial_context: C | None = None,
        subtree_policy: LazySubtreeVisitingPolicy = "children_and_subtree",
    ) -> None:
        self.get_children_fn = get_children_fn
        self.get_lazy_subtree_fn = get_lazy_subtree_fn
        self.filter_fn = filter_fn
        self.initial_context = initial_context
        self.subtree_policy = subtree_policy

    def walk(
        self,
        root: T,
        visit_fn: Callable[[TraversalInfo[T, C]], None],
        order: TraversalOrder = "pre",
    ) -> None:
        for node_info in self.traverse(root, order=order):
            visit_fn(node_info)

    def iterate(self, root: T, order: TraversalOrder = "pre") -> Iterator[T]:
        for node_info in self.traverse(root, order=order):
            yield node_info.node

    def traverse(
        self, root: T, order: TraversalOrder = "pre"
    ) -> Iterator[TraversalInfo[T, C]]:
        if order in ["pre", "post", "pre_post"]:
            yield from self._traverse_dfs(root, order=order)
            return

        raise ValueError(f"Unsupported traversal order: {order}")

    def _traverse_dfs(
        self, root: T, order: TraversalOrder
    ) -> Iterator[TraversalInfo[T, C]]:
        stack = [TraversalInfo[T, C](root, TreePathIndex(), 0)]

        while len(stack) > 0:
            node_info = stack[-1]
            skip_node = self.filter_fn and not self.filter_fn(node_info.node)

            if node_info.current_order == "post":
                assert order in ["post", "pre_post"]
                # This is second visit in post-order traversal - pop the node from the stack
                stack.pop()
                if not skip_node:
                    yield node_info
                continue

            match order:
                case "pre":
                    # This is the first visit of the node in pre-order traversal -
                    # pop the node from stack
                    stack.pop()
                    if not skip_node:
                        yield node_info

                case "pre_post":
                    # This is the first visit in pre_post-order traversal -
                    # yield node and mark for post-visit. Do not pop for post-visit yet.
                    if not skip_node:
                        yield node_info
                    node_info.current_order = "post"

                case "post":
                    # Mark for post-visit
                    node_info.current_order = "post"

            # Get children and push them to the stack in reverse order
            if node_info.visit_children:
                children = self._get_all_children(node_info.node)
                total = len(children)
                stack += [
                    TraversalInfo(
                        node=child,
                        path=node_info.path.add_child(i),
                        total_siblings=total,
                        context=self.initial_context,
                    )
                    for i, child in enumerate(children[::-1])
                ]

    def _get_all_children(self, node: T) -> list[T]:
        children = list(self.get_children_fn(node))
        subtree = list(self._get_lazy_subtree(node))

        match self.subtree_policy:
            case "children_and_subtree":
                return children + subtree
            case "children_only":
                return children
            case "subtree_only":
                return subtree
            case "children_or_subtree":
                return children if len(children) > 0 else subtree
            case "subtree_or_children":
                return subtree if len(subtree) > 0 else children

    def _get_lazy_subtree(self, node: T) -> Sequence[T]:
        if self.get_lazy_subtree_fn is None:
            return []
        return self.get_lazy_subtree_fn(node) or []

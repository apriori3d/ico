from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar, final


@final
@dataclass(slots=True, frozen=True)
class TreePathIndex:
    """Immutable index representing a path through a tree structure.

    TreePathIndex provides a way to uniquely identify any node in a tree by
    storing the sequence of child indices from root to that node. This enables
    efficient tree navigation, node identification, and path-based operations.

    Example:
        >>> # Root node path
        >>> root = TreePathIndex()
        >>> assert str(root) == ""

        >>> # First child of root
        >>> child0 = root.add_child(0)
        >>> assert str(child0) == "0"

        >>> # Second child of first child
        >>> grandchild = child0.add_child(1)
        >>> assert str(grandchild) == "0.1"

        >>> # Create path directly
        >>> path = TreePathIndex.create(2, 1, 3)
        >>> assert str(path) == "2.1.3"

        >>> # Reverse path direction
        >>> reversed_path = path.reverse()
        >>> assert str(reversed_path) == "3.1.2"

    Attributes:
        path_index: Tuple of integers representing the path from root to node.

    Note:
        Immutable design ensures thread safety and enables safe sharing
        across different traversal contexts.
    """

    path_index: tuple[int, ...] = ()

    def add_child(self, index: int) -> TreePathIndex:
        """Create a new path by appending a child index.

        Args:
            index: The child index to append to the current path.

        Returns:
            New TreePathIndex representing the path to the specified child.

        Note:
            Returns a new instance - original path is unchanged due to immutability.
        """
        return TreePathIndex(path_index=self.path_index + (index,))

    def reverse(self) -> TreePathIndex:
        """Create a new path with reversed child indices.

        Returns:
            New TreePathIndex with path_index tuple reversed.

        Note:
            Useful for converting bottom-up paths to top-down or vice versa.
        """
        return TreePathIndex(path_index=self.path_index[::-1])

    def __str__(self) -> str:
        """Return dot-separated string representation of the path.

        Returns:
            String like "0.1.2" representing path through tree indices.

        Note:
            Empty path (root) returns empty string. Useful for debugging and display.
        """
        return ".".join(map(str, self.path_index))

    @staticmethod
    def create(*path_index: int) -> TreePathIndex:
        """Create a TreePathIndex directly from path components.

        Args:
            *path_index: Variable number of integers representing the path.

        Returns:
            New TreePathIndex with the specified path components.

        Example:
            >>> path = TreePathIndex.create(1, 2, 3)
            >>> # Equivalent to: TreePathIndex(path_index=(1, 2, 3))
        """
        return TreePathIndex(path_index=tuple(path_index))


T = TypeVar("T")
C = TypeVar("C")
TraversalOrder = Literal["pre", "post", "pre_post"]


@dataclass(slots=True)
class TraversalInfo(Generic[T, C]):
    """Information about a node during tree traversal.

    TraversalInfo encapsulates all relevant data about a node being visited
    during tree traversal, including its position, context, and traversal state.
    This enables flexible visitation patterns and context-aware tree processing.

    Generic Parameters:
        T: Type of tree nodes being traversed (e.g., str, IcoNode, dict, etc.).
        C: Type of context object passed through traversal (optional, can be None).

    Example:
        >>> # Basic traversal info for a root node
        >>> info = TraversalInfo(
        ...     node="root",
        ...     path=TreePathIndex(),
        ...     total_siblings=0
        ... )
        >>> assert info.is_root is True
        >>> assert info.is_last is False

        >>> # Child node info
        >>> child_info = TraversalInfo(
        ...     node="child",
        ...     path=TreePathIndex.create(2),
        ...     total_siblings=3
        ... )
        >>> assert child_info.is_last is True  # Index 2 of 3 siblings (0, 1, 2)
        >>> assert child_info.is_root is False

    Attributes:
        node: The tree node being visited.
        path: TreePathIndex representing the path to this node.
        total_siblings: Number of sibling nodes at this level.
        current_order: Current traversal order ("pre", "post", "pre_post").
        context: Optional context object passed through traversal.
        visit_children: Whether to visit children of this node.

    Note:
        The visit_children flag allows dynamic control over traversal depth
        during the traversal process.
    """

    node: T
    path: TreePathIndex
    total_siblings: int
    current_order: TraversalOrder = "pre"
    context: C | None = None
    visit_children: bool = True

    @property
    def is_last(self) -> bool:
        """Check if this is the last sibling at its level.

        Returns:
            True if this node is the last among its siblings, False otherwise.
            Root nodes always return False since they have no siblings.

        Note:
            Useful for formatting tree displays or handling last-child logic.
        """
        if self.is_root:
            return False
        return self.path.path_index[-1] == self.total_siblings - 1

    @property
    def is_root(self) -> bool:
        """Check if this is the root node.

        Returns:
            True if this node has no parent (empty path), False otherwise.

        Note:
            Root detection is based on path length - empty path means root.
        """
        return len(self.path.path_index) == 0

    @property
    def node_path(self) -> tuple[T, TreePathIndex]:
        """Get a tuple containing the node and its path.

        Returns:
            Tuple of (node, path) for convenient paired access.

        Note:
            Convenience property for operations that need both node and path together.
        """
        return (self.node, self.path)


@final
class TreeWalker(Generic[T, C]):
    """Generic tree traversal utility with configurable visitation patterns.

    TreeWalker provides flexible tree traversal capabilities with support for
    different traversal orders, context passing, and custom visitation functions.
    It abstracts the tree navigation logic while allowing customized node processing.

    Generic Parameters:
        T: Type of tree nodes (e.g., IcoNode, dict, custom classes, etc.).
        C: Type of context object for state passing (optional, can be None).

    Example:
        >>> # Simple tree structure using lists
        >>> tree = [
        ...     "root",
        ...     ["child1", ["grandchild1"], ["grandchild2"]],
        ...     ["child2"]
        ... ]

        >>> # Define how to extract children
        >>> def get_children(node: list | str) -> list:
        ...     return node[1:] if isinstance(node, list) else []

        >>> walker = TreeWalker(get_children)

        >>> # Collect all nodes in pre-order
        >>> nodes = list(walker.iterate(tree, order="pre"))
        >>> # Result: [tree, "child1", "grandchild1", "grandchild2", "child2"]

        >>> # Custom visitation with path info
        >>> def visit_node(info: TraversalInfo) -> None:
        ...     node_name = info.node[0] if isinstance(info.node, list) else info.node
        ...     print(f"Visited {node_name} at path {info.path}")

        >>> walker.walk(tree, visit_node, order="pre")
        >>> # Prints: Visited root at path
        >>> #         Visited child1 at path 0
        >>> #         Visited grandchild1 at path 0.0
        >>> #         ...

    Attributes:
        get_children_fn: Function that extracts children from a node.
        initial_context: Optional context passed to all traversal operations.

    Note:
        TreeWalker is used internally by ICO framework for traversing operator
        trees during inspection, visualization, and runtime operations.
    """

    __slots__ = (
        "get_children_fn",
        "initial_context",
    )

    get_children_fn: Callable[[T], Sequence[T]]
    initial_context: C | None

    def __init__(
        self,
        get_children_fn: Callable[[T], Sequence[T]],
        initial_context: C | None = None,
    ) -> None:
        """Initialize TreeWalker with child extraction function and optional context.

        Args:
            get_children_fn: Function that takes a node and returns its children.
                           Must return a sequence (list, tuple, etc.) of child nodes.
            initial_context: Optional context object passed to all traversal info.
                           Useful for maintaining state during traversal.

        Note:
            The get_children_fn should handle leaf nodes by returning empty sequences.
            Context is shared across all nodes in a single traversal operation.
        """
        self.get_children_fn = get_children_fn
        self.initial_context = initial_context

    def walk(
        self,
        root: T,
        visit_fn: Callable[[TraversalInfo[T, C]], None],
        order: TraversalOrder = "pre",
    ) -> None:
        """Traverse tree and apply visitor function to each node.

        Args:
            root: The root node to start traversal from.
            visit_fn: Function called for each node with TraversalInfo.
                     Can modify traversal behavior via info.visit_children.
            order: Traversal order - "pre", "post", or "pre_post".
                  "pre" visits node before children.
                  "post" visits node after children.
                  "pre_post" visits node both before and after children.

        Note:
            The visit_fn can control traversal by setting info.visit_children = False
            to skip subtrees. This enables conditional tree pruning during traversal.
        """
        for node_info in self.traverse(root, order=order):
            visit_fn(node_info)

    def iterate(self, root: T, order: TraversalOrder = "pre") -> Iterator[T]:
        """Iterate over all nodes in the tree.

        Args:
            root: The root node to start traversal from.
            order: Traversal order - "pre", "post", or "pre_post".

        Yields:
            T: Each node in the tree according to the specified order.

        Note:
            Convenience method that extracts just the nodes from full traversal info.
            Use traverse() if you need path information or traversal control.
        """
        for node_info in self.traverse(root, order=order):
            yield node_info.node

    def traverse(
        self, root: T, order: TraversalOrder = "pre"
    ) -> Iterator[TraversalInfo[T, C]]:
        """Traverse tree and yield detailed information for each node.

        Args:
            root: The root node to start traversal from.
            order: Traversal order - "pre", "post", or "pre_post".

        Yields:
            TraversalInfo[T, C]: Detailed information about each visited node,
                               including node, path, sibling info, and context.

        Raises:
            ValueError: If traversal order is not supported.

        Note:
            This is the most general traversal method providing full control
            and information about each node during traversal.
        """
        if order in ["pre", "post", "pre_post"]:
            yield from self._traverse_dfs(root, order=order)
            return

        raise ValueError(f"Unsupported traversal order: {order}")

    def _traverse_dfs(
        self, root: T, order: TraversalOrder
    ) -> Iterator[TraversalInfo[T, C]]:
        """Internal depth-first traversal implementation.

        Uses an explicit stack to perform iterative DFS traversal, supporting
        pre-order, post-order, and pre-post-order visitation patterns.

        Args:
            root: The root node to start traversal from.
            order: Traversal order determining when nodes are yielded.

        Yields:
            TraversalInfo[T, C]: Node information at appropriate traversal points.

        Note:
            Stack-based implementation avoids recursion limits and provides
            better control over traversal state management.
        """
        stack = [
            TraversalInfo[T, C](
                node=root,
                path=TreePathIndex(),
                total_siblings=0,
                context=self.initial_context,
            )
        ]

        while len(stack) > 0:
            node_info = stack[-1]

            if node_info.current_order == "post":
                assert order in ["post", "pre_post"]
                # This is second visit in post-order traversal - pop the node from the stack
                yield stack.pop()
                continue

            match order:
                case "pre":
                    # This is the first visit of the node in pre-order traversal -
                    # pop the node from stack
                    yield stack.pop()

                case "pre_post":
                    # This is the first visit in pre_post-order traversal -
                    # yield node and mark for post-visit. Do not pop for post-visit yet.
                    yield node_info
                    node_info.current_order = "post"

                case "post":
                    # Mark for post-visit
                    node_info.current_order = "post"

            # Get children and push them to the stack in reverse order
            if node_info.visit_children:
                children = list(self.get_children_fn(node_info.node))
                total = len(children)
                stack += [
                    TraversalInfo(
                        node=child,
                        path=node_info.path.add_child(i),
                        total_siblings=total,
                        context=self.initial_context,
                    )
                    for i, child in reversed(list(enumerate(children)))
                ]

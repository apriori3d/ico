from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator

from apriori.ico.core.node import IcoNode
from apriori.ico.core.runtime.node import (
    IcoRuntimeNode,
)

# ────────────────────────────────────────────────
# Runtime Discovery and Connection Utilities
# ────────────────────────────────────────────────


def discover_runtime_subtrees(flow: IcoNode) -> list[IcoRuntimeNode]:
    """Discover all runtime hosts within the given flow."""
    roots = OrderedDict[IcoRuntimeNode, None]()

    for runtime in discover_runtime_nodes(flow):
        if runtime.runtime_parent is None:
            roots[runtime] = None
            continue

        all_parents = list(runtime.iterate_parents())
        if len(all_parents) > 0:
            roots[all_parents[-1]] = None

    return list(roots.keys())


def discover_runtime_nodes(node: IcoNode) -> Iterator[IcoRuntimeNode]:
    """Discover all runtime hosts within the given flow."""

    if isinstance(node, IcoRuntimeNode):
        yield node

    for child in node.children:
        yield from discover_runtime_nodes(child)


def discover_and_connect_runtime_nodes(
    runtime: IcoRuntimeNode,
    flow: IcoNode,
) -> None:
    """Discover and connect all runtime hosts within the given flow."""
    for nested_runtime in discover_runtime_subtrees(flow):
        runtime.add_runtime_children(nested_runtime)

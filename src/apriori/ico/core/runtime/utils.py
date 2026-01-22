from __future__ import annotations

from collections import OrderedDict

from apriori.ico.core.node import IcoNode, create_flow_walker
from apriori.ico.core.runtime.node import (
    IcoRuntimeNode,
)

# ────────────────────────────────────────────────
# Runtime Nodes Discovery and Connection Utilities
# ────────────────────────────────────────────────


def discover_and_connect_runtime_nodes(
    runtime_node: IcoRuntimeNode, flow: IcoNode
) -> None:
    """Discover and connect all runtime hosts within the given flow."""
    for nested_runtime in _discover_runtime_subtrees(flow):
        runtime_node.add_runtime_children(nested_runtime)


def _discover_runtime_subtrees(flow: IcoNode) -> list[IcoRuntimeNode]:
    """Discover all runtime hosts within the given flow."""
    roots = OrderedDict[IcoRuntimeNode, None]()
    walker = create_flow_walker(visit_subflows=False)

    for node_info in walker.traverse(flow):
        if not isinstance(node_info.node, IcoRuntimeNode):
            continue

        if node_info.node.runtime_parent is None:
            roots[node_info.node] = None
            continue

        all_parents = list(node_info.node.iterate_parents())
        if len(all_parents) > 0:
            roots[all_parents[-1]] = None

    return list(roots.keys())

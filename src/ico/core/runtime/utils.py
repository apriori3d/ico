from __future__ import annotations

from collections import OrderedDict

from ico.core.node import IcoNode, create_flow_walker
from ico.core.runtime.node import (
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
    runtime_nodes = set[IcoRuntimeNode]()

    for node_info in create_flow_walker().traverse(flow):
        if not isinstance(node_info.node, IcoRuntimeNode):
            continue

        runtime_nodes.add(node_info.node)

        if node_info.node.runtime_parent is None:
            roots[node_info.node] = None
            continue

        all_parents = list(node_info.node.iterate_parents())
        if len(all_parents) > 0:
            roots[all_parents[-1]] = None

    root_nodes = list(roots.keys())

    # Check for external connections - it is possible if flow factory returns existing runtime nodes
    external_runtimes = [r for r in root_nodes if r not in runtime_nodes]
    if len(external_runtimes) > 0:
        raise RuntimeError(
            "Some runtime nodes already connected to external runtimes: "
            + ", ".join(str(r) for r in external_runtimes)
            + ". Possibly the flow factory returned existing runtime nodes."
        )
    return root_nodes

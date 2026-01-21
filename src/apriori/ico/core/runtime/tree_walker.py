from collections.abc import Sequence
from typing import TypeAlias, cast

from apriori.ico.core.runtime.command import IcoRuntimeCommand
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.core.tree_utils import TraversalInfo, TreeWalker

BroadcastTreeWalker: TypeAlias = TreeWalker[IcoRuntimeNode, IcoRuntimeCommand]
BroadcastTraversalInfo: TypeAlias = TraversalInfo[IcoRuntimeNode, IcoRuntimeCommand]


def create_broadcast_walker(
    command: IcoRuntimeCommand,
) -> BroadcastTreeWalker:
    return BroadcastTreeWalker(
        get_children_fn=lambda node: node.runtime_children,
        initial_context=command,
    )


RuntimeTreeWalker: TypeAlias = TreeWalker[IcoRuntimeNode, None]
RuntimeTraversalInfo: TypeAlias = TraversalInfo[IcoRuntimeNode, None]


def create_runtime_tree_walker(
    *,
    include_agent_worker: bool = True,
) -> RuntimeTreeWalker:
    """Get a tree walker for ICO runtime nodes."""
    return RuntimeTreeWalker(
        get_children_fn=lambda node: node.runtime_children,
        get_lazy_subtree_fn=_create_worker_node if include_agent_worker else None,
    )


def _create_worker_node(node: IcoRuntimeNode) -> Sequence[IcoRuntimeNode] | None:
    from apriori.ico.core.runtime.agent import IcoAgent

    if isinstance(node, IcoAgent):
        worker = cast(IcoRuntimeNode, node.worker_factory())
        return [worker]
    return None

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

from apriori.ico.core.node import IcoNode
from apriori.ico.core.runtime.agent import IcoAgent
from apriori.ico.core.tree_utils import TraversalInfo, TreeWalker


@dataclass(slots=True)
class FlowTreeWalkerContext:
    group_opened: bool = False


FlowTreeWalker: TypeAlias = TreeWalker[IcoNode, FlowTreeWalkerContext]
FlowTraversalInfo: TypeAlias = TraversalInfo[IcoNode, FlowTreeWalkerContext]


def create_flow_tree_walker(*, include_agent_subflows: bool = True) -> FlowTreeWalker:
    """Get a tree walker for ICO runtime nodes."""
    return FlowTreeWalker(
        get_children_fn=lambda node: node.children,
        get_lazy_subtree_fn=_create_agent_subflow if include_agent_subflows else None,
    )


def _create_agent_subflow(node: IcoNode) -> Sequence[IcoNode] | None:
    if isinstance(node, IcoAgent):
        # Special case for nodes with subflows - render the subflow inside the group
        subflow = cast(IcoNode, node.subflow_factory())
        return [subflow]

    return None

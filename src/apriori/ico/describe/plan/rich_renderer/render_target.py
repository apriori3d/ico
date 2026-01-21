from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from rich.text import Text

from apriori.ico.core.node import (
    HasSubflowFactory,
    IcoNode,
)
from apriori.ico.core.tree_utils import TraversalInfo, TreeWalker
from apriori.ico.describe.plan.rich_renderer.row_renderer import RowRenderer


class PlanRenderTarget(Protocol):
    def render_node(self, node_info: PlanTraversalInfo) -> None: ...

    def render_row(
        self,
        row_renderer: RowRenderer,
        node: IcoNode,
        indent: Text | None = None,
    ) -> None: ...

    def push_group_indent(self, indent: Text) -> None: ...

    def pop_group_indent(self) -> None: ...


# ────────────────────────────────────────────────
# Tree walker api
# ────────────────────────────────────────────────


@dataclass(slots=True)
class PlanTreeWalkerContext:
    group_opened: bool = False


PlanTreeWalker: TypeAlias = TreeWalker[IcoNode, PlanTreeWalkerContext]
PlanTraversalInfo: TypeAlias = TraversalInfo[IcoNode, PlanTreeWalkerContext]


def create_plan_tree_walker(include_subflows: bool) -> PlanTreeWalker:
    """Get a tree walker for ICO runtime nodes."""
    return PlanTreeWalker(
        get_children_fn=lambda node: node.children,
        get_lazy_subtree_fn=_create_node_subflow if include_subflows else None,
        subtree_policy="subtree_or_children",  # to show one instance of subflow if available (like in async steam)
    )


def _create_node_subflow(node: IcoNode) -> Sequence[IcoNode] | None:
    if isinstance(node, HasSubflowFactory) and node.subflow_factory is not None:
        return [node.subflow_factory()]

    return None

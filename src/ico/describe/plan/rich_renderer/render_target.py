from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from rich.text import Text

from ico.core.async_stream import IcoAsyncStream
from ico.core.node import (
    HasRemoteFlow,
    IcoNode,
)
from ico.core.tree_utils import TraversalInfo, TreeWalker
from ico.describe.plan.rich_renderer.row_renderer import RowRenderer


class PlanRenderTarget(Protocol):
    """Protocol defining plan renderer interface for node traversal."""

    def render_node(self, node_info: PlanTraversalInfo) -> None:
        """Render individual node during tree traversal."""
        ...

    def render_row(
        self,
        row_renderer: RowRenderer,
        node: IcoNode,
        indent: Text | None = None,
    ) -> None:
        """Render table row with optional indentation."""
        ...

    def push_group_indent(self, indent: Text) -> None:
        """Add group indentation level."""
        ...

    def pop_group_indent(self) -> None:
        """Remove group indentation level."""
        ...


# ────────────────────────────────────────────────
# Tree walker api
# ────────────────────────────────────────────────


@dataclass(slots=True)
class PlanTreeWalkerContext:
    """Context data for plan tree traversal state."""

    group_opened: bool = False


PlanTreeWalker: TypeAlias = TreeWalker[IcoNode, PlanTreeWalkerContext]
PlanTraversalInfo: TypeAlias = TraversalInfo[IcoNode, PlanTreeWalkerContext]


def create_plan_walker(expand_remote_flows: bool) -> PlanTreeWalker:
    """Create tree walker for ICO plan nodes with remote flow expansion."""

    def _get_children(node: IcoNode) -> Sequence[IcoNode]:
        # Flatten async stream pool to a single flow
        if isinstance(node, IcoAsyncStream) and node.pool_from_factory:
            return [node.children[0]]

        children = list(node.children)

        if expand_remote_flows and isinstance(node, HasRemoteFlow):
            factory = node.get_remote_flow_factory()
            flow = factory()
            children.append(flow)

        return children

    return PlanTreeWalker(get_children_fn=_get_children)

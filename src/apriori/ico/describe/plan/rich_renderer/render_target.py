from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from rich.text import Text

from apriori.ico.core.async_stream import IcoAsyncStream
from apriori.ico.core.node import (
    HasRemoteFlow,
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


def create_plan_walker(expand_remote_flows: bool) -> PlanTreeWalker:
    """Get a tree walker for ICO runtime nodes."""

    def _get_children(node: IcoNode) -> Sequence[IcoNode]:
        # Flatten async stream pool to a single flow
        if isinstance(node, IcoAsyncStream) and node.pool_from_factory:
            return [node.children[0]]

        children = list(node.children)

        if expand_remote_flows and isinstance(node, HasRemoteFlow):
            factory = node.get_remote_flow_factory()
            children.append(factory())

        return children

    return PlanTreeWalker(get_children_fn=_get_children)

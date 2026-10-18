from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from rich.text import Text

from ico.core.node import IcoNodeProtocol
from ico.core.tree_utils import TraversalInfo, TreeWalker
from ico.describe.plan.keys import plan_children
from ico.describe.plan.options import PlanRendererOptions
from ico.describe.plan.rich_renderer.row_renderer import RowRenderer


class PlanRenderTarget(Protocol):
    """Protocol defining plan renderer interface for node traversal."""

    def render_node(self, node_info: PlanTraversalInfo) -> None:
        """Render individual node during tree traversal."""
        ...

    def render_row(
        self,
        row_renderer: RowRenderer,
        node: IcoNodeProtocol,
        indent: Text | None = None,
        *,
        keyed: bool = True,
    ) -> None:
        """Render table row with optional indentation.

        The row gets the plan key of ``node`` (the node being rendered or
        one of its children) unless ``keyed`` is false (closing rows such as
        ``yield``) or a row already has that key.
        """
        ...

    def render_subtree(self, child: IcoNodeProtocol) -> None:
        """Render a child of the node being rendered (a branch, a part of an
        epoch) with its subtree. Custom renderers use it instead of walking
        the tree themselves, so every row keeps its plan key."""
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


PlanTreeWalker: TypeAlias = TreeWalker[IcoNodeProtocol, PlanTreeWalkerContext]
PlanTraversalInfo: TypeAlias = TraversalInfo[IcoNodeProtocol, PlanTreeWalkerContext]


def create_plan_walker(expand_remote_flows: bool) -> PlanTreeWalker:
    """Create tree walker for ICO plan nodes with remote flow expansion."""

    def _get_children(node: IcoNodeProtocol) -> Sequence[IcoNodeProtocol]:
        return plan_children(node, expand_remote_flows=expand_remote_flows)

    return PlanTreeWalker(get_children_fn=_get_children)


def flattened(options: PlanRendererOptions, node: IcoNodeProtocol) -> bool:
    """Whether the plan shows no row for ``node`` itself (chains, pipelines),
    only rows for its children."""
    return type(node) in options.flatten_node_type

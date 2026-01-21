from typing import Protocol

from rich.text import Text

from apriori.ico.core.node import IcoNode
from apriori.ico.core.tree_walker import FlowTraversalInfo
from apriori.ico.describe.plan.rich_renderer.row_renderer import RowRenderer


class PlanRenderTarget(Protocol):
    def render_node(self, node_info: FlowTraversalInfo) -> None: ...

    def render_row(
        self,
        row_renderer: RowRenderer,
        node: IcoNode,
        indent: Text | None = None,
    ) -> None: ...

    def push_group_indent(self, indent: Text) -> None: ...

    def pop_group_indent(self) -> None: ...

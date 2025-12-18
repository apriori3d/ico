from typing import Protocol

from rich.text import Text

from apriori.ico.core.node import IcoNode
from apriori.ico.describe.plan.rich_render.row_renderer import RowRenderer


class RenderTarget(Protocol):
    def render_node(self, node: IcoNode) -> None: ...

    def render_row(
        self,
        row_renderer: RowRenderer,
        node: IcoNode,
        indent: Text | None = None,
    ) -> None: ...

    def push_group_indent(self, indent: Text) -> None: ...

    def pop_group_indent(self) -> None: ...

from __future__ import annotations

from collections.abc import Callable

from rich.text import Text

from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.describe.rich_style import DescribeStyle
from apriori.ico.describe.rich_utils import (
    render_node_class,
)
from apriori.ico.describe.runtime.options import (
    RuntimeRendererColumn,
    RuntimeRendererOptions,
)
from apriori.ico.describe.runtime.rich_renderer.utils import get_state_color
from apriori.ico.describe.utils import match_icon


class RuntimeRowRenderer:
    options: RuntimeRendererOptions
    _column_renderer: dict[RuntimeRendererColumn, Callable[[IcoRuntimeNode], Text]]

    def __init__(self, options: RuntimeRendererOptions) -> None:
        self.options = options
        self._column_renderer: dict[
            RuntimeRendererColumn, Callable[[IcoRuntimeNode], Text]
        ] = {
            "Tree": self.render_tree_column,
            "State": self.render_state_column,
            "Name": self.render_name_column,
        }

    def render(self, node: IcoRuntimeNode, column: RuntimeRendererColumn) -> Text:
        return self._column_renderer[column](node)

    def render_tree_column(self, node: IcoRuntimeNode) -> Text:
        text = render_node_class(node, options=self.options)

        if self.options.show_node_icons:
            icon = match_icon(self.options.node_icons, node)
            if icon:
                text = Text(icon) + text

        return text

    def render_name_column(self, node: IcoRuntimeNode) -> Text:
        return Text(node.runtime_name or "")

    def render_type_column(self, node: IcoRuntimeNode) -> Text:
        return Text(node.__class__.__name__, style=DescribeStyle.dimmed.value)

    def render_signature_column(self, node: IcoRuntimeNode) -> Text:
        return Text("")

    def render_state_column(self, node: IcoRuntimeNode) -> Text:
        color = get_state_color(node.state)
        return Text(f"<{node.state}> ", style=color)

from typing import Protocol

from rich.text import Text

from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.core.runtime.tree_walker import RuntimeTraversalInfo
from apriori.ico.describe.runtime.rich_renderer.row_renderer import RuntimeRowRenderer


class RuntimeTreeRenderTarget(Protocol):
    def render_node(self, node_info: RuntimeTraversalInfo) -> None: ...

    def render_row(
        self,
        row_renderer: RuntimeRowRenderer,
        node: IcoRuntimeNode,
        indent: Text | None = None,
    ) -> None: ...

    def push_group_indent(self, indent: Text) -> None: ...

    def pop_group_indent(self) -> None: ...

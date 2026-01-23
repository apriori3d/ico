from typing import TypeAlias

from rich.console import Console
from rich.table import Table
from rich.text import Text

from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.node import (
    IcoRuntimeNode,
    RuntimeTraversalInfo,
)
from apriori.ico.core.runtime.state import (
    IcoRuntimeState,
    IcoStateEvent,
    IcoStateRequestCommand,
)
from apriori.ico.core.tree_utils import TreePathIndex
from apriori.ico.describe.rich_style import DescribeStyle
from apriori.ico.describe.runtime.options import RuntimeRendererOptions
from apriori.ico.describe.runtime.rich_renderer.custom_renderer import (
    RuntimeCustomRenderer,
)
from apriori.ico.describe.runtime.rich_renderer.render_target import (
    create_runtime_renderer_walker,
)
from apriori.ico.describe.runtime.rich_renderer.renderer_registry import (
    RendererRegistry,
)
from apriori.ico.describe.runtime.rich_renderer.row_renderer import (
    RuntimeRowRenderer,
)
from apriori.ico.describe.utils import import_all_renderers

RendererTypes: TypeAlias = RuntimeRowRenderer | RuntimeCustomRenderer


class RuntimeTreeRenderer(IcoRuntimeNode):
    options: RuntimeRendererOptions
    console: Console

    _table: Table | None
    _group_indents: list[Text]
    _default_renderer: RuntimeRowRenderer
    _selected_renderers: dict[type[IcoRuntimeNode], RendererTypes]
    _node_states: dict[TreePathIndex, IcoRuntimeState]

    def __init__(
        self,
        options: RuntimeRendererOptions | None = None,
        console: Console | None = None,
    ) -> None:
        super().__init__()

        self.options = options or RuntimeRendererOptions()
        self.console = console or Console()

        self._default_renderer = RuntimeRowRenderer(options=self.options)
        self._table = None
        self._group_indents = []
        self._selected_renderers = {}
        self._node_states = {}

        for path in self.options.renderers_paths:
            import_all_renderers(path)

    def _create_table(self) -> Table:
        table = Table(show_lines=False, show_edge=False, box=None)
        for column in self.options.columns:
            table.add_column(column, no_wrap=True)
        return table

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        super().on_event(event)

        if isinstance(event, IcoStateEvent) and len(event.trace.path_index) > 0:
            runtime_index = TreePathIndex(
                path_index=event.trace.reverse().path_index[1:]
            )
            self._node_states[runtime_index] = event.state

    def render(self, root: IcoRuntimeNode) -> None:
        # Collect states via event API
        self._node_states = {}
        self.add_runtime_children(root)
        self.broadcast_command(IcoStateRequestCommand.create())

        self._table = self._create_table()

        tree_walker = create_runtime_renderer_walker(
            expand_remote_runtimes=self.options.expand_agents
        )
        tree_walker.walk(
            root,
            visit_fn=self.render_node,
            order="pre_post",  # Need post order to close subtree groups
        )

        self.console.rule(f"[bold blue]Runtime tree: {root}", style="dim blue")
        self.console.print(self._table)

        self.remove_runtime_child(root)

    def _select_renderer(
        self,
        node: IcoRuntimeNode,
    ) -> RendererTypes:
        renderer = self._selected_renderers.get(type(node))
        if renderer is not None:
            return renderer

        renderer_class = RendererRegistry.get(type(node))

        if renderer_class is not None:
            renderer = renderer_class(self.options)
        else:
            renderer = self._default_renderer

        self._selected_renderers[type(node)] = renderer

        return renderer

    def render_node(self, node_info: RuntimeTraversalInfo) -> None:
        node = node_info.node
        if node_info.path in self._node_states:
            node.state_model.state = self._node_states[node_info.path]

        if node_info.current_order == "pre":
            renderer = self._select_renderer(node)

            # Custom rendering logic
            if isinstance(renderer, RuntimeCustomRenderer):
                renderer.render(self, node)
                return

            if node_info.is_root:
                branch = None
            else:
                branch = Text(
                    "└──" if node_info.is_last else "├──",
                    style=DescribeStyle.tree.value,
                )

            self.render_row(renderer, node, indent=branch)

            # Open subtree group for children.
            if not node_info.is_root:
                indent = Text(
                    "    " if node_info.is_last else "│   ",
                    style=DescribeStyle.tree.value,
                )
                self.push_group_indent(indent)

        elif node_info.current_order == "post" and not node_info.is_root:
            # Close the subtree group
            self.pop_group_indent()

    def push_group_indent(self, indent: Text) -> None:
        self._group_indents.append(indent)

    def pop_group_indent(self) -> None:
        self._group_indents.pop()

    def render_row(
        self,
        row_renderer: RuntimeRowRenderer,
        node: IcoRuntimeNode,
        indent: Text | None = None,
    ) -> None:
        assert self._table is not None

        columns = [row_renderer.render(node, column) for column in self.options.columns]

        indentation = self._gather_indentation(indent)
        if indentation:
            columns[0] = indentation + columns[0]

        self._table.add_row(*columns)

    def _gather_indentation(self, indent: Text | None = None) -> Text | None:
        all_indents = self._group_indents.copy()

        if indent is not None:
            all_indents.append(indent)

        full_indent = None

        for group_indent in all_indents:
            if full_indent is None:
                full_indent = group_indent
            else:
                full_indent += group_indent

        return full_indent

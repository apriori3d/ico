from typing import TypeAlias

from rich.console import Console
from rich.table import Table
from rich.text import Text

from apriori.ico.core.node import IcoNode
from apriori.ico.core.types import HasSubflowFactory
from apriori.ico.describe.plan.options import (
    PlanRendererOptions,
)
from apriori.ico.describe.plan.rich_renderer.custom_renderer import CustomRenderer
from apriori.ico.describe.plan.rich_renderer.group_renderer import (
    GroupRenderer,
)
from apriori.ico.describe.plan.rich_renderer.renderer_registry import (
    RendererRegistry,
)
from apriori.ico.describe.plan.rich_renderer.row_renderer import RowRenderer
from apriori.ico.describe.rich_style import DescribeStyle
from apriori.ico.describe.utils import import_all_renderers

RendererTypes: TypeAlias = RowRenderer | GroupRenderer | CustomRenderer


class PlanRenderer:
    options: PlanRendererOptions
    console: Console

    _table: Table | None
    _group_indents: list[Text]
    _default_renderer: RowRenderer
    _default_group_renderer: GroupRenderer
    _selected_renderers: dict[type[IcoNode], RendererTypes]

    def __init__(
        self,
        options: PlanRendererOptions | None = None,
        console: Console | None = None,
    ) -> None:
        self.options = options or PlanRendererOptions()
        self.console = console or Console()

        self._default_renderer = RowRenderer(options=self.options)
        self._default_group_renderer = GroupRenderer(options=self.options)
        self._table = None
        self._group_indents = []
        self._selected_renderers = {}

        for path in self.options.renderers_paths:
            import_all_renderers(path)

    def _create_table(self) -> Table:
        table = Table(show_lines=False, show_edge=False, box=None)
        for column in self.options.columns:
            table.add_column(column, no_wrap=True)
        return table

    def render(self, root: IcoNode) -> None:
        self._table = self._create_table()

        self.render_node(root)
        self.console.rule(f"[bold blue]Flow plan: {root}", style="dim blue")
        self.console.print(self._table)

    def _select_renderer(
        self,
        node: IcoNode,
    ) -> RendererTypes:
        renderer = self._selected_renderers.get(type(node))
        if renderer is not None:
            return renderer

        renderer_class = RendererRegistry.get(type(node))

        if renderer_class is not None:
            renderer = renderer_class(self.options)
        else:
            renderer = (
                self._default_renderer
                if len(node.children) == 0
                else self._default_group_renderer
            )

        self._selected_renderers[type(node)] = renderer
        return renderer

    def render_node(self, node: IcoNode) -> None:
        # Do not render nodes responsible for flow structure - focus on the actual order of operations
        if (
            type(node) in self.options.flatten_node_type
        ):  # or isinstance(node, IcoMonitor):
            for child in node.children:
                self.render_node(child)
            return

        renderer = self._select_renderer(node)

        # Custom rendering logic (for IcoEpoch, etc)
        if isinstance(renderer, CustomRenderer):
            renderer.render(self, node)
            return

        if isinstance(renderer, RowRenderer):
            # Render node as a single row element
            self.render_row(renderer, node)
            return

        # Render subflow as a group element
        assert isinstance(renderer, GroupRenderer)

        # Render node info if required
        if renderer.node_renderer is not None:
            self.render_row(renderer.node_renderer, node)

        if not self.options.expand_subflows:
            return

        if (
            isinstance(node, HasSubflowFactory)
            and not self.options.expand_subflow_factories
        ):
            return

        # Expand subflow

        # Render header
        self.render_row(
            renderer.header_renderer,
            node,
            indent=Text("╭── ", style=DescribeStyle.group.value),
        )

        self.push_group_indent(Text("│   ", style=DescribeStyle.group.value))

        if isinstance(node, HasSubflowFactory):
            # Special case for nodes with subflows - render the subflow inside the group
            subflow = node.subflow_factory()
            self.render_node(subflow)
        else:
            for child in node.children:
                self.render_node(child)

        self.pop_group_indent()

        self.render_row(
            renderer.footer_renderer,
            node,
            indent=Text("╰─▸ ", style=DescribeStyle.group.value),
        )

    def push_group_indent(self, indent: Text) -> None:
        self._group_indents.append(indent)

    def pop_group_indent(self) -> None:
        self._group_indents.pop()

    def render_row(
        self,
        row_renderer: RowRenderer,
        node: IcoNode,
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

from __future__ import annotations

from typing import TypeAlias

from rich.console import Console
from rich.table import Table
from rich.text import Text

from ico.core.node import (
    IcoNode,
)
from ico.describe.plan.options import (
    PlanRendererOptions,
)
from ico.describe.plan.rich_renderer.custom_renderer import CustomRenderer
from ico.describe.plan.rich_renderer.group_renderer import (
    GroupRenderer,
)
from ico.describe.plan.rich_renderer.render_target import (
    PlanTraversalInfo,
    PlanTreeWalkerContext,
    create_plan_walker,
)
from ico.describe.plan.rich_renderer.renderer_registry import (
    RendererRegistry,
)
from ico.describe.plan.rich_renderer.row_renderer import RowRenderer
from ico.describe.rich_style import DescribeStyle
from ico.describe.utils import import_all_renderers

RendererTypes: TypeAlias = RowRenderer | GroupRenderer | CustomRenderer


class PlanRenderer:
    """
    Main plan rendering engine using Rich console for ICO flow visualization.

    Renders ICO node trees as formatted tables with configurable columns,
    automatic renderer selection, and subflow grouping support.
    """

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
        """Initialize plan renderer with options and console."""
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
        """Create Rich table with configured columns."""
        table = Table(show_lines=False, show_edge=False, box=None)
        for column in self.options.columns:
            table.add_column(column, no_wrap=True)
        return table

    def render(self, root: IcoNode) -> None:
        """Render complete ICO node tree to console."""
        self._table = self._create_table()

        tree_walker = create_plan_walker(
            expand_remote_flows=self.options.show_remote_flows
        )
        tree_walker.walk(
            root,
            visit_fn=self.render_node,
            order="pre_post",  # Need post order to close groups
        )

        self.console.rule(f"[bold blue]Flow plan: {root.name}", style="dim blue")
        self.console.print(self._table)

    def _select_renderer(self, node: IcoNode) -> RendererTypes:
        """Select appropriate renderer for node type via registry lookup."""
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

    def render_node(self, node_info: PlanTraversalInfo) -> None:
        """Render individual node using pre/post order traversal logic."""
        node = node_info.node

        if node_info.current_order == "pre":
            # Main render logic - render node before its children.

            # Do not render nodes responsible for flow structure - focus on the actual order of operations
            if type(node_info.node) in self.options.flatten_node_type:
                return

            renderer = self._select_renderer(node)

            # Custom rendering logic (for IcoEpoch, etc)
            if isinstance(renderer, CustomRenderer):
                renderer.render(self, node)
                node_info.visit_children = False
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

            if self.options.show_remote_flows:
                # Render header
                self.render_row(
                    renderer.header_renderer,
                    node,
                    indent=Text("╭── ", style=DescribeStyle.group.value),
                )

                # Add branch for group children
                self.push_group_indent(Text("│   ", style=DescribeStyle.group.value))

                # Set context to close group on post-order visit
                node_info.context = PlanTreeWalkerContext(group_opened=True)

        elif (
            node_info.current_order == "post"
            and node_info.context is not None
            and node_info.context.group_opened
        ):
            # The group was open in pre-order visit and here it should be closed.
            renderer = self._select_renderer(node)
            assert isinstance(renderer, GroupRenderer)

            self.pop_group_indent()

            self.render_row(
                renderer.footer_renderer,
                node,
                indent=Text("╰─▸ ", style=DescribeStyle.group.value),
            )

    def push_group_indent(self, indent: Text) -> None:
        """Add indentation level for nested groups."""
        self._group_indents.append(indent)

    def pop_group_indent(self) -> None:
        """Remove last indentation level when exiting group."""
        self._group_indents.pop()

    def render_row(
        self,
        row_renderer: RowRenderer,
        node: IcoNode,
        indent: Text | None = None,
    ) -> None:
        """Render single table row with proper indentation."""
        assert self._table is not None

        columns = [row_renderer.render(node, column) for column in self.options.columns]

        indentation = self._gather_indentation(indent)
        if indentation:
            columns[0] = indentation + columns[0]

        self._table.add_row(*columns)

    def _gather_indentation(self, indent: Text | None = None) -> Text | None:
        """Combine all group indentations into single Text object."""
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

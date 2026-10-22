from __future__ import annotations

from typing import TypeAlias, cast

from rich.console import Console
from rich.table import Table
from rich.text import Text

from ico.core.node import (
    IcoNodeProtocol,
)
from ico.core.plan_keys import ROOT_KEY, child_key, path_key
from ico.describe.plan.options import (
    PlanRendererOptions,
)
from ico.describe.plan.rich_renderer.custom_renderer import CustomRenderer
from ico.describe.plan.rich_renderer.group_renderer import (
    GroupRenderer,
)
from ico.describe.plan.rich_renderer.monitors_renderer import (
    OwnedMonitorsRenderer,
    owns_monitors,
)
from ico.describe.plan.rich_renderer.render_target import (
    PlanTraversalInfo,
    PlanTreeWalkerContext,
    create_plan_walker,
)
from ico.describe.plan.rich_renderer.renderer_registry import (
    select_renderer,
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
    _walk_keys: list[str]
    """Plan key of the root of every walk in progress (the main walk and the
    subtrees of custom renderers)."""
    _rendering: list[tuple[IcoNodeProtocol, str]]
    """The nodes being rendered, innermost last, with their plan keys."""
    _used_keys: set[str]
    _default_renderer: RowRenderer
    _default_group_renderer: GroupRenderer
    _monitors_renderer: OwnedMonitorsRenderer
    _selected_renderers: dict[type[IcoNodeProtocol], RendererTypes]

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
        self._monitors_renderer = OwnedMonitorsRenderer(self.options)
        self._table = None
        self._group_indents = []
        self._walk_keys = []
        self._rendering = []
        self._used_keys = set()
        self._selected_renderers = {}

        for path in self.options.renderers_paths:
            import_all_renderers(path)

    def _create_table(self) -> Table:
        """Create Rich table with configured columns."""
        table = Table(show_lines=False, show_edge=False, box=None)
        for column in self.options.columns:
            table.add_column(column, no_wrap=True)
        return table

    def render(self, root: IcoNodeProtocol) -> None:
        """Render complete ICO node tree to console."""
        self._table = self._create_table()
        self.walk(root)

        self.console.rule(f"[bold blue]Flow plan: {root.name}", style="dim blue")
        self.console.print(self._table)

    def walk(self, root: IcoNodeProtocol, key: str = ROOT_KEY) -> None:
        """Render ``root`` (keyed ``key``) and its subtree."""
        if not self._walk_keys:
            self._used_keys = set()
        self._walk_keys.append(key)
        try:
            create_plan_walker(expand_remote_flows=self.options.show_remote_flows).walk(
                root,
                visit_fn=self.render_node,
                order="pre_post",  # Need post order to close groups
            )
        finally:
            self._walk_keys.pop()

    def render_subtree(self, child: IcoNodeProtocol) -> None:
        """Render a child of the node being rendered with its subtree."""
        node, key = self._rendering[-1]
        index = _index(node.children, child)
        if index is None:
            raise ValueError(f"{child!r} is not a child of {node!r}")
        self.walk(child, child_key(key, index))

    def _key_of(self, node: IcoNodeProtocol) -> str | None:
        """The plan key of the node being rendered or of one of its children."""
        if not self._rendering:
            return None
        current, key = self._rendering[-1]
        if node is current:
            return key
        index = _index(current.children, node)
        return child_key(key, index) if index is not None else None

    def _select_renderer(self, node: IcoNodeProtocol) -> RendererTypes:
        """Select appropriate renderer for node type via registry lookup;
        an operator that owns monitors (its children, all ``IcoMonitor``)
        without a renderer of its own is a row with its monitors below."""
        if select_renderer(type(node)) is None and owns_monitors(node):
            return self._monitors_renderer
        renderer = self._selected_renderers.get(type(node))
        if renderer is not None:
            return renderer

        renderer_class = select_renderer(type(node))

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
        if node_info.current_order == "pre":
            key = path_key(self._walk_keys[-1], node_info.path)
            self._rendering.append((node_info.node, key))
        try:
            self._render_node(node_info)
        finally:
            if node_info.current_order == "post":
                self._rendering.pop()

    def _render_node(self, node_info: PlanTraversalInfo) -> None:
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
                keyed=False,
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
        node: IcoNodeProtocol,
        indent: Text | None = None,
        *,
        keyed: bool = True,
    ) -> None:
        """Render single table row with proper indentation.

        The row gets the plan key of ``node`` unless ``keyed`` is false or
        another row already has it.
        """
        columns = [row_renderer.render(node, column) for column in self.options.columns]

        indentation = self._gather_indentation(indent)
        if indentation:
            columns[0] = indentation + columns[0]

        key = self._key_of(node) if keyed else None
        if key is not None and key in self._used_keys:
            key = None
        if key is not None:
            self._used_keys.add(key)
        self.add_row(columns, key)

    def add_row(self, columns: list[Text], key: str | None) -> None:
        """Add a rendered row; ``key`` is its node's plan key, if any."""
        assert self._table is not None
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


def _index(children: object, child: IcoNodeProtocol) -> int | None:
    """The position of ``child`` among ``children`` (by identity)."""
    for i, candidate in enumerate(cast(list[IcoNodeProtocol], children)):
        if candidate is child:
            return i
    return None

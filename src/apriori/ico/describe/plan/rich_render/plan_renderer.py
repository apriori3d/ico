from typing import TypeAlias

from rich.console import Console
from rich.table import Table
from rich.text import Text

from apriori.ico.core.node import IcoNode
from apriori.ico.core.sink import IcoSink
from apriori.ico.core.source import IcoSource
from apriori.ico.core.stream import IcoStream
from apriori.ico.core.types import HasSubflowFactory
from apriori.ico.describe.plan.rich_render.group_renderer import (
    DefaultGroupRenderer,
    GroupRenderer,
)
from apriori.ico.describe.plan.rich_render.mp_process import MPProcessRenderer
from apriori.ico.describe.plan.rich_render.options import (
    RenderOptions,
)
from apriori.ico.describe.plan.rich_render.row_renderer import RowRenderer
from apriori.ico.describe.plan.rich_render.sink import IcoSinkRender
from apriori.ico.describe.plan.rich_render.source import IcoSourceRender
from apriori.ico.describe.plan.rich_render.stream import IcoStreamRenderer
from apriori.ico.runtime.agent.mp_process.mp_process import MPProcess

RendererTypes: TypeAlias = RowRenderer | GroupRenderer


class PlanRenderer:
    options: RenderOptions
    console: Console
    renderers: dict[type[IcoNode], RendererTypes]

    _table: Table
    _group_indents: list[Text]
    _default_renderer: RowRenderer
    _default_group_renderer: DefaultGroupRenderer

    def __init__(
        self,
        options: RenderOptions | None = None,
        console: Console | None = None,
        renderers: dict[type[IcoNode], RendererTypes] | None = None,
    ) -> None:
        self.options = options or RenderOptions()
        self.console = console or Console()

        self.renderers = renderers or {
            IcoSource: IcoSourceRender(options=self.options),
            IcoSink: IcoSinkRender(options=self.options),
            IcoStream: IcoStreamRenderer(options=self.options),
            MPProcess: MPProcessRenderer(options=self.options),
        }
        self._default_renderer = RowRenderer(options=self.options)
        self._default_group_renderer = DefaultGroupRenderer(options=self.options)

        table = Table(show_lines=False, show_edge=False, box=None)

        for column in self.options.columns:
            table.add_column(column, no_wrap=True)

        self._table = table

        self._group_indents = []

    def render(self, root: IcoNode) -> None:
        self(root)
        self.console.rule(f"[bold blue]Flow plan: {root}", style="dim blue")
        self.console.print(self._table)

    # ──────────── RichRenderTarget Protocol  ────────────

    def __call__(self, node: IcoNode) -> None:
        # Do not render nodes responsible for flow structure - focus on the actual order of operations
        if type(node) in self.options.flatten_node_type:
            for child in node.children:
                self(child)
            return

        renderer = self.renderers.get(type(node))
        if renderer is None:
            renderer = (
                self._default_renderer
                if len(node.children) == 0
                else self._default_group_renderer
            )

        # Render node as a single row element
        if isinstance(renderer, RowRenderer):
            self._render_row(renderer, node)
            return

        # Render subflow as a group element
        self._render_row(renderer.node_renderer, node)

        if not self.options.expand_subflows:
            return

        if (
            isinstance(node, HasSubflowFactory)
            and not self.options.expand_subflow_factories
        ):
            return

        self._render_row(renderer.entry_renderer, node, indent=Text("╭── "))
        self._group_indents.append(Text("│   "))

        if isinstance(node, HasSubflowFactory):
            # Special case for nodes with subflows - render the subflow inside the group
            subflow = node.flow_factory()
            self(subflow)
        else:
            for child in node.children:
                self(child)

        self._group_indents.pop()
        self._render_row(renderer.exit_renderer, node, indent=Text("╰─▸ "))

    def _render_row(
        self,
        row_renderer: RowRenderer,
        node: IcoNode,
        indent: Text | None = None,
    ) -> None:
        columns = [row_renderer(node, column) for column in self.options.columns]

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

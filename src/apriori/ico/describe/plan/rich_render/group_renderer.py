from dataclasses import replace

from rich.text import Text

from apriori.ico.core.node import IcoNode
from apriori.ico.describe.plan.rich_render.options import RenderOptions
from apriori.ico.describe.plan.rich_render.row_renderer import RowRenderer
from apriori.ico.describe.plan.rich_render.utils import PlanStyle


class GroupRenderer:
    node_renderer: RowRenderer | None
    header_renderer: RowRenderer
    footer_renderer: RowRenderer


class GroupPartRenderer(RowRenderer):
    _flow_text: Text
    _render_node: bool

    def __init__(
        self,
        flow_text: Text,
        options: RenderOptions,
        render_node: bool = False,
    ) -> None:
        super().__init__(options)

        self._flow_text = flow_text
        self._render_node = render_node

    def render_flow_column(self, node: IcoNode) -> Text:
        return (
            self._flow_text + super().render_flow_column(node)
            if self._render_node
            else self._flow_text
        )

    def render_name_column(self, node: IcoNode) -> Text:
        return super().render_name_column(node) if self._render_node else Text("")

    def render_type_column(self, node: IcoNode) -> Text:
        return super().render_type_column(node) if self._render_node else Text("")

    def render_state_column(self, node: IcoNode) -> Text:
        return super().render_state_column(node) if self._render_node else Text("")


class DefaultGroupRenderer(GroupRenderer):
    def __init__(self, options: RenderOptions) -> None:
        self.node_renderer = RowRenderer(options)

        self.header_renderer = GroupPartRenderer(
            flow_text=Text("Subflow Entry", style=PlanStyle.dimmed.value),
            options=replace(options, signature_format="Input"),
        )
        self.footer_renderer = GroupPartRenderer(
            flow_text=Text("Subflow Exit", style=PlanStyle.dimmed.value),
            options=replace(options, signature_format="Output"),
        )

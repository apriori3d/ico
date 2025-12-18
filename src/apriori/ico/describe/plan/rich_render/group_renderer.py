from dataclasses import replace

from rich.text import Text

from apriori.ico.describe.plan.options import RenderOptions
from apriori.ico.describe.plan.rich_render.row_renderer import RowRenderer
from apriori.ico.describe.plan.rich_render.utils import PlanStyle


class GroupRenderer:
    node_renderer: RowRenderer | None
    header_renderer: RowRenderer
    footer_renderer: RowRenderer

    def __init__(
        self,
        options: RenderOptions,
        node_renderer: RowRenderer | None = None,
        header_renderer: RowRenderer | None = None,
        footer_renderer: RowRenderer | None = None,
    ) -> None:
        self.node_renderer = node_renderer

        self.header_renderer = header_renderer or RowRenderer(
            flow_column_prefix=Text("Subflow Entry", style=PlanStyle.dimmed.value),
            options=replace(options, signature_format="Input"),
        )
        self.footer_renderer = footer_renderer or RowRenderer(
            flow_column_prefix=Text("Subflow Exit", style=PlanStyle.dimmed.value),
            options=replace(options, signature_format="Output"),
        )


# class GroupPartRenderer(RowRenderer):
#     _flow_text: Text
#     _node_renderer: RowRenderer | None

#     def __init__(
#         self,
#         flow_text: Text,
#         options: RenderOptions,
#         node_renderer: RowRenderer | None = None,
#     ) -> None:
#         super().__init__(options=options)

#         self._flow_text = flow_text
#         self._node_renderer = node_renderer

#     def render_flow_column(self, node: IcoNode) -> Text:
#         return (
#             self._flow_text + self._node_renderer.render_flow_column(node)
#             if self._node_renderer
#             else self._flow_text
#         )

#     def render_signature_column(self, node: IcoNode) -> Text:
#         return (
#             self._node_renderer.render_signature_column(node)
#             if self._node_renderer
#             else Text("")
#         )

#     def render_name_column(self, node: IcoNode) -> Text:
#         return (
#             self._node_renderer.render_name_column(node)
#             if self._node_renderer
#             else Text("")
#         )

#     def render_type_column(self, node: IcoNode) -> Text:
#         return (
#             self._node_renderer.render_type_column(node)
#             if self._node_renderer
#             else Text("")
#         )

#     def render_state_column(self, node: IcoNode) -> Text:
#         return (
#             self._node_renderer.render_state_column(node)
#             if self._node_renderer
#             else Text("")
#         )

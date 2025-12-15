from typing import Literal

from rich.text import Text

from apriori.ico.core.node import IcoNode
from apriori.ico.describe.plan.rich_render.options import RenderOptions
from apriori.ico.describe.plan.rich_render.row_renderer import RowRenderer
from apriori.ico.describe.plan.rich_render.utils import PlanStyle
from apriori.ico.inspect.signature import format_ico_type, infer_signature


class GroupRenderer:
    node_renderer: RowRenderer
    entry_renderer: RowRenderer
    exit_renderer: RowRenderer
    pass


GroupPart = Literal["entry", "exit"]


class GroupPartRenderer(RowRenderer):
    _flow_text: Text
    _group_part: GroupPart

    def __init__(
        self,
        flow_text: Text,
        group_part: GroupPart,
        options: RenderOptions,
    ) -> None:
        super().__init__(options)
        self._flow_text = flow_text
        self._group_part = group_part

    def render_flow_column(self, node: IcoNode) -> Text:
        return self._flow_text

    def render_name_column(self, node: IcoNode) -> Text:
        return Text("")

    def render_signature_column(self, node: IcoNode) -> Text:
        signature = infer_signature(node)

        if self._group_part == "entry":
            return Text(format_ico_type(signature.i), style=PlanStyle.signature.value)

        return Text(format_ico_type(signature.o), style=PlanStyle.signature.value)

    def render_type_column(self, node: IcoNode) -> Text:
        return Text("")

    def render_state_column(self, node: IcoNode) -> Text:
        return Text("")


class DefaultGroupRenderer(GroupRenderer):
    def __init__(self, options: RenderOptions) -> None:
        self.node_renderer = RowRenderer(options)

        self.entry_renderer = GroupPartRenderer(
            flow_text=Text("Subflow Entry", style=PlanStyle.dimmed.value),
            group_part="entry",
            options=options,
        )
        self.exit_renderer = GroupPartRenderer(
            flow_text=Text("Subflow Exit", style=PlanStyle.dimmed.value),
            group_part="exit",
            options=options,
        )

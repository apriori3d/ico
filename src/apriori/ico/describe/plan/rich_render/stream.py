from dataclasses import replace

from rich.text import Text

from apriori.ico.core.node import IcoNode
from apriori.ico.core.operator import operator
from apriori.ico.describe.plan.rich_render.group_renderer import (
    GroupPartRenderer,
    GroupRenderer,
)
from apriori.ico.describe.plan.rich_render.options import RenderOptions
from apriori.ico.describe.plan.rich_render.utils import PlanStyle


class IcoStreamRenderer(GroupRenderer):
    def __init__(self, options: RenderOptions) -> None:
        super().__init__()

        self.node_renderer = None

        self.header_renderer = StreamGroupPartRenderer(
            flow_text=Text("iterate in ", style=PlanStyle.keyword.value),
            options=replace(options),
            render_node=True,
        )
        self.footer_renderer = StreamGroupPartRenderer(
            flow_text=Text("yield", style=PlanStyle.keyword.value),
            options=replace(options, signature_format="Output"),
        )


class StreamGroupPartRenderer(GroupPartRenderer):
    def __init__(
        self,
        flow_text: Text,
        options: RenderOptions,
        render_node: bool = False,
    ) -> None:
        super().__init__(
            flow_text=flow_text,
            options=options,
            render_node=render_node,
        )

    def render_signature_column(self, node: IcoNode) -> Text:
        signature = super().render_signature_column(node)
        # Add dim styling to the Iterator[...] parts
        signature.stylize("dim", 0, len("Iterator["))
        signature.stylize("dim", len(signature) - 1, len(signature))
        return signature


if __name__ == "__main__":
    from apriori.ico.describe.plan.rich_render.plan_renderer import PlanRenderer

    @operator()
    def produce_data_item(index: int) -> float:
        return 1

    # ──── 2. Define augmentation & collation pipelines per item ────

    @operator()
    def scale(x: float) -> float:
        return x * 1.1

    @operator()
    def shift(x: float) -> float:
        return x + 0.1

    stream = (produce_data_item | scale).stream()
    plan_renderer = PlanRenderer()
    plan_renderer.render(stream)

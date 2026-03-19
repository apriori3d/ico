from dataclasses import replace

from rich.text import Text

from ico.core.node import IcoNode
from ico.core.operator import operator
from ico.core.stream import IcoStream
from ico.describe.plan.options import PlanRendererOptions
from ico.describe.plan.rich_renderer.group_renderer import (
    GroupRenderer,
)
from ico.describe.plan.rich_renderer.renderer_registry import register_renderer
from ico.describe.plan.rich_renderer.row_renderer import RowRenderer
from ico.describe.rich_style import DescribeStyle


@register_renderer(IcoStream)
class IcoStreamRenderer(GroupRenderer):
    """Specialized group renderer for IcoStream nodes with flow visualization."""

    def __init__(self, options: PlanRendererOptions) -> None:
        super().__init__(
            options=options,
            header_renderer=StreamGroupPartRenderer(
                flow_column_prefix=Text(
                    "for each in ", style=DescribeStyle.keyword.value
                ),
                options=options,
            ),
            footer_renderer=StreamGroupPartRenderer(
                flow_column_prefix=Text("yield", style=DescribeStyle.keyword.value),
                options=replace(options, signature_format="Output"),
                show_name_column=False,
                show_type_column=False,
                show_state_column=False,
                flow_includes_node_info=False,
            ),
        )


class StreamGroupPartRenderer(RowRenderer):
    def __init__(
        self,
        flow_column_prefix: Text,
        options: PlanRendererOptions,
        show_name_column: bool = True,
        show_signature_column: bool = True,
        show_type_column: bool = True,
        show_state_column: bool = True,
        flow_includes_node_info: bool = True,
    ) -> None:
        super().__init__(
            flow_column_prefix=flow_column_prefix,
            options=options,
            show_name_column=show_name_column,
            show_signature_column=show_signature_column,
            show_type_column=show_type_column,
            show_state_column=show_state_column,
            flow_includes_node_info=flow_includes_node_info,
        )

    def render_signature_column(self, node: IcoNode) -> Text:
        signature = super().render_signature_column(node)
        # Add dim styling to the Iterator[...] parts
        signature.stylize("dim", 0, len("Iterator["))
        signature.stylize("dim", len(signature) - 1, len(signature))
        return signature


if __name__ == "__main__":
    from ico.describe.plan.rich_renderer.plan_renderer import PlanRenderer

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

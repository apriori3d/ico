from dataclasses import replace

from rich.text import Text

from ico.core.operator import IcoOperatorProtocol, operator
from ico.describe.plan.options import PlanRendererOptions
from ico.describe.plan.rich_renderer.group_renderer import (
    GroupRenderer,
)
from ico.describe.plan.rich_renderer.renderer_registry import register_renderer
from ico.describe.plan.rich_renderer.row_renderer import RowRenderer
from ico.describe.rich_style import DescribeStyle
from ico.runtime.agent.mp.mp_agent import MPAgent


@register_renderer(MPAgent)
class MPProcessRenderer(GroupRenderer):
    """Specialized group renderer for MPAgent nodes with worker process display."""

    def __init__(self, options: PlanRendererOptions) -> None:
        super().__init__(
            options=options,
            header_renderer=RowRenderer(
                flow_column_prefix=Text("send to ", style=DescribeStyle.keyword.value),
                options=options,
            ),
            footer_renderer=RowRenderer(
                flow_column_prefix=Text("receive", style=DescribeStyle.keyword.value),
                options=replace(options, signature_format="Output"),
                show_name_column=False,
                show_type_column=False,
                show_state_column=False,
                flow_includes_node_info=False,
            ),
        )


if __name__ == "__main__":
    from ico.describe.plan.rich_renderer.plan_renderer import PlanRenderer

    @operator()
    def fetch_data_item(index: int) -> float:
        return 1

    # ──── 2. Define augmentation & collation pipelines per item ────

    @operator()
    def scale(x: float) -> float:
        return x * 1.1

    @operator()
    def shift(x: float) -> float:
        return x + 0.1

    def create_worker_flow() -> IcoOperatorProtocol[int, float]:
        return fetch_data_item | scale | shift

    process = MPAgent(create_worker_flow)

    plan_renderer = PlanRenderer()
    plan_renderer.render(process)

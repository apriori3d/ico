from dataclasses import replace

from rich.text import Text

from apriori.ico.core.operator import IcoOperator, operator
from apriori.ico.describe.plan.options import PlanRendererOptions
from apriori.ico.describe.plan.rich_renderer.group_renderer import (
    GroupRenderer,
)
from apriori.ico.describe.plan.rich_renderer.renderer_registry import register_renderer
from apriori.ico.describe.plan.rich_renderer.row_renderer import RowRenderer
from apriori.ico.describe.rich_style import DescribeStyle
from apriori.ico.runtime.agent.mp.mp_agent import MPAgent


@register_renderer(MPAgent)
class MPProcessRenderer(GroupRenderer):
    def __init__(self, options: PlanRendererOptions) -> None:
        super().__init__(
            options=options,
            header_renderer=RowRenderer(
                flow_column_prefix=Text("send to ", style=DescribeStyle.keyword.value),
                options=replace(options, signature_format="Input"),
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
    from apriori.ico.describe.plan.rich_renderer.plan_renderer import PlanRenderer

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

    def create_worker_flow() -> IcoOperator[int, float]:
        return fetch_data_item | scale | shift

    process = MPAgent(create_worker_flow)

    plan_renderer = PlanRenderer()
    plan_renderer.render(process)

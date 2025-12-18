from dataclasses import replace

from rich.text import Text

from apriori.ico.core.operator import IcoOperator, operator
from apriori.ico.describe.plan.options import RenderOptions
from apriori.ico.describe.plan.rich_render.group_renderer import (
    GroupRenderer,
)
from apriori.ico.describe.plan.rich_render.renderer_registry import register_renderer
from apriori.ico.describe.plan.rich_render.row_renderer import RowRenderer
from apriori.ico.describe.plan.rich_render.utils import PlanStyle
from apriori.ico.runtime.agent.mp_process.mp_agent import MPAgent


@register_renderer(MPAgent)
class MPProcessRenderer(GroupRenderer):
    def __init__(self, options: RenderOptions) -> None:
        super().__init__(
            options=options,
            header_renderer=RowRenderer(
                flow_column_prefix=Text("send to ", style=PlanStyle.keyword.value),
                options=replace(options, signature_format="Input"),
            ),
            footer_renderer=RowRenderer(
                flow_column_prefix=Text("receive", style=PlanStyle.keyword.value),
                options=replace(options, signature_format="Output"),
                show_name_column=False,
                show_type_column=False,
                show_state_column=False,
                flow_includes_node_info=False,
            ),
        )


if __name__ == "__main__":
    from apriori.ico.describe.plan.rich_render.plan_renderer import PlanRenderer

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

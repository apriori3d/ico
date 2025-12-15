from rich.text import Text

from apriori.ico.core.operator import IcoOperator, operator
from apriori.ico.describe.plan.rich_render.group_renderer import (
    GroupPartRenderer,
    GroupRenderer,
)
from apriori.ico.describe.plan.rich_render.options import RenderOptions
from apriori.ico.describe.plan.rich_render.row_renderer import RowRenderer
from apriori.ico.describe.plan.rich_render.utils import PlanStyle
from apriori.ico.runtime.agent.mp_process.mp_process import MPProcess


class MPProcessRenderer(GroupRenderer):
    def __init__(self, options: RenderOptions) -> None:
        super().__init__()

        self.node_renderer = RowRenderer(options=options)

        self.entry_renderer = GroupPartRenderer(
            flow_text=Text("send", style=PlanStyle.keyword.value),
            group_part="entry",
            options=options,
        )
        self.exit_renderer = GroupPartRenderer(
            flow_text=Text("receive", style=PlanStyle.keyword.value),
            group_part="exit",
            options=options,
        )


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

    def create_worker_flow() -> IcoOperator[int, float]:
        return produce_data_item | scale | shift

    process = MPProcess(create_worker_flow)

    plan_renderer = PlanRenderer()
    plan_renderer.render(process)

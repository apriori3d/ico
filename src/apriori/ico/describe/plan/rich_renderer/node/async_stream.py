from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from rich.text import Text

from apriori.ico.core.async_stream import IcoAsyncStream
from apriori.ico.core.node import IcoNode
from apriori.ico.core.operator import IcoOperator, operator
from apriori.ico.describe.plan.options import PlanRendererOptions
from apriori.ico.describe.plan.rich_renderer.group_renderer import GroupRenderer
from apriori.ico.describe.plan.rich_renderer.node.stream import (
    StreamGroupPartRenderer,
)
from apriori.ico.describe.plan.rich_renderer.renderer_registry import register_renderer
from apriori.ico.describe.rich_style import DescribeStyle
from apriori.ico.runtime.agent.mp.mp_agent import MPAgent


@register_renderer(IcoAsyncStream)
class IcoAsyncStreamRenderer(GroupRenderer):
    def __init__(self, options: PlanRendererOptions) -> None:
        super().__init__(
            options=options,
            header_renderer=IcoAsyncStreamHeaderRender(
                flow_column_prefix=Text(
                    "parallel in ", style=DescribeStyle.keyword.value
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


class IcoAsyncStreamHeaderRender(StreamGroupPartRenderer):
    def _render_node_args_info(self, node: IcoNode) -> Text:
        assert isinstance(node, IcoAsyncStream)
        astream = cast(IcoAsyncStream[Any, Any], node)
        return Text(f"pool_size={astream.pool_size}", style=DescribeStyle.meta.value)


@operator()
def produce_data_item(index: int) -> float:
    return 1


@operator()
def scale(x: float) -> float:
    return x * 1.1


@operator()
def shift(x: float) -> float:
    return x + 0.1


def create_worker_flow() -> IcoOperator[float, float]:
    return scale | shift


if __name__ == "__main__":
    from apriori.ico.describe.plan.rich_renderer.plan_renderer import PlanRenderer

    pool = [MPAgent(create_worker_flow) for _ in range(4)]

    astream1 = IcoAsyncStream(
        lambda: MPAgent(create_worker_flow),
        pool_size=4,
        name="With subflow factory",
    )
    astream2 = IcoAsyncStream(
        [
            MPAgent(create_worker_flow),
            MPAgent(create_worker_flow),
        ],
        name="With explicit pool",
    )
    plan_renderer = PlanRenderer()
    plan_renderer.render(astream1)
    plan_renderer.render(astream2)

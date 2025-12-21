from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from rich.text import Text

from apriori.ico.core.async_stream import IcoAsyncStream
from apriori.ico.core.node import IcoNode
from apriori.ico.core.operator import IcoOperator, operator
from apriori.ico.describe.plan.options import PlanRendererOptions
from apriori.ico.describe.plan.rich_renderer.custom_renderer import CustomRenderer
from apriori.ico.describe.plan.rich_renderer.node.stream import (
    StreamGroupPartRenderer,
)
from apriori.ico.describe.plan.rich_renderer.render_target import PlanRenderTarget
from apriori.ico.describe.plan.rich_renderer.renderer_registry import register_renderer
from apriori.ico.describe.plan.rich_renderer.row_renderer import FlowTextRowRenderer
from apriori.ico.describe.rich_style import DescribeStyle
from apriori.ico.runtime.agent.mp.mp_agent import MPAgent


@register_renderer(IcoAsyncStream)
class IcoAsyncStreamRenderer(CustomRenderer):
    _header_renderer: IcoAsyncStreamNodeRender
    _footer_renderer: IcoAsyncStreamFooterRender

    def __init__(self, options: PlanRendererOptions) -> None:
        super().__init__(
            options=options,
        )
        header_text = Text("╭── ") + Text(
            "for each in ", style=DescribeStyle.keyword.value
        )
        self._header_renderer = IcoAsyncStreamNodeRender(
            flow_column_prefix=header_text,
            options=replace(options, signature_format="Input"),
        )

        self._desc_renderer = FlowTextRowRenderer(
            options=options,
            flow_column_prefix=Text("⚙️ parallel", style=DescribeStyle.meta.value),
        )

        footer_text = Text("╰─▸ ") + Text("yield", style=DescribeStyle.keyword.value)
        self._footer_renderer = IcoAsyncStreamFooterRender(
            flow_column_prefix=footer_text,
            options=replace(options, signature_format="Output"),
            show_name_column=False,
            show_type_column=False,
            show_state_column=False,
            flow_includes_node_info=False,
        )

    def render(self, plan: PlanRenderTarget, node: IcoNode) -> None:
        assert isinstance(node, IcoAsyncStream)
        astream = cast(IcoAsyncStream[Any, Any], node)

        plan.render_row(self._header_renderer, astream)

        plan.push_group_indent(Text("│   "))

        self._desc_renderer.flow_column_prefix = Text(
            "parallel", style=DescribeStyle.semantic_meta.value
        ) + Text(
            f"(pool_size={astream.pool_size}, ordered={astream.ordered})",
            style=DescribeStyle.meta.value,
        )
        # plan.render_row(self._desc_renderer, astream)

        if astream.subflow_factory is not None:
            plan.render_node(astream.subflow_factory())
        else:
            for child in astream.pool:
                plan.render_node(child)

        plan.pop_group_indent()

        plan.render_row(self._footer_renderer, astream)


class IcoAsyncStreamNodeRender(StreamGroupPartRenderer):
    def _render_node_args_info(self, node: IcoNode) -> Text:
        assert isinstance(node, IcoAsyncStream)
        astream = cast(IcoAsyncStream[Any, Any], node)
        return Text(
            f"pool_size={astream.pool_size}",
            style=DescribeStyle.meta.value,
        )


class IcoAsyncStreamFooterRender(StreamGroupPartRenderer):
    def render_flow_column(self, node: IcoNode) -> Text:
        args_info = self._render_node_args_info(node)
        assert self.flow_column_prefix is not None

        return (
            self.flow_column_prefix
            + Text("(", style=DescribeStyle.keyword.value)
            + args_info
            + Text(")", style=DescribeStyle.keyword.value)
        )

    def _render_node_args_info(self, node: IcoNode) -> Text:
        assert isinstance(node, IcoAsyncStream)
        astream = cast(IcoAsyncStream[Any, Any], node)
        return Text(
            f"ordered={astream.ordered}",
            style=DescribeStyle.meta.value,
        )


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

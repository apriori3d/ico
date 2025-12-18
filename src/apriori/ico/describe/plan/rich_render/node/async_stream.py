from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from rich.text import Text

from apriori.ico.core.async_stream import IcoAsyncStream
from apriori.ico.core.node import IcoNode
from apriori.ico.core.operator import IcoOperator, operator
from apriori.ico.describe.plan.options import RenderOptions
from apriori.ico.describe.plan.rich_render.custom_renderer import CustomRenderer
from apriori.ico.describe.plan.rich_render.node.stream import (
    StreamGroupPartRenderer,
)
from apriori.ico.describe.plan.rich_render.render_target import RenderTarget
from apriori.ico.describe.plan.rich_render.renderer_registry import register_renderer
from apriori.ico.describe.plan.rich_render.row_renderer import FlowTextRowRenderer
from apriori.ico.describe.plan.rich_render.utils import PlanStyle
from apriori.ico.runtime.agent.mp_process.mp_agent import MPAgent


@register_renderer(IcoAsyncStream)
class IcoAsyncStreamRenderer(CustomRenderer):
    _header_renderer: IcoAsyncStreamNodeRender
    _footer_renderer: IcoAsyncStreamFooterRender

    def __init__(self, options: RenderOptions) -> None:
        super().__init__(
            options=options,
        )
        header_text = Text("╭── ") + Text("for each in ", style=PlanStyle.keyword.value)
        self._header_renderer = IcoAsyncStreamNodeRender(
            flow_column_prefix=header_text,
            options=replace(options, signature_format="Input"),
        )

        self._desc_renderer = FlowTextRowRenderer(
            options=options,
            flow_column_prefix=Text("⚙️ parallel", style=PlanStyle.meta.value),
        )

        footer_text = Text("╰─▸ ") + Text("yield", style=PlanStyle.keyword.value)
        self._footer_renderer = IcoAsyncStreamFooterRender(
            flow_column_prefix=footer_text,
            options=replace(options, signature_format="Output"),
            show_name_column=False,
            show_type_column=False,
            show_state_column=False,
            flow_includes_node_info=False,
        )

    def render(self, plan: RenderTarget, node: IcoNode) -> None:
        assert isinstance(node, IcoAsyncStream)
        astream = cast(IcoAsyncStream[Any, Any], node)

        plan.render_row(self._header_renderer, astream)

        plan.push_group_indent(Text("│   "))

        self._desc_renderer.flow_column_prefix = Text(
            "parallel", style=PlanStyle.semantic_meta.value
        ) + Text(
            f"(pool_size={astream.pool_size}, ordered={astream.ordered})",
            style=PlanStyle.meta.value,
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
            style=PlanStyle.meta.value,
        )


class IcoAsyncStreamFooterRender(StreamGroupPartRenderer):
    def render_flow_column(self, node: IcoNode) -> Text:
        args_info = self._render_node_args_info(node)
        assert self.flow_column_prefix is not None

        return (
            self.flow_column_prefix
            + Text("(", style=PlanStyle.keyword.value)
            + args_info
            + Text(")", style=PlanStyle.keyword.value)
        )

    def _render_node_args_info(self, node: IcoNode) -> Text:
        assert isinstance(node, IcoAsyncStream)
        astream = cast(IcoAsyncStream[Any, Any], node)
        return Text(
            f"ordered={astream.ordered}",
            style=PlanStyle.meta.value,
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
    from apriori.ico.describe.plan.rich_render.plan_renderer import PlanRenderer

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

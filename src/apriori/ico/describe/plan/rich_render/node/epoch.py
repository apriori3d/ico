from collections.abc import Iterable
from dataclasses import replace
from typing import Any, cast

from rich.text import Text

from apriori.ico.core.async_stream import IcoAsyncStream
from apriori.ico.core.context_operator import context_operator
from apriori.ico.core.context_pipeline import IcoContextPipeline
from apriori.ico.core.epoch import IcoEpoch
from apriori.ico.core.node import IcoNode
from apriori.ico.core.operator import IcoOperator, operator
from apriori.ico.core.process import IcoProcess
from apriori.ico.core.source import source
from apriori.ico.describe.plan.options import RenderOptions
from apriori.ico.describe.plan.rich_render.custom_renderer import CustomRenderer
from apriori.ico.describe.plan.rich_render.node.stream import StreamGroupPartRenderer
from apriori.ico.describe.plan.rich_render.render_target import RenderTarget
from apriori.ico.describe.plan.rich_render.renderer_registry import register_renderer
from apriori.ico.describe.plan.rich_render.row_renderer import (
    RowRenderer,
)
from apriori.ico.describe.plan.rich_render.utils import PlanStyle
from apriori.ico.runtime.agent.mp_process.mp_agent import MPAgent


@register_renderer(IcoEpoch)
class IcoEpochRenderer(CustomRenderer):
    def __init__(self, options: RenderOptions) -> None:
        super().__init__(options=options)
        self._row_renderer = RowRenderer(options=options)

        source_header = Text("╭── ") + Text(
            "for each in ", style=PlanStyle.keyword.value
        )
        self._source_header_renderer = StreamGroupPartRenderer(
            flow_column_prefix=source_header,
            options=replace(options, signature_format="Output"),
            show_name_column=False,
            show_type_column=False,
            show_state_column=False,
            flow_includes_node_info=False,
        )

        context_header = Text("├─▸ ") + Text("apply", style=PlanStyle.keyword.value)
        self._context_header_renderer = RowRenderer(
            flow_column_prefix=context_header,
            options=replace(options, signature_format="Input"),
            show_name_column=False,
            show_type_column=False,
            show_state_column=False,
            flow_includes_node_info=False,
        )

        footer_text = Text("╰─▸ ") + Text("emit", style=PlanStyle.keyword.value)
        self._footer_renderer = RowRenderer(
            flow_column_prefix=footer_text,
            options=replace(options, signature_format="Output"),
            show_name_column=False,
            show_type_column=False,
            show_state_column=False,
            flow_includes_node_info=False,
        )

    def render(self, plan: RenderTarget, node: IcoNode) -> None:
        assert isinstance(node, IcoEpoch)
        epoch = cast(IcoEpoch[Any, Any], node)

        plan.render_row(self._row_renderer, epoch)

        plan.render_row(self._source_header_renderer, epoch.source)

        plan.push_group_indent(Text("│   "))
        plan.render_node(epoch.source)
        plan.pop_group_indent()

        plan.render_row(self._context_header_renderer, epoch.context_operator)

        plan.push_group_indent(Text("│   "))
        plan.render_node(epoch.context_operator)
        plan.pop_group_indent()

        plan.render_row(self._footer_renderer, epoch)


class IcoAsyncStreamNodeRender(RowRenderer):
    def _render_node_args_info(self, node: IcoNode) -> Text:
        assert isinstance(node, IcoAsyncStream)
        astream = cast(IcoAsyncStream[Any, Any], node)
        return Text(f"pool_size={len(astream.pool)}", style=PlanStyle.meta.value)


@source()
def produce_data_item() -> Iterable[float]:
    yield 1
    yield 2


# ──── 2. Define augmentation & collation pipelines per item ────


@operator()
def scale(x: float) -> float:
    return x * 1.1


@operator()
def shift(x: float) -> float:
    return x + 0.1


def create_worker_flow() -> IcoOperator[float, float]:
    return scale | shift


class TrainContext:
    pass


@context_operator()
def train_step(item: float, context: TrainContext) -> TrainContext:
    return context


@operator()
def logging_step(context: TrainContext) -> TrainContext:
    return context


if __name__ == "__main__":
    from apriori.ico.describe.plan.rich_render.plan_renderer import PlanRenderer

    """
    IcoEpoch()
    ╭── for each in IcoSource():
    │   next_index()
    │   fetch_item()
    │   ...
    │─▸ update context in IcoContextPipeline():
    │   train_step()
    │   logging_step()
    │   save_checkpoint_step()
    ╰─▸ return

    """
    pool = [MPAgent(create_worker_flow) for _ in range(4)]

    astream = IcoAsyncStream(
        lambda: MPAgent(create_worker_flow),
        pool_size=4,
        name="With subflow factory",
    )
    data_flow = produce_data_item | astream

    train_pipeline = IcoContextPipeline(
        train_step,
        logging_step,
        name="Train Pipeline",
    )
    epoch_flow = IcoEpoch(
        source=data_flow,
        context_operator=train_pipeline,
        name="Training Epoch",
    )
    train_loop = IcoProcess(epoch_flow, num_iterations=100, name="Train Loop")
    plan_renderer = PlanRenderer()
    plan_renderer.render(train_loop)

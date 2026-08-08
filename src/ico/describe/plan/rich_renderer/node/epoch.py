from collections.abc import Iterable
from dataclasses import replace
from typing import Any, cast

from rich.text import Text

from ico.core.async_stream import IcoAsyncStream
from ico.core.context_operator import context_operator
from ico.core.context_pipeline import IcoContextPipeline
from ico.core.epoch import IcoEpoch
from ico.core.node import IcoNodeProtocol
from ico.core.operator import IcoOperatorProtocol, operator
from ico.core.process import IcoProcess
from ico.core.source import source
from ico.describe.plan.options import PlanRendererOptions
from ico.describe.plan.rich_renderer.custom_renderer import CustomRenderer
from ico.describe.plan.rich_renderer.node.stream import StreamGroupPartRenderer
from ico.describe.plan.rich_renderer.render_target import (
    PlanRenderTarget,
    create_plan_walker,
)
from ico.describe.plan.rich_renderer.renderer_registry import register_renderer
from ico.describe.plan.rich_renderer.row_renderer import (
    RowRenderer,
)
from ico.describe.rich_style import DescribeStyle
from ico.runtime.agent.mp.mp_agent import MPAgent


@register_renderer(IcoEpoch)
class IcoEpochRenderer(CustomRenderer):
    """
    Custom renderer for IcoEpoch nodes with iteration structure display.

    Renders epochs as expandable groups showing source iteration,
    item processing pipelines, and result aggregation.
    """

    def __init__(self, options: PlanRendererOptions) -> None:
        super().__init__(options=options)
        self._row_renderer = RowRenderer(options=options)

        self._source_header_renderer = StreamGroupPartRenderer(
            flow_column_prefix=Text("for each in ", style=DescribeStyle.keyword.value),
            options=replace(options, signature_format="Output"),
            show_name_column=False,
            show_type_column=False,
            show_state_column=False,
            flow_includes_node_info=False,
        )

        context_header = Text("├─▸ ") + Text("apply", style=DescribeStyle.keyword.value)
        self._context_header_renderer = RowRenderer(
            flow_column_prefix=context_header,
            options=replace(options, signature_format="Input"),
            show_name_column=False,
            show_type_column=False,
            show_state_column=False,
            flow_includes_node_info=False,
        )

        footer_text = Text("╰─▸ ") + Text("emit", style=DescribeStyle.keyword.value)
        self._footer_renderer = RowRenderer(
            flow_column_prefix=footer_text,
            options=replace(options, signature_format="Output"),
            show_name_column=False,
            show_type_column=False,
            show_state_column=False,
            flow_includes_node_info=False,
        )

    def render(self, plan: PlanRenderTarget, node: IcoNodeProtocol) -> None:
        assert isinstance(node, IcoEpoch)
        epoch = cast(IcoEpoch[Any, Any], node)

        plan.render_row(self._row_renderer, epoch)

        # Use tree walker to render nested epoch elements
        tree_walker = create_plan_walker(
            expand_remote_flows=self.options.show_remote_flows
        )

        # Render epoch source part
        plan.render_row(
            self._source_header_renderer,
            epoch.source,
            indent=Text("╭── ", style=DescribeStyle.group.value),
        )

        # Render source flow
        plan.push_group_indent(Text("│   "))

        tree_walker.walk(
            epoch.source,
            visit_fn=plan.render_node,
            order="pre_post",  # Need post order to close groups
        )

        # Clear group indent
        plan.pop_group_indent()

        # Render context operator part
        plan.render_row(self._context_header_renderer, epoch.context_operator)

        # Render context operator flow
        plan.push_group_indent(Text("│   "))

        tree_walker.walk(
            epoch.context_operator,
            visit_fn=plan.render_node,
            order="pre_post",  # Need post order to close groups
        )
        plan.pop_group_indent()

        plan.render_row(self._footer_renderer, epoch)


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


def create_worker_flow() -> IcoOperatorProtocol[float, float]:
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
    from ico.describe.plan.rich_renderer.plan_renderer import PlanRenderer

    """
    IcoEpoch()
    ╭── for each in IcoSource():
    │   next_index()
    │   fetch_item()
    │   ...
    │─▸ apply
    │   train_step()
    │   logging_step()
    │   save_checkpoint_step()
    ╰─▸ emit

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

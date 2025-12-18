from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any, cast

from rich.text import Text

from apriori.ico.core.node import IcoNode
from apriori.ico.core.process import IcoProcess
from apriori.ico.core.sink import sink
from apriori.ico.core.source import source
from apriori.ico.describe.plan.options import RenderOptions
from apriori.ico.describe.plan.rich_render.group_renderer import (
    GroupRenderer,
)
from apriori.ico.describe.plan.rich_render.renderer_registry import register_renderer
from apriori.ico.describe.plan.rich_render.row_renderer import (
    RowRenderer,
)
from apriori.ico.describe.plan.rich_render.utils import (
    PlanStyle,
)


@register_renderer(IcoProcess)
class IcoProcessRenderer(GroupRenderer):
    def __init__(self, options: RenderOptions) -> None:
        super().__init__(
            options=options,
            header_renderer=IcoProcessNodeRender(
                flow_column_prefix=Text("iterate in ", style=PlanStyle.keyword.value),
                options=replace(options, signature_format="Input"),
            ),
            footer_renderer=RowRenderer(
                flow_column_prefix=Text("emit", style=PlanStyle.keyword.value),
                options=replace(options, signature_format="Output"),
                show_name_column=False,
                show_type_column=False,
                show_state_column=False,
                flow_includes_node_info=False,
            ),
        )


class IcoProcessNodeRender(RowRenderer):
    def _render_node_args_info(self, node: IcoNode) -> Text:
        assert isinstance(node, IcoProcess)
        process = cast(IcoProcess[Any], node)
        return Text(f"num_iter={process.num_iterations}", style=PlanStyle.meta.value)


if __name__ == "__main__":
    from apriori.ico.describe.plan.rich_render.plan_renderer import PlanRenderer

    data = [0.5, 1.0, 1.5, 2.0, 0.8, 0.2, 1.0, 1.2, 0.9]

    @source(name="Floats generator")
    def floats() -> Iterable[float]:
        return data

    @sink(name="Read results")
    def read_result(item: float) -> None:
        pass

    iter_flow = floats | read_result
    iter_flow.name = "Iteration Flow"
    process = IcoProcess(iter_flow, num_iterations=5, name="Main Process")

    plan_renderer = PlanRenderer()
    plan_renderer.render(process)

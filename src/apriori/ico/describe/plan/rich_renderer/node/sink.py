from __future__ import annotations

from typing import Any, cast

from rich.text import Text

from apriori.ico.core.node import IcoNode
from apriori.ico.core.sink import IcoSink, sink
from apriori.ico.describe.plan.rich_renderer.renderer_registry import register_renderer
from apriori.ico.describe.plan.rich_renderer.row_renderer import (
    RowRenderer,
)
from apriori.ico.describe.rich_utils import render_callable


@register_renderer(IcoSink)
class IcoSinkRender(RowRenderer):
    def _render_node_args_info(self, node: IcoNode) -> Text:
        assert isinstance(node, IcoSink)
        sink = cast(IcoSink[Any], node)

        return render_callable(sink.consumer, options=self.options)


if __name__ == "__main__":
    from apriori.ico.describe.plan.rich_renderer.plan_renderer import PlanRenderer

    @sink(name="Read results")
    def read_result(item: float) -> None:
        pass

    plan_renderer = PlanRenderer()
    plan_renderer.render(read_result)

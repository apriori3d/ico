from __future__ import annotations

from typing import Any, cast

from rich.text import Text

from apriori.ico.core.node import IcoNode
from apriori.ico.core.sink import IcoSink, sink
from apriori.ico.describe.plan.rich_render.row_renderer import (
    RowRenderer,
)


class IcoSinkRender(RowRenderer):
    def _render_node_args_info(self, node: IcoNode) -> Text:
        assert isinstance(node, IcoSink)
        sink = cast(IcoSink[Any], node)

        return self._render_callable(sink.consumer)


if __name__ == "__main__":
    from apriori.ico.describe.plan.rich_render.plan_renderer import PlanRenderer

    @sink(name="Read results")
    def read_result(item: float) -> None:
        pass

    plan_renderer = PlanRenderer()
    plan_renderer.render(read_result)

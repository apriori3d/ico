from __future__ import annotations

from typing import Any, cast

from rich.text import Text

from ico.core.node import IcoNodeProtocol
from ico.core.sink import IcoSink, sink
from ico.describe.plan.rich_renderer.renderer_registry import register_renderer
from ico.describe.plan.rich_renderer.row_renderer import (
    RowRenderer,
)
from ico.describe.rich_utils import render_callable


@register_renderer(IcoSink)
class IcoSinkRender(RowRenderer):
    """Specialized renderer for IcoSink nodes showing consumer function."""

    def _render_node_args_info(self, node: IcoNodeProtocol) -> Text:
        """Render sink consumer function information."""
        assert isinstance(node, IcoSink)
        sink = cast(IcoSink[Any], node)

        return render_callable(sink.consumer, options=self.options)


if __name__ == "__main__":
    from ico.describe.plan.rich_renderer.plan_renderer import PlanRenderer

    @sink(name="Read results")
    def read_result(item: float) -> None:
        pass

    plan_renderer = PlanRenderer()
    plan_renderer.render(read_result)

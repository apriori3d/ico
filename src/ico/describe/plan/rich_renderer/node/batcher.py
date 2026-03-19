from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

from rich.text import Text

from ico.core.batcher import IcoBatcher
from ico.core.node import IcoNode
from ico.core.source import source
from ico.describe.plan.rich_renderer.renderer_registry import register_renderer
from ico.describe.plan.rich_renderer.row_renderer import (
    RowRenderer,
)
from ico.describe.rich_style import DescribeStyle


@register_renderer(IcoBatcher)
class IcoBatcherRender(RowRenderer):
    """Specialized renderer for IcoBatcher nodes showing batch size."""

    def _render_node_args_info(self, node: IcoNode) -> Text:
        """Render batcher configuration with batch size info."""
        assert isinstance(node, IcoBatcher)
        batcher = cast(IcoBatcher[Any], node)
        return Text(f"batch_size={batcher.batch_size}", style=DescribeStyle.meta.value)


if __name__ == "__main__":
    from ico.describe.plan.rich_renderer.plan_renderer import PlanRenderer

    data = [0.5, 1.0, 1.5, 2.0, 0.8, 0.2, 1.0, 1.2, 0.9]

    @source(name="Floats generator")
    def floats() -> Iterable[float]:
        return data

    plan_renderer = PlanRenderer()
    plan_renderer.render(floats | IcoBatcher[float](batch_size=4))

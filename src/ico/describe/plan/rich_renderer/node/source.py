from __future__ import annotations

from collections.abc import Iterable, Sized
from typing import Any, cast

from rich.text import Text

from ico.core.node import IcoNode
from ico.core.source import IcoSource, source
from ico.describe.plan.rich_renderer.renderer_registry import register_renderer
from ico.describe.plan.rich_renderer.row_renderer import (
    RowRenderer,
)
from ico.describe.rich_style import DescribeStyle
from ico.describe.rich_utils import (
    render_callable,
)


@register_renderer(IcoSource)
class IcoSourceRender(RowRenderer):
    """Specialized renderer for IcoSource nodes with data size information."""

    def _render_node_args_info(self, node: IcoNode) -> Text:
        """Render source provider info with optional size details."""
        assert isinstance(node, IcoSource)
        source = cast(IcoSource[Any], node)

        provider_info = render_callable(source.provider, options=self.options)

        # Add size info if required
        if self.options.query_iterable_size:
            data_source = source.provider()

            if isinstance(data_source, Sized):
                provider_info += Text(
                    f", size={len(data_source)}", style=DescribeStyle.meta.value
                )

        return provider_info


if __name__ == "__main__":
    from ico.describe.plan.rich_renderer.plan_renderer import PlanRenderer

    data = [0.5, 1.0, 1.5, 2.0, 0.8, 0.2, 1.0, 1.2, 0.9]

    @source(name="Floats generator")
    def floats() -> Iterable[float]:
        return data

    plan_renderer = PlanRenderer()
    plan_renderer.render(floats)

    source2 = IcoSource(lambda: data, name="Floats data")
    plan_renderer.render(source2)

    class FloatProvider:
        def __init__(self, data: list[float]):
            self.data = data

        def __call__(self) -> Iterable[float]:
            return self.data

        def __str__(self) -> str:
            return f"FloatProvider(total={len(self.data)})"

    source3 = IcoSource(FloatProvider(data), name="Float Provider instance")
    plan_renderer.render(source3)

    plan_renderer.options.callable_format = "str()"
    plan_renderer.render(source3)

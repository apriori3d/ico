"""Plan renderer for an operator whose children are monitors it drives
itself, e.g. an ``IcoProgress`` that counts the steps of one call: the
operator's row, then a row per monitor. The monitors are not a subflow:
items do not pass them on their way through the operator::

    Wan22Loop()                     LoopPlan → LoopFrames
    ╰ reports ⏳IcoProgress()                               denoise
"""

from __future__ import annotations

from rich.text import Text

from ico.core.node import IcoNodeProtocol
from ico.core.runtime.monitor import IcoMonitor
from ico.describe.plan.options import PlanRendererOptions
from ico.describe.plan.rich_renderer.custom_renderer import CustomRenderer
from ico.describe.plan.rich_renderer.render_target import PlanRenderTarget
from ico.describe.plan.rich_renderer.row_renderer import RowRenderer
from ico.describe.rich_style import DescribeStyle


def owns_monitors(node: IcoNodeProtocol) -> bool:
    """Whether all children of ``node`` are monitors (and it has some)."""
    children = list(node.children)
    return bool(children) and all(isinstance(c, IcoMonitor) for c in children)


class OwnedMonitorsRenderer(CustomRenderer):
    """The operator as a row, its monitors below it without a signature."""

    def __init__(self, options: PlanRendererOptions) -> None:
        super().__init__(options)
        self.operator = RowRenderer(options=options)
        self.monitor = RowRenderer(
            options=options,
            show_signature_column=False,
            flow_column_prefix=Text("reports ", style=DescribeStyle.keyword.value),
        )

    def render(self, plan: PlanRenderTarget, node: IcoNodeProtocol) -> None:
        plan.render_row(self.operator, node)
        children = list(node.children)
        for i, child in enumerate(children):
            branch = "╰ " if i == len(children) - 1 else "├ "
            plan.render_row(
                self.monitor,
                child,
                indent=Text(branch, style=DescribeStyle.group.value),
            )

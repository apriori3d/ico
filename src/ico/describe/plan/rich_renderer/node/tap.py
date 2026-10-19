"""Plan renderer for ``IcoTap``: the child runs inside the group, and the
footer says that the input goes on, not the child's output::

    ╭── tap 🔍IcoTap()          VideoRef → VideoRef
    │   scan_text@1             VideoRef → TextReport
    ╰─▸ pass the input on       VideoRef
"""

from __future__ import annotations

from dataclasses import replace

from rich.text import Text

from ico.core.tap import IcoTap
from ico.describe.plan.options import PlanRendererOptions
from ico.describe.plan.rich_renderer.group_renderer import GroupRenderer
from ico.describe.plan.rich_renderer.renderer_registry import register_renderer
from ico.describe.plan.rich_renderer.row_renderer import RowRenderer
from ico.describe.rich_style import DescribeStyle


@register_renderer(IcoTap)
class IcoTapRenderer(GroupRenderer):
    """Runs its child for the side effect and passes the input on."""

    def __init__(self, options: PlanRendererOptions) -> None:
        super().__init__(
            options=options,
            header_renderer=RowRenderer(
                flow_column_prefix=_keyword("tap "),
                options=options,
            ),
            footer_renderer=RowRenderer(
                flow_column_prefix=_keyword("pass the input on"),
                options=replace(options, signature_format="Output"),
                show_name_column=False,
                show_type_column=False,
                show_state_column=False,
                flow_includes_node_info=False,
            ),
        )


def _keyword(text: str) -> Text:
    return Text(text, style=DescribeStyle.keyword.value)

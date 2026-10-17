"""Plan renderers for predicate nodes: ``IcoFilter`` and ``IcoPartition``.

The predicate is rendered inside the group, so it reads as a condition and
not as a step of the stream::

    ╭── keep where 🔍IcoFilter()     Iterator[int] → Iterator[int]   even
    │   is_even                      int → bool
    ╰─▸ yield kept                   Iterator[int]
"""

from __future__ import annotations

from dataclasses import replace

from rich.text import Text

from ico.core.filter import IcoFilter
from ico.core.partition import IcoPartition
from ico.describe.plan.options import PlanRendererOptions
from ico.describe.plan.rich_renderer.group_renderer import GroupRenderer
from ico.describe.plan.rich_renderer.node.stream import StreamGroupPartRenderer
from ico.describe.plan.rich_renderer.renderer_registry import register_renderer
from ico.describe.plan.rich_renderer.row_renderer import RowRenderer
from ico.describe.rich_style import DescribeStyle


@register_renderer(IcoFilter)
class IcoFilterRenderer(GroupRenderer):
    """Keeps the stream items for which the predicate is true."""

    def __init__(self, options: PlanRendererOptions) -> None:
        super().__init__(
            options=options,
            header_renderer=StreamGroupPartRenderer(
                flow_column_prefix=_keyword("keep where "),
                options=options,
            ),
            footer_renderer=StreamGroupPartRenderer(
                flow_column_prefix=_keyword("yield kept"),
                options=replace(options, signature_format="Output"),
                show_name_column=False,
                show_type_column=False,
                show_state_column=False,
                flow_includes_node_info=False,
            ),
        )


@register_renderer(IcoPartition)
class IcoPartitionRenderer(GroupRenderer):
    """Splits a stream into the items that match the predicate and the rest."""

    def __init__(self, options: PlanRendererOptions) -> None:
        super().__init__(
            options=options,
            header_renderer=RowRenderer(
                flow_column_prefix=_keyword("split by "),
                options=options,
            ),
            footer_renderer=RowRenderer(
                flow_column_prefix=_keyword("collect matched, rest"),
                options=replace(options, signature_format="Output"),
                show_name_column=False,
                show_type_column=False,
                show_state_column=False,
                flow_includes_node_info=False,
            ),
        )


def _keyword(text: str) -> Text:
    return Text(text, style=DescribeStyle.keyword.value)

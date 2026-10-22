"""Plan renderer for ``IcoFold``: the source the items come from, then the
step that folds each item into the state, then the state it emits::

    🧠IcoFold()                          CifarTrainContext → CifarTrainContext
    ╭── for each in                      CifarBatch
    │   ...                              (the source)
    ├─▸ fold                             tuple[CifarBatch, CifarTrainContext]
    │   ...                              (the step)
    ╰─▸ emit                             CifarTrainContext
"""

from dataclasses import replace
from typing import Any, cast

from rich.text import Text

from ico.core.fold import IcoFold
from ico.core.node import IcoNodeProtocol
from ico.describe.plan.options import PlanRendererOptions
from ico.describe.plan.rich_renderer.custom_renderer import CustomRenderer
from ico.describe.plan.rich_renderer.node.stream import StreamGroupPartRenderer
from ico.describe.plan.rich_renderer.render_target import (
    PlanRenderTarget,
    flattened,
)
from ico.describe.plan.rich_renderer.renderer_registry import register_renderer
from ico.describe.plan.rich_renderer.row_renderer import RowRenderer
from ico.describe.rich_style import DescribeStyle


@register_renderer(IcoFold)
class IcoFoldRenderer(CustomRenderer):
    """The fold's row, its source as a group, its step, and what it emits."""

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

        step_header = Text("├─▸ ") + Text("fold", style=DescribeStyle.keyword.value)
        self._step_header_renderer = RowRenderer(
            flow_column_prefix=step_header,
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
        assert isinstance(node, IcoFold)
        fold = cast(IcoFold[Any, Any], node)

        plan.render_row(self._row_renderer, fold)

        # A part's header row has its key when the part has no row of its
        # own (a chain).
        plan.render_row(
            self._source_header_renderer,
            fold.source,
            indent=Text("╭── ", style=DescribeStyle.group.value),
            keyed=flattened(self.options, fold.source),
        )
        plan.push_group_indent(Text("│   "))
        plan.render_subtree(fold.source)
        plan.pop_group_indent()

        plan.render_row(
            self._step_header_renderer,
            fold.step,
            keyed=flattened(self.options, fold.step),
        )
        plan.push_group_indent(Text("│   "))
        plan.render_subtree(fold.step)
        plan.pop_group_indent()

        plan.render_row(self._footer_renderer, fold, keyed=False)

"""Plan renderer for ``IcoFold`` and ``IcoScan``: the source the items come
from, then the step that folds each item into the state, then the state it
emits (a scan: ``scan`` and ``yield each``, a pair per item)::

    🧠IcoFold()                          CifarTrainContext → CifarTrainContext
    ╭── for each in                      CifarBatch
    │   ...                              (the source)
    ├─▸ fold                             tuple[CifarBatch, CifarTrainContext]
    │   ...                              (the step)
    ├─▸ until                            CifarTrainContext
    │   ...                              (the stop condition, if any)
    ╰─▸ emit                             CifarTrainContext
"""

from dataclasses import replace
from typing import Any, cast

from rich.text import Text

from ico.core.fold import IcoFold
from ico.core.node import IcoNodeProtocol
from ico.core.scan import IcoScan
from ico.describe.plan.options import PlanRendererOptions, SignatureFormat
from ico.describe.plan.rich_renderer.custom_renderer import CustomRenderer
from ico.describe.plan.rich_renderer.node.stream import StreamGroupPartRenderer
from ico.describe.plan.rich_renderer.render_target import (
    PlanRenderTarget,
    flattened,
)
from ico.describe.plan.rich_renderer.renderer_registry import register_renderer
from ico.describe.plan.rich_renderer.row_renderer import RowRenderer
from ico.describe.rich_style import DescribeStyle


@register_renderer(IcoFold, IcoScan)
class IcoFoldRenderer(CustomRenderer):
    """The fold's (or scan's) row, its source as a group, its step, and what
    it emits."""

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

        self._step_header_renderers = {
            keyword: self._header("├─▸ ", keyword, "Input")
            for keyword in ("fold", "scan")
        }

        self._until_header_renderer = self._header("├─▸ ", "until", "Input")
        self._footer_renderers = {
            keyword: self._header("╰─▸ ", keyword, "Output")
            for keyword in ("emit", "yield each")
        }

    def _header(
        self, prefix: str, keyword: str, signature_format: SignatureFormat
    ) -> RowRenderer:
        return RowRenderer(
            flow_column_prefix=Text(prefix)
            + Text(keyword, style=DescribeStyle.keyword.value),
            options=replace(self.options, signature_format=signature_format),
            show_name_column=False,
            show_type_column=False,
            show_state_column=False,
            flow_includes_node_info=False,
        )

    def render(self, plan: PlanRenderTarget, node: IcoNodeProtocol) -> None:
        assert isinstance(node, IcoFold | IcoScan)
        fold = cast(IcoFold[Any, Any] | IcoScan[Any, Any, Any], node)
        is_scan = isinstance(node, IcoScan)

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
            self._step_header_renderers["scan" if is_scan else "fold"],
            fold.step,
            keyed=flattened(self.options, fold.step),
        )
        plan.push_group_indent(Text("│   "))
        plan.render_subtree(fold.step)
        plan.pop_group_indent()

        if fold.until is not None:
            plan.render_row(
                self._until_header_renderer,
                fold.until,
                keyed=flattened(self.options, fold.until),
            )
            plan.push_group_indent(Text("│   "))
            plan.render_subtree(fold.until)
            plan.pop_group_indent()

        footer = self._footer_renderers["yield each" if is_scan else "emit"]
        plan.render_row(footer, fold, keyed=False)

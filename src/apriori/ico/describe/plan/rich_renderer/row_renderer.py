from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from rich.text import Text

from apriori.ico.core.context_operator import IcoContextOperator
from apriori.ico.core.node import IcoNode
from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.signature_utils import format_ico_type
from apriori.ico.describe.plan.options import (
    PlanRendererColumn,
    PlanRendererOptions,
)
from apriori.ico.describe.rich_style import DescribeStyle
from apriori.ico.describe.rich_utils import (
    render_callable,
    render_node_class,
)
from apriori.ico.describe.utils import match_icon


class RowRenderer:
    """
    Renders individual nodes as table rows with configurable columns.

    Supports Flow, Signature, Name columns with customizable formatting,
    prefix/postfix text, and column visibility controls.
    """

    options: PlanRendererOptions
    show_name_column: bool
    show_signature_column: bool
    show_type_column: bool
    show_state_column: bool
    flow_column_prefix: Text | None
    flow_column_postfix: Text | None
    flow_includes_node_info: bool = True
    _column_renderer: dict[PlanRendererColumn, Callable[[IcoNode], Text]]

    def __init__(
        self,
        options: PlanRendererOptions,
        show_name_column: bool = True,
        show_signature_column: bool = True,
        show_type_column: bool = True,
        show_state_column: bool = True,
        flow_column_prefix: Text | None = None,
        flow_column_postfix: Text | None = None,
        flow_includes_node_info: bool = True,
    ) -> None:
        """Initialize row renderer with column visibility and formatting options."""
        self.options = options
        self.show_name_column = show_name_column
        self.show_signature_column = show_signature_column
        self.show_type_column = show_type_column
        self.show_state_column = show_state_column
        self.flow_column_prefix = flow_column_prefix
        self.flow_column_postfix = flow_column_postfix
        self.flow_includes_node_info = flow_includes_node_info

        self._column_renderer: dict[PlanRendererColumn, Callable[[IcoNode], Text]] = {
            "Flow": self.render_flow_column,
            "Signature": self.render_signature_column,
            "Name": self.render_name_column,
        }

    def render(self, node: IcoNode, column: PlanRendererColumn) -> Text:
        """Render specific column for node using registered column renderers."""
        return self._column_renderer[column](node)

    def render_flow_column(self, node: IcoNode) -> Text:
        """Render Flow column: icons, class name, and arguments."""
        args_info = self._render_node_args_info(node)
        text = self.flow_column_prefix or Text("")

        if not self.flow_includes_node_info:
            return text

        if (
            type(node) in [IcoOperator, IcoContextOperator]
            and not self.options.show_ico_operator
        ):
            return text + args_info

        if self.options.show_node_icons:
            icon = match_icon(self.options.node_icons, node)
            if icon:
                text += Text(icon)

        text += render_node_class(
            node,
            options=self.options,
            args_info=args_info,
        )

        if self.flow_column_postfix:
            text += self.flow_column_postfix

        return text

    def render_name_column(self, node: IcoNode) -> Text:
        """Render Name column with node name text."""
        return (
            Text(node.name or "", style=DescribeStyle.text.value)
            if self.show_name_column
            else Text("")
        )

    def render_signature_column(self, node: IcoNode) -> Text:
        """Render Signature column with type information and arrows."""
        if not self.show_signature_column:
            return Text("")

        signature = node.signature
        i = format_ico_type(signature.i) if signature.has_input else None
        c = format_ico_type(signature.c) if signature.has_context else None
        o = format_ico_type(signature.o) if signature.has_output else None
        prefix_length = 0

        if o is not None and not (i is None and c is None):
            if i is not None:
                prefix_length = same_prefix_length(i, o)
            if c is not None:
                prefix_length = min(prefix_length, same_prefix_length(c, o))

        i_text = Text(i, style=DescribeStyle.signature.value) if i is not None else None
        c_text = Text(c, style=DescribeStyle.signature.value) if c is not None else None
        o_text = Text(o, style=DescribeStyle.signature.value) if o is not None else None

        arrow = Text(" → ", style=DescribeStyle.dimmed.value)

        if self.options.signature_format in ("Full", "Input"):
            if i_text is not None:
                if prefix_length > 0 and self.options.signature_format == "Full":
                    i_text.stylize("dim", 0, prefix_length)
                text = i_text
            else:
                text = Text("()", style=DescribeStyle.signature.value)
        else:
            text = Text("")

        if c_text is not None and self.options.signature_format in ("Full", "Input"):
            if prefix_length > 0 and self.options.signature_format == "Full":
                c_text.stylize("dim", 0, prefix_length)
            text += Text(", ") + c_text

        if self.options.signature_format in ("Full"):
            text += arrow

        if self.options.signature_format in ("Full", "Output"):
            if o_text is not None:
                if prefix_length > 0 and self.options.signature_format == "Full":
                    o_text.stylize("dim", 0, prefix_length)
                text += o_text
            else:
                text += Text("()", style=DescribeStyle.signature.value)

        return text

    def _render_node_args_info(self, node: IcoNode) -> Text | None:
        if type(node) is IcoOperator:
            fn = cast(IcoOperator[Any, Any], node).fn
            return render_callable(fn, options=self.options)

        if type(node) is IcoContextOperator:
            fn = cast(IcoContextOperator[Any, Any, Any], node).fn
            return render_callable(fn, options=self.options)

        return None


class FlowTextRowRenderer(RowRenderer):
    def render_flow_column(self, node: IcoNode) -> Text:
        return self.flow_column_prefix or Text("")

    def render_signature_column(self, node: IcoNode) -> Text:
        return Text("")

    def render_type_column(self, node: IcoNode) -> Text:
        return Text("")

    def render_name_column(self, node: IcoNode) -> Text:
        return Text("")

    def render_state_column(self, node: IcoNode) -> Text:
        return Text("")


def same_prefix_length(s1: str, s2: str) -> int:
    length = min(len(s1), len(s2))
    for i in range(length):
        if s1[i] != s2[i]:
            return i
    return length

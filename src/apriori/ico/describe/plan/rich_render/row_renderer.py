from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from rich.text import Text

from apriori.ico.core.context_operator import IcoContextOperator
from apriori.ico.core.node import IcoNode
from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.describe.plan.options import (
    RenderColumn,
    RenderOptions,
)
from apriori.ico.describe.plan.rich_render.utils import (
    PlanStyle,
    get_state_color,
)
from apriori.ico.inspect.signature import format_ico_type, infer_signature
from apriori.ico.inspect.utils import (
    extract_class_display_name,
    extract_fn_display_name,
)


class RowRenderer:
    options: RenderOptions
    show_name_column: bool
    show_signature_column: bool
    show_type_column: bool
    show_state_column: bool
    flow_column_prefix: Text | None
    flow_column_postfix: Text | None
    flow_includes_node_info: bool = True

    def __init__(
        self,
        options: RenderOptions,
        show_name_column: bool = True,
        show_signature_column: bool = True,
        show_type_column: bool = True,
        show_state_column: bool = True,
        flow_column_prefix: Text | None = None,
        flow_column_postfix: Text | None = None,
        flow_includes_node_info: bool = True,
    ) -> None:
        self.options = options
        self.show_name_column = show_name_column
        self.show_signature_column = show_signature_column
        self.show_type_column = show_type_column
        self.show_state_column = show_state_column
        self.flow_column_prefix = flow_column_prefix
        self.flow_column_postfix = flow_column_postfix
        self.flow_includes_node_info = flow_includes_node_info

        self._column_renderer: dict[RenderColumn, Callable[[IcoNode], Text]] = {
            "Flow": self.render_flow_column,
            "Name": self.render_name_column,
            "Type": self.render_type_column,
            "Signature": self.render_signature_column,
            "State": self.render_state_column,
        }

    def render(self, node: IcoNode, column: RenderColumn) -> Text:
        return self._column_renderer[column](node)

    def render_flow_column(self, node: IcoNode) -> Text:
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
            icon = self.options.node_icons.get(type(node))
            if icon:
                text += Text(icon)

        text += self._render_node_class(node, args_info=args_info)

        if self.flow_column_postfix:
            text += self.flow_column_postfix

        return text

    def render_name_column(self, node: IcoNode) -> Text:
        return Text(node.name or "") if self.show_name_column else Text("")

    def render_type_column(self, node: IcoNode) -> Text:
        if not self.show_type_column:
            return Text("")

        if isinstance(node, IcoRuntimeNode):
            # Node has both operator and runtime bases and we want to show both
            operator_base = next(
                p for p in node.__class__.__bases__ if isinstance(p, IcoOperator)
            )
            runtime_base = next(
                p for p in node.__class__.__bases__ if isinstance(p, IcoRuntimeNode)
            )
            label = f"{operator_base.__name__}/{runtime_base.__name__}"
        else:
            # Node is an flow element with a single base
            label = node.__class__.__name__

        return Text(label, style=PlanStyle.dimmed.value)

    def render_signature_column(self, node: IcoNode) -> Text:
        if not self.show_signature_column:
            return Text("")

        signature = infer_signature(node)
        if signature.has_context:
            print("!")
        i = format_ico_type(signature.i) if signature.has_input else None
        c = format_ico_type(signature.c) if signature.has_context else None
        o = format_ico_type(signature.o) if signature.has_output else None
        prefix_length = 0

        if o is not None and not (i is None and c is None):
            if i is not None:
                prefix_length = same_prefix_length(i, o)
            if c is not None:
                prefix_length = min(prefix_length, same_prefix_length(c, o))

        i_text = Text(i, style=PlanStyle.signature.value) if i is not None else None
        c_text = Text(c, style=PlanStyle.signature.value) if c is not None else None
        o_text = Text(o, style=PlanStyle.signature.value) if o is not None else None

        arrow = Text(" → ", style=PlanStyle.dimmed.value)

        if self.options.signature_format in ("Full", "Input"):
            if i_text is not None:
                if prefix_length > 0 and self.options.signature_format == "Full":
                    i_text.stylize("dim", 0, prefix_length)
                text = i_text
            else:
                text = Text("()", style=PlanStyle.signature.value)
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
                text += Text("()", style=PlanStyle.signature.value)

        return text

    def render_state_column(self, node: IcoNode) -> Text:
        if not self.show_state_column:
            return Text("")
        if not isinstance(node, IcoRuntimeNode):
            return Text("")

        color = get_state_color(node.state)

        return Text(f"<{node.state}> ", style=color)

    def _render_node_class(
        self,
        node: IcoNode,
        args_info: Text | None = None,
    ) -> Text:
        display_name = Text(f"{type(node).__name__}(", style=PlanStyle.class_.value)

        if self.options.dim_ico_nodes:
            display_name.stylize("dim", 0, len(display_name))

        if args_info is not None:
            display_name += args_info

        display_name += Text(")", style=PlanStyle.class_.value)

        if self.options.dim_ico_nodes:
            display_name.stylize("dim", len(display_name) - 1, len(display_name))

        return display_name

    def _render_node_args_info(self, node: IcoNode) -> Text | None:
        if type(node) is IcoOperator:
            fn = cast(IcoOperator[Any, Any], node).fn
            return self.render_callable(fn)

        if type(node) is IcoContextOperator:
            fn = cast(IcoContextOperator[Any, Any, Any], node).fn
            return self.render_callable(fn)

        return None

    def render_callable(self, obj: object) -> Text:
        name = extract_fn_display_name(obj)  # type: ignore
        if name:
            return Text(name, style=PlanStyle.fn.value)

        if self.options.callable_format == "str()":
            return Text(str(obj), style=PlanStyle.meta.value)

        name = extract_class_display_name(obj)  # type: ignore
        if name:
            return Text(f"{name}()", style=PlanStyle.class_.value)

        return Text("Unknown", style=PlanStyle.dimmed.value)


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

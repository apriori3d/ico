from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from rich.text import Text

from apriori.ico.core.node import IcoNode
from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.describe.plan.rich_render.options import (
    RenderColumn,
    RenderOptions,
)
from apriori.ico.describe.plan.rich_render.utils import (
    PlanStyle,
    get_state_color,
)
from apriori.ico.inspect.signature import infer_signature
from apriori.ico.inspect.utils import (
    extract_class_display_name,
    extract_fn_display_name,
)


class RowRenderer:
    options: RenderOptions

    def __init__(self, options: RenderOptions) -> None:
        self.options = options

        self._column_renderer: dict[RenderColumn, Callable[[IcoNode], Text]] = {
            "Flow": self.render_flow_column,
            "Name": self.render_name_column,
            "Type": self.render_type_column,
            "Signature": self.render_signature_column,
            "State": self.render_state_column,
        }

    def __call__(self, node: IcoNode, column: RenderColumn) -> Text:
        return self._column_renderer[column](node)

    def render_flow_column(self, node: IcoNode) -> Text:
        args_info = self._render_node_args_info(node)

        if type(node) is IcoOperator and not self.options.show_ico_operator:
            return args_info or Text("", style=PlanStyle.dimmed.value)

        node = cast(IcoNode, node)
        return self._render_node_class(node, args_info=args_info)

    def render_name_column(self, node: IcoNode) -> Text:
        return Text(node.name or "", style=PlanStyle.text.value)

    def render_type_column(self, node: IcoNode) -> Text:
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
        signature = infer_signature(node)

        return Text(signature.format(), style=PlanStyle.signature.value)

    def render_state_column(self, node: IcoNode) -> Text:
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
            return self._render_callable(fn)
        return None

    def _render_callable(self, obj: object) -> Text:
        name = extract_fn_display_name(obj)  # type: ignore
        if name:
            return Text(name, style=PlanStyle.fn.value)

        if self.options.callable_format == "str()":
            return Text(str(obj), style=PlanStyle.meta.value)

        name = extract_class_display_name(obj)  # type: ignore
        if name:
            return Text(f"{name}()", style=PlanStyle.class_.value)

        return Text("Unknown", style=PlanStyle.dimmed.value)
